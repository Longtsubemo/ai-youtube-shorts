"""Import and validate Colab-generated SDXL scene images for one pipeline run.

The Colab SDXL worker returns a ZIP of generated scene backgrounds plus a
``result.json`` manifest. This module maps those images back onto the run's
image-job contract and validates them **before** writing anything:

- the expected assets come from ``data/<run_id>/image_jobs.json`` (the real
  ``image-jobs/1.0`` contract) - no hardcoded scene list, no invented fields;
- the source ZIP and the image-job JSON are only ever read;
- every candidate image must be a readable PNG of the expected size;
- the whole archive is validated before any file is copied, so an invalid,
  duplicate, missing or unsafe archive can never cause a partial import;
- existing files are never overwritten unless ``--overwrite`` is passed;
- destination paths are confined to ``data/<run_id>/assets/``.

CLI::

    python -m agents.image_jobs.import_assets --run-id <run_id> --zip <path>

    python -m agents.image_jobs.import_assets \
        --run-id <run_id> --zip <path> --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from agents.image_jobs.jobs import JOB_SCHEMA_VERSION, ImageJobBatch

REPORT_SCHEMA_VERSION = "asset-import/1.0"
REPORT_FILENAME = "asset_import.json"
RESULT_MANIFEST_NAME = "result.json"
EXPECTED_IMAGE_FORMAT = "PNG"
DEFAULT_IMAGE_SIZE = (768, 1344)

# A bare, safe scene asset id such as ``sc01_main``. No separators, no drive
# letters, no leading dot - anything else is rejected as unsafe.
SAFE_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

ASSET_STATUSES_ERROR = frozenset(
    {"missing", "duplicate", "corrupt", "invalid_dimension", "conflict", "unsafe"}
)


class AssetImportError(ValueError):
    """Raised for structural input problems the importer refuses to proceed on."""

    def __init__(self, message: str, code: str = "im00") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ImportIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class AssetImportReport:
    run_id: str = ""
    zip_path: str = ""
    jobs_path: str = ""
    dry_run: bool = False
    overwrite: bool = False
    expected_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    status: str = "failed"
    assets: list[dict] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    issues: list[ImportIssue] = field(default_factory=list)
    report_path: str = ""

    @property
    def errors(self) -> list[ImportIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ImportIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return self.status in ("imported", "validated")

    def to_dict(self) -> dict:
        counts = Counter(asset["status"] for asset in self.assets)
        return {
            "schema": REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "zip_path": self.zip_path,
            "jobs_path": self.jobs_path,
            "dry_run": self.dry_run,
            "overwrite": self.overwrite,
            "expected_size": {
                "width": self.expected_size[0],
                "height": self.expected_size[1],
            },
            "status": self.status,
            "counts": {
                "expected": len(self.assets),
                "written": counts.get("imported", 0) + counts.get("overwritten", 0),
                "missing": counts.get("missing", 0),
                "duplicates": counts.get("duplicate", 0),
                "corrupt": counts.get("corrupt", 0),
                "invalid_dimension": counts.get("invalid_dimension", 0),
                "conflicts": counts.get("conflict", 0),
                "unsafe": counts.get("unsafe", 0),
                "unexpected": len(self.unexpected),
            },
            "assets": list(self.assets),
            "unexpected": list(self.unexpected),
            "issues": [i.to_dict() for i in self.issues],
        }


def _norm_member(name: str) -> str:
    """Normalize a ZIP member name to forward slashes without a leading ``./``."""
    normalized = str(name).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_image_bytes(data: bytes, expected_size: tuple[int, int]) -> tuple[str | None, str]:
    """Return ``(None, "")`` for a valid PNG, else ``(status, detail)``."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many exception types
        return "corrupt", f"not a readable image ({type(exc).__name__}: {exc})"

    try:
        with Image.open(io.BytesIO(data)) as img:
            image_format = img.format
            size = (img.width, img.height)
    except Exception as exc:  # noqa: BLE001
        return "corrupt", f"image could not be reopened ({type(exc).__name__}: {exc})"

    if image_format != EXPECTED_IMAGE_FORMAT:
        return "corrupt", f"image format is {image_format!r}, expected {EXPECTED_IMAGE_FORMAT}"

    if size != (int(expected_size[0]), int(expected_size[1])):
        return (
            "invalid_dimension",
            f"image is {size[0]}x{size[1]}, expected "
            f"{int(expected_size[0])}x{int(expected_size[1])}",
        )

    return None, ""


def _load_batch(jobs_path: Path) -> ImageJobBatch:
    try:
        raw = jobs_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssetImportError(
            f"cannot read image-job JSON {jobs_path}: {exc}", "im01"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssetImportError(
            f"image-job JSON is not valid JSON: {jobs_path} ({exc})", "im02"
        ) from exc

    if not isinstance(data, dict):
        raise AssetImportError(
            f"image-job JSON must be a JSON object: {jobs_path}", "im02"
        )

    if data.get("schema") != JOB_SCHEMA_VERSION:
        raise AssetImportError(
            f"unsupported image-job schema {data.get('schema')!r}; "
            f"expected {JOB_SCHEMA_VERSION}",
            "im03",
        )

    batch = ImageJobBatch.from_dict(data)
    if not batch.jobs:
        raise AssetImportError(
            f"image-job JSON contains no jobs: {jobs_path}", "im16"
        )
    return batch


def _public_assets(expected: list[dict]) -> list[dict]:
    public: list[dict] = []
    for entry in expected:
        item = {key: value for key, value in entry.items() if not key.startswith("_")}
        public.append(item)
    return public


def _write_report(report: AssetImportReport, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    report.report_path = str(report_path)


def _finalize(report: AssetImportReport, expected: list[dict], run_dir: Path) -> AssetImportReport:
    report.assets = _public_assets(expected)
    if not report.dry_run:
        _write_report(report, run_dir)
    return report


def _resolve_expected_assets(
    batch: ImageJobBatch, run_id: str, project_root: Path, assets_dir: Path, report: AssetImportReport
) -> tuple[list[dict], bool]:
    """Build the expected-asset plan. Returns ``(entries, contract_error)``."""
    rel_assets = Path("data") / run_id / "assets"
    expected: list[dict] = []
    seen_ids: dict[str, int] = {}
    contract_error = False

    for index, job in enumerate(batch.jobs, start=1):
        scene_id = (job.scene_id or "").strip()
        asset_id = (job.asset_id or "").strip()
        output_path = (job.output_path or "").strip()

        def _unsafe(entry_asset: str, entry_scene: str, code: str, message: str) -> None:
            nonlocal contract_error
            contract_error = True
            report.issues.append(ImportIssue(code, f"job {index}: {message}"))
            expected.append(
                {
                    "scene_id": entry_scene,
                    "asset_id": entry_asset,
                    "destination": "",
                    "member": None,
                    "status": "unsafe",
                    "detail": message,
                }
            )

        if not scene_id or not asset_id or not output_path:
            _unsafe(
                asset_id,
                scene_id,
                "im16",
                "missing scene_id/asset_id/output_path in the image-job contract",
            )
            continue

        if not SAFE_ASSET_ID_RE.match(asset_id):
            _unsafe(asset_id, scene_id, "im05", f"unsafe asset_id {asset_id!r}")
            continue

        if asset_id in seen_ids:
            _unsafe(
                asset_id,
                scene_id,
                "im07",
                f"duplicate asset_id {asset_id!r} (also used by job {seen_ids[asset_id]})",
            )
            continue

        op = Path(output_path.replace("\\", "/"))
        unsafe_reason = ""
        if op.is_absolute() or op.drive:
            unsafe_reason = f"output_path {output_path!r} is absolute"
        elif ".." in op.parts:
            unsafe_reason = f"output_path {output_path!r} contains '..'"
        elif op.parent != rel_assets:
            unsafe_reason = (
                f"output_path {output_path!r} is not directly under "
                f"{rel_assets.as_posix()}/"
            )
        elif op.suffix != ".png":
            unsafe_reason = f"output_path {output_path!r} must end in .png"
        elif op.stem != asset_id:
            unsafe_reason = (
                f"output_path {output_path!r} does not match asset_id {asset_id!r}"
            )

        dest = project_root / op if not unsafe_reason else None
        if dest is not None and not _is_within(dest, assets_dir):
            unsafe_reason = f"output_path {output_path!r} escapes {assets_dir}"

        if unsafe_reason:
            _unsafe(asset_id, scene_id, "im06", unsafe_reason)
            continue

        seen_ids[asset_id] = index
        expected.append(
            {
                "scene_id": scene_id,
                "asset_id": asset_id,
                "destination": str(dest),
                "member": None,
                "status": "planned",
                "_dest": dest,
                "_data": None,
                "_overwrite": False,
            }
        )

    return expected, contract_error


def import_generated_assets(
    zip_path: str | Path,
    run_id: str,
    *,
    project_root: str | Path = ".",
    jobs_path: str | Path | None = None,
    expected_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    overwrite: bool = False,
    dry_run: bool = False,
) -> AssetImportReport:
    """Validate (and optionally import) one Colab asset ZIP into a run.

    Reads ``image_jobs.json`` and the ZIP read-only. Writes images only into
    ``data/<run_id>/assets/`` and, for non-dry-runs, the report under the run.
    Raises :class:`AssetImportError` only for structural input problems
    (unreadable/mismatched image-job JSON or a missing/invalid ZIP); asset-level
    problems are collected into the returned report.
    """
    project_root = Path(project_root)
    run_id = str(run_id)
    run_dir = project_root / "data" / run_id
    assets_dir = run_dir / "assets"
    zip_path = Path(zip_path)
    jobs_path = Path(jobs_path) if jobs_path is not None else (run_dir / "image_jobs.json")
    expected_size = (int(expected_size[0]), int(expected_size[1]))

    report = AssetImportReport(
        run_id=run_id,
        zip_path=str(zip_path),
        jobs_path=str(jobs_path),
        dry_run=dry_run,
        overwrite=overwrite,
        expected_size=expected_size,
    )

    batch = _load_batch(jobs_path)
    if batch.run_id and batch.run_id != run_id:
        raise AssetImportError(
            f"image-job run_id {batch.run_id!r} does not match requested run {run_id!r}",
            "im04",
        )

    expected, contract_error = _resolve_expected_assets(
        batch, run_id, project_root, assets_dir, report
    )

    if contract_error:
        report.status = "failed"
        return _finalize(report, expected, run_dir)

    if not zip_path.is_file():
        raise AssetImportError(f"source ZIP not found: {zip_path}", "im08")
    if not zipfile.is_zipfile(zip_path):
        raise AssetImportError(f"source is not a valid ZIP archive: {zip_path}", "im09")

    accepted: dict[str, str] = {}
    for entry in expected:
        accepted[f"{entry['asset_id']}.png"] = entry["asset_id"]
        accepted[f"assets/{entry['asset_id']}.png"] = entry["asset_id"]

    with zipfile.ZipFile(zip_path) as archive:
        raw_names = archive.namelist()
        name_counts = Counter(raw_names)
        dup_names = sorted(
            {name for name, count in name_counts.items() if count > 1 and not name.endswith("/")}
        )
        members = [name for name in raw_names if not name.endswith("/")]

        candidates: dict[str, list[str]] = defaultdict(list)
        unexpected: list[str] = []
        for raw in members:
            normalized = _norm_member(raw)
            if normalized == RESULT_MANIFEST_NAME:
                continue
            asset_id = accepted.get(normalized)
            if asset_id is None:
                unexpected.append(raw)
            else:
                candidates[asset_id].append(normalized)

        report.unexpected = sorted(unexpected)
        if report.unexpected:
            report.issues.append(
                ImportIssue(
                    "im12",
                    f"unexpected ZIP member(s): {', '.join(report.unexpected)}",
                    severity="warning",
                )
            )
        for name in dup_names:
            report.issues.append(
                ImportIssue("im11", f"duplicate ZIP member name: {name}")
            )

        for entry in expected:
            found = candidates.get(entry["asset_id"], [])
            if not found:
                entry["status"] = "missing"
            elif len(found) > 1 or found[0] in dup_names:
                entry["status"] = "duplicate"
                entry["detail"] = f"ZIP entries {sorted(found)} map to one asset"
            else:
                entry["member"] = found[0]
                entry["status"] = "planned"

        for entry in expected:
            if entry["status"] != "planned":
                continue
            data = archive.read(entry["member"])
            kind, detail = _validate_image_bytes(data, expected_size)
            if kind is not None:
                entry["status"] = kind
                entry["detail"] = detail
            else:
                entry["_data"] = data

        for entry in expected:
            if entry["status"] != "planned":
                continue
            if entry["_dest"].exists():
                if overwrite:
                    entry["_overwrite"] = True
                else:
                    entry["status"] = "conflict"
                    entry["detail"] = (
                        f"{entry['destination']} already exists "
                        "(use --overwrite to replace it)"
                    )

    errors_present = bool(dup_names) or any(
        entry["status"] in ASSET_STATUSES_ERROR for entry in expected
    )

    if errors_present:
        report.status = "failed"
    elif dry_run:
        for entry in expected:
            if entry["status"] == "planned":
                entry["status"] = "validated"
        report.status = "validated"
    else:
        assets_dir.mkdir(parents=True, exist_ok=True)
        for entry in expected:
            entry["_dest"].write_bytes(entry["_data"])
            entry["status"] = "overwritten" if entry["_overwrite"] else "imported"
        report.status = "imported"

    for entry in expected:
        label = f"scene {entry['scene_id']} ({entry['asset_id']})"
        status = entry["status"]
        if status == "missing":
            report.issues.append(ImportIssue("im10", f"{label}: no image found in ZIP"))
        elif status == "duplicate":
            report.issues.append(
                ImportIssue("im11", f"{label}: {entry.get('detail', 'duplicate ZIP entry')}")
            )
        elif status == "corrupt":
            report.issues.append(ImportIssue("im13", f"{label}: {entry.get('detail', 'corrupt image')}"))
        elif status == "invalid_dimension":
            report.issues.append(
                ImportIssue("im14", f"{label}: {entry.get('detail', 'invalid dimensions')}")
            )
        elif status == "conflict":
            report.issues.append(ImportIssue("im15", f"{label}: {entry.get('detail', 'conflict')}"))

    return _finalize(report, expected, run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agents.image_jobs.import_assets",
        description=(
            "Validate and import a Colab SDXL asset ZIP into one run's "
            "data/<run_id>/assets/ directory."
        ),
    )
    parser.add_argument("--run-id", required=True, help="run id (data/<run_id>)")
    parser.add_argument("--zip", required=True, type=Path, help="source asset ZIP")
    parser.add_argument(
        "--jobs",
        type=Path,
        default=None,
        help="image-job JSON (default: data/<run_id>/image_jobs.json)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="project root containing data/ (default: current directory)",
    )
    parser.add_argument("--expected-width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--expected-height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing asset files after full validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate only; write no files, not even the report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        report = import_generated_assets(
            args.zip,
            args.run_id,
            project_root=args.project_root,
            jobs_path=args.jobs,
            expected_size=(args.expected_width, args.expected_height),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except AssetImportError as exc:
        print(f"[image_jobs.import] input error [{exc.code}]: {exc.message}")
        return 2

    print(f"[image_jobs.import] run_id={report.run_id} zip={report.zip_path}")
    for issue in report.warnings:
        print(f"[image_jobs.import] warning [{issue.code}] {issue.message}")
    for issue in report.errors:
        print(f"[image_jobs.import] error [{issue.code}] {issue.message}")

    counts = Counter(asset["status"] for asset in report.assets)
    written = counts.get("imported", 0) + counts.get("overwritten", 0)
    print(
        f"[image_jobs.import] status={report.status} expected={len(report.assets)} "
        f"written={written} missing={counts.get('missing', 0)} "
        f"duplicates={counts.get('duplicate', 0)} corrupt={counts.get('corrupt', 0)} "
        f"invalid_dimension={counts.get('invalid_dimension', 0)} "
        f"conflicts={counts.get('conflict', 0)} unexpected={len(report.unexpected)}"
    )
    if report.report_path:
        print(f"[image_jobs.import] report written to {report.report_path}")
    else:
        print("[image_jobs.import] dry run: no files written")

    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
