"""Asset resolution for the Stage 5 renderer.

The renderer never guesses file names. Every drawable thing references a
stable **asset id**, and this module maps ids to real files:

* **channel assets** - canonical, read-only, shared by every run. They live
  under ``assets/channels/`` (for example the channel mascot) and are resolved
  against the project root. The resolver never creates, copies or renames
  them.
* **scene assets** - generated per run. A bare id such as ``sc01_main`` maps to
  ``data/<run_id>/assets/sc01_main.<ext>`` with deterministic extension
  probing, so generated images need no manual renaming.
* **path assets** - an explicit relative/absolute path, tried against the
  project root and then against the run directory.

Resolution never writes to disk. When a file is absent the resolver reports
``status = "missing"`` and the renderer substitutes a deterministic in-memory
placeholder so one absent file cannot break a whole render.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image

CHANNEL_ASSET_PREFIX = "assets/channels/"
SCENE_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AssetKind(str, Enum):
    CHANNEL = "channel"
    SCENE = "scene"
    PATH = "path"


class AssetResolutionError(RuntimeError):
    """Raised when an asset reference cannot even be classified."""


@dataclass(frozen=True)
class ResolvedAsset:
    """Outcome of mapping an asset id onto the filesystem."""

    asset_id: str
    kind: AssetKind
    path: Path
    status: str  # "resolved" | "missing"
    scene_id: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def provenance(self) -> str:
        if not self.resolved:
            return "missing_placeholder"
        if self.kind is AssetKind.CHANNEL:
            return "channel_asset"
        return "existing_file"

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind.value,
            "path": str(self.path),
            "status": self.status,
            "scene_id": self.scene_id,
            "provenance": self.provenance,
        }


def _normalize(ref: str) -> str:
    return ref.replace("\\", "/").strip()


def is_channel_asset_ref(ref: str) -> bool:
    """True for canonical channel assets such as the mascot path."""
    return _normalize(ref).startswith(CHANNEL_ASSET_PREFIX)


def looks_like_path(ref: str) -> bool:
    """True when the reference contains path separators or a drive/URL prefix."""
    normalized = _normalize(ref)
    return "/" in normalized or normalized.startswith(".")


class AssetResolver:
    """Maps asset ids to files for one run. Read-only by construction."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        project_root: str | Path | None = None,
        scene_assets_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.scene_assets_dir = (
            Path(scene_assets_dir) if scene_assets_dir else self.run_dir / "assets"
        )

    def classify(self, asset_ref: str) -> AssetKind:
        ref = _normalize(str(asset_ref))
        if not ref:
            raise AssetResolutionError("empty asset reference")
        if is_channel_asset_ref(ref):
            return AssetKind.CHANNEL
        if looks_like_path(ref):
            return AssetKind.PATH
        return AssetKind.SCENE

    def resolve(self, asset_ref: str, *, scene_id: str = "") -> ResolvedAsset:
        """Resolve one asset id. Never writes to disk."""
        ref = _normalize(str(asset_ref))
        kind = self.classify(ref)

        if kind is AssetKind.CHANNEL:
            candidate = self.project_root / ref
            return self._make(ref, kind, candidate, scene_id)

        if kind is AssetKind.SCENE:
            found = self._probe_scene_asset(ref)
            if found is not None:
                return self._make(ref, kind, found, scene_id)
            default = self.scene_assets_dir / (ref + SCENE_ASSET_EXTENSIONS[0])
            return self._make(ref, kind, default, scene_id)

        # Explicit path: project root first, then the run dir, then assets/.
        raw = Path(ref)
        candidates = [
            raw if raw.is_absolute() else self.project_root / raw,
            self.run_dir / raw,
            self.scene_assets_dir / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return self._make(ref, kind, candidate, scene_id)
        return self._make(ref, kind, candidates[0], scene_id)

    def _probe_scene_asset(self, ref: str) -> Path | None:
        base = self.scene_assets_dir / ref
        candidates: list[Path] = []
        if base.suffix:
            candidates.append(base)
        else:
            candidates.extend(Path(str(base) + ext) for ext in SCENE_ASSET_EXTENSIONS)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _make(ref: str, kind: AssetKind, path: Path, scene_id: str) -> ResolvedAsset:
        return ResolvedAsset(
            asset_id=ref,
            kind=kind,
            path=path,
            status="resolved" if path.exists() else "missing",
            scene_id=scene_id,
        )

    def open_image(self, resolved: ResolvedAsset) -> Image.Image | None:
        """Open a resolved asset as RGBA, or ``None`` when it is missing."""
        if not resolved.resolved:
            return None
        try:
            with Image.open(resolved.path) as img:
                return img.convert("RGBA")
        except (OSError, ValueError):
            return None
