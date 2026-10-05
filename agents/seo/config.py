"""Topic-selection configuration.

Reads config/topic_selection.yaml into typed dataclasses so the scorer
and selector never hard-code weights, exploration knobs, pillars, or
topic types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.channel import CHANNEL, ChannelConfig

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only without deps
    yaml = None


class TopicSelectionConfigError(ValueError):
    """Raised when the config file is missing or malformed."""


SCORE_KEYS = (
    "interest",
    "trend_momentum",
    "competition",
    "curiosity",
    "explainer_potential",
    "visual_potential",
    "audience_fit",
)

DEFAULT_CONFIG_PATH = Path("config") / "topic_selection.yaml"


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float]
    competition_transformation: str
    historical_enabled: bool
    historical_weight: float

    def to_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "competition_transformation": self.competition_transformation,
            "historical_enabled": self.historical_enabled,
            "historical_weight": self.historical_weight,
        }


@dataclass(frozen=True)
class SelectionConfig:
    exploration_fraction: float
    seed: int
    top_n: int
    coverage_penalty: float

    def to_dict(self) -> dict:
        return {
            "exploration_fraction": self.exploration_fraction,
            "seed": self.seed,
            "top_n": self.top_n,
            "coverage_penalty": self.coverage_penalty,
        }


@dataclass(frozen=True)
class PillarInfo:
    """Editorial definition of a single content pillar."""

    pillar_id: str
    label: str
    focus: str
    guidance: str

    def to_dict(self) -> dict:
        return {
            "pillar_id": self.pillar_id,
            "label": self.label,
            "focus": self.focus,
            "guidance": self.guidance,
        }


@dataclass(frozen=True)
class TopicSelectionConfig:
    version: int
    scoring: ScoringConfig
    selection: SelectionConfig
    pillars: tuple[str, ...]
    topic_types: tuple[str, ...]
    pillar_catalog: tuple[PillarInfo, ...] = ()
    channel: ChannelConfig = CHANNEL

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "scoring": self.scoring.to_dict(),
            "selection": self.selection.to_dict(),
            "pillars": list(self.pillars),
            "topic_types": list(self.topic_types),
            "pillar_catalog": [pillar.to_dict() for pillar in self.pillar_catalog],
            "channel": self.channel.to_dict(),
        }

    def pillar_info(self, pillar_id: str) -> PillarInfo:
        """Return the editorial definition for ``pillar_id``.

        Raises if the pillar is unknown so a candidate can never be emitted
        for a pillar that has no defined editorial meaning.
        """
        for pillar in self.pillar_catalog:
            if pillar.pillar_id == pillar_id:
                return pillar
        raise TopicSelectionConfigError(f"pillar not in catalog: {pillar_id}")


def _require_mapping(node: object, key: str) -> dict:
    value = node.get(key) if isinstance(node, dict) else None
    if not isinstance(value, dict):
        raise TopicSelectionConfigError(f"config missing or invalid section: {key}")
    return value


def _number(node: object, section: str, key: str) -> float:
    value = node.get(key) if isinstance(node, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TopicSelectionConfigError(f"{section}.{key} must be a number, got {value!r}") from None


def _in_unit_range(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise TopicSelectionConfigError(f"{name} must be in [0,1], got {value}")
    return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TopicSelectionConfig:
    if yaml is None:
        raise TopicSelectionConfigError(
            "PyYAML is required to load topic-selection config. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    path = Path(path)
    if not path.is_file():
        raise TopicSelectionConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    scoring = _require_mapping(raw, "scoring")
    weights_node = _require_mapping(scoring, "weights")
    missing = [key for key in SCORE_KEYS if key not in weights_node]
    if missing:
        raise TopicSelectionConfigError(f"scoring.weights missing keys: {', '.join(missing)}")
    weights = {key: _in_unit_range(_number(weights_node, "weights", key), "weights." + key) for key in SCORE_KEYS}

    competition = _require_mapping(scoring, "competition")
    transformation = competition.get("transformation", "inverse")
    if transformation not in ("inverse",):
        raise TopicSelectionConfigError(f"unknown competition transformation: {transformation}")

    historical = _require_mapping(scoring, "historical")

    selection = _require_mapping(raw, "selection")
    exploration_fraction = _in_unit_range(_number(selection, "selection", "exploration_fraction"), "exploration_fraction")
    seed = int(selection.get("seed", 42))
    top_n = int(selection.get("top_n", 3))
    coverage_penalty = _in_unit_range(_number(selection, "selection", "coverage_penalty"), "coverage_penalty")
    if top_n < 1:
        raise TopicSelectionConfigError(f"selection.top_n must be >= 1, got {top_n}")

    pillars = tuple(raw.get("pillars") or ())
    topic_types = tuple(raw.get("topic_types") or ())
    if not pillars:
        raise TopicSelectionConfigError("config lists no pillars")
    if not topic_types:
        raise TopicSelectionConfigError("config lists no topic_types")

    catalog_node = _require_mapping(raw, "pillar_catalog")
    pillar_catalog: list[PillarInfo] = []
    for pillar_id in pillars:
        entry = catalog_node.get(pillar_id)
        if not isinstance(entry, dict):
            raise TopicSelectionConfigError(f"pillar_catalog missing entry for pillar: {pillar_id}")
        label = entry.get("label")
        focus = entry.get("focus")
        guidance = entry.get("guidance")
        if not isinstance(label, str) or not label.strip():
            raise TopicSelectionConfigError(f"pillar_catalog.{pillar_id}.label must be a non-empty string")
        if not isinstance(focus, str) or not focus.strip():
            raise TopicSelectionConfigError(f"pillar_catalog.{pillar_id}.focus must be a non-empty string")
        if not isinstance(guidance, str) or not guidance.strip():
            raise TopicSelectionConfigError(f"pillar_catalog.{pillar_id}.guidance must be a non-empty string")
        pillar_catalog.append(
            PillarInfo(
                pillar_id=pillar_id,
                label=label.strip(),
                focus=focus.strip(),
                guidance=" ".join(guidance.split()),
            )
        )
    extra_catalog = sorted(set(catalog_node) - set(pillars))
    if extra_catalog:
        raise TopicSelectionConfigError(
            "pillar_catalog defines pillars that are not listed under 'pillars': " + ", ".join(extra_catalog)
        )

    return TopicSelectionConfig(
        version=int(raw.get("version", 1)),
        scoring=ScoringConfig(
            weights=weights,
            competition_transformation=transformation,
            historical_enabled=bool(historical.get("enabled", True)),
            historical_weight=_in_unit_range(_number(historical, "historical", "weight"), "historical.weight"),
        ),
        selection=SelectionConfig(
            exploration_fraction=exploration_fraction,
            seed=seed,
            top_n=top_n,
            coverage_penalty=coverage_penalty,
        ),
        pillars=pillars,
        topic_types=topic_types,
        pillar_catalog=tuple(pillar_catalog),
    )