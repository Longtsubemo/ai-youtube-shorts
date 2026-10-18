"""Stage 4 (Scene Planning) validation.

Two layers, deliberately separate:

1. ``ScenePlan.validate()`` (``scene_models.py``) - the ORIGINAL structural
   checks (topic, scenes, narration, durations). Unchanged. Stage 5 also loads
   historical artifacts through it, so it must stay tolerant of pre-existing
   plans.

2. ``validate_scene_plan_generation()`` (this module) - the SDXL image-prompt
   gate, enforced by the Stage 4 pipeline itself (``run_scene_planning``).
   Every scene's ``generation_prompt`` must be a compact, SDXL-compatible
   background-image prompt.

Why compact: the image-generation model is SDXL-Lightning and its CLIP text
encoder accepts at most 77 tokens (including 2 special tokens). Long prompts
are silently truncated by CLIP, so validation FAILS CLEARLY instead - it never
truncates.

Token counting uses a lightweight, deterministic approximation
(``estimate_clip_tokens``) rather than the real CLIP tokenizer: the pipeline
stays offline, provider-independent and free of Diffusers/Transformers
dependencies. The estimate is deliberately slightly conservative (it
over-counts), so the 70-token hard ceiling trips BEFORE real CLIP truncation
could ever occur.
"""

from __future__ import annotations

import math
import re

from config.channel import CHANNEL

# --- budget -----------------------------------------------------------------

# Target: prompts should land here (approx. 25-55 real CLIP tokens).
GENERATION_PROMPT_TARGET_TOKENS = 55
# Hard ceiling: validation fails above this. Never exceeded, never truncated.
GENERATION_PROMPT_HARD_CEILING_TOKENS = 70

# --- forbidden visual-generation concepts -----------------------------------

# Concept name -> pattern (case-insensitive). These describe render modes,
# motion/video instructions, camera language, generated typography, people and
# the mascot - none of which belong in a static background-image prompt.
FORBIDDEN_PROMPT_CONCEPTS: dict[str, str] = {
    "photorealism": r"\bphoto[-\s]?real\w*|\bhyperreal\w*|\brealistic\b|\brealism\b",
    "3d_render": r"\b3[-\s]?d\b",
    "animation": r"\banimat|\bmotion\b|\banimated\b",
    "video": r"\bvideo\b|\bfootage\b|\bfilm(?:ing|ic)?\b|\blive[-\s]action\b",
    "camera_motion": (
        r"\bcamera\b|\bpush[-\s]?in\b|\btracking shot\b|\bdolly\b|"
        r"\bzoom(?:s|ed|ing)?\b|\bhandheld\b|\bcrane shot\b|\bwhip pan\b|"
        r"\bcinematic\b|\bsteadicam\b|\bparallax\b"
    ),
    "generated_text": (
        r"\btext\b|\btypography\b|\bheadline\b|\bcaption\b|\blabels?\b|"
        r"\blogos?\b|\bwatermark\b|\bsubtitles?\b|\breadable\b|\bsignage\b|"
        r"\bspell(?:ing|s)?\b"
    ),
    "characters": (
        r"\bpeople\b|\bpersons?\b|\bmen\b|\bwomen\b|\bwoman\b|\bman\b|"
        r"\bhuman\b|\bcharacters?\b|\bcrowd\b|\bfaces?\b|\bportrait\b|"
        r"\bavatar\b"
    ),
    "mascot": r"mascot|\bfox(?:es)?\b",
}

# Consecutive narration words that indicate the narration was copied into the
# generation prompt instead of kept in the separate narration field.
NARRATION_WINDOW_WORDS = 6


class ValidationIssue:
    def __init__(self, code=None, message=None, severity="error"):
        self.code = code
        self.message = message
        self.severity = severity

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


class ScenePlanningValidation:
    def __init__(self, issues=None, passed=True):
        self.issues = issues or []
        self.passed = passed

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self):
        return {"issues": [i.to_dict() for i in self.issues], "passed": self.passed}


def validate_production_plan(*args, **kwargs):
    return ScenePlanningValidation()


# --- token estimation -------------------------------------------------------

_CHARS_PER_TOKEN = 4


def estimate_clip_tokens(text: str) -> int:
    """Approximate the CLIP token count of ``text`` without a tokenizer.

    CLIP tokenizes on whitespace/punctuation first, then applies byte-pair
    encoding, so a short word costs ~1 token and longer words split into
    subwords. Each whitespace-delimited piece therefore contributes
    ``ceil(len(piece) / 4)`` tokens (minimum 1).

    The estimate intentionally over-counts typical English by a small margin
    so a prompt accepted here is safely below CLIP's real 77-token limit
    (77 includes 2 special tokens; the hard ceiling of 70 estimated text
    tokens leaves that headroom).
    """
    if not text or not text.strip():
        return 0
    return sum(
        max(1, math.ceil(len(piece) / _CHARS_PER_TOKEN)) for piece in text.split()
    )


# --- individual checks ------------------------------------------------------


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def contains_narration(prompt: str, narration: str, window: int = NARRATION_WINDOW_WORDS) -> bool:
    """True when ``prompt`` embeds narration it should not carry.

    Detection: ``window`` consecutive narration words appearing consecutively
    in the prompt (a copied run), or - for very short narrations - the whole
    narration appearing as a run. Common-word overlap alone never fires.
    """
    prompt_tokens = _norm_tokens(prompt)
    narration_tokens = _norm_tokens(narration)
    if not narration_tokens or not prompt_tokens:
        return False
    size = min(window, len(narration_tokens))
    for start in range(len(narration_tokens) - size + 1):
        span = narration_tokens[start : start + size]
        for offset in range(len(prompt_tokens) - size + 1):
            if prompt_tokens[offset : offset + size] == span:
                return True
    return False


def style_fragments(channel_style: str) -> list[str]:
    """Distinctive comma-separated fragments of the channel image style."""
    return [f.strip().lower() for f in (channel_style or "").split(",") if f.strip()]


def repeats_channel_style(prompt: str, channel_style: str) -> bool:
    """True when the prompt repeats the channel style block it must not carry."""
    normalized = " ".join((prompt or "").lower().split())
    if not normalized:
        return False
    return any(fragment in normalized for fragment in style_fragments(channel_style))


def check_generation_prompt(
    prompt,
    *,
    scene_id: str = "",
    narration: str = "",
    channel_style: str = "",
) -> list[ValidationIssue]:
    """Validate ONE generation prompt. Returns error and warning issues."""
    issues: list[ValidationIssue] = []
    label = f"scene {scene_id}" if scene_id else "scene"

    text = prompt if isinstance(prompt, str) else ""
    if not text.strip():
        issues.append(
            ValidationIssue(
                "gp01",
                f"{label}: generation_prompt is empty; every scene needs a "
                "compact image prompt",
            )
        )
        return issues

    estimated = estimate_clip_tokens(text)
    if estimated > GENERATION_PROMPT_HARD_CEILING_TOKENS:
        issues.append(
            ValidationIssue(
                "gp02",
                f"{label}: generation_prompt is ~{estimated} tokens, above the "
                f"hard ceiling of {GENERATION_PROMPT_HARD_CEILING_TOKENS} CLIP "
                "tokens; refusing (prompts are never truncated)",
            )
        )
    elif estimated > GENERATION_PROMPT_TARGET_TOKENS:
        issues.append(
            ValidationIssue(
                "gp03",
                f"{label}: generation_prompt is ~{estimated} tokens, above the "
                f"target of {GENERATION_PROMPT_TARGET_TOKENS} CLIP tokens; "
                "tighten it before it drifts toward the ceiling",
                severity="warning",
            )
        )

    for concept, pattern in FORBIDDEN_PROMPT_CONCEPTS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            if concept == "mascot":
                issues.append(
                    ValidationIssue(
                        "gp05",
                        f"{label}: generation_prompt mentions the mascot/fox; "
                        "the mascot is composited separately and never part of "
                        "the generated background",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        "gp04",
                        f"{label}: generation_prompt contains forbidden concept "
                        f"{concept!r}; background prompts must not request "
                        "photorealism, 3D, animation, video, camera moves, "
                        "generated text or people",
                    )
                )

    if contains_narration(text, narration):
        issues.append(
            ValidationIssue(
                "gp06",
                f"{label}: generation_prompt embeds the scene narration; "
                "narration stays in the narration field only",
            )
        )

    if repeats_channel_style(text, channel_style or CHANNEL.image_style):
        issues.append(
            ValidationIssue(
                "gp07",
                f"{label}: generation_prompt repeats the channel image style "
                "block; the image worker prepends the style later",
            )
        )

    return issues


def validate_scene_plan_generation(plan) -> ScenePlanningValidation:
    """Validate every scene's generation_prompt in a Stage 4 plan."""
    channel_style = getattr(plan, "channel_image_style", "") or CHANNEL.image_style
    issues: list[ValidationIssue] = []
    for scene in getattr(plan, "scenes", []) or []:
        issues.extend(
            check_generation_prompt(
                scene.generation_prompt,
                scene_id=scene.scene_id,
                narration=scene.narration,
                channel_style=channel_style,
            )
        )
    return ScenePlanningValidation(
        issues=issues,
        passed=not any(issue.severity == "error" for issue in issues),
    )
