"""Generic 2D motion primitives for the Stage 5 renderer.

Five reusable primitives only: ``zoom``, ``pan``, ``slide``, ``fade`` and
``rotate``. Everything is evaluated as a pure function of time, so the same
render spec always produces the same frame.

Timeline rule (see ``render_spec`` for the contract):

* ``t < start``            -> hold the ``from`` value
* ``start <= t <= end``    -> interpolate with the easing curve
* ``t > end``              -> hold the ``to`` value
* ``duration <= 0``        -> snap to ``to`` at ``start``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agents.video_generation.render_spec import ANIMATION_TYPES, Animation, RenderElement


def linear(progress: float) -> float:
    return progress


def ease_in_out(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def ease_out(progress: float) -> float:
    return 1.0 - (1.0 - progress) ** 2


EASING_FUNCS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in_out": ease_in_out,
    "ease_out": ease_out,
}


def segment_progress(animation: Animation, t: float) -> float:
    """Normalized 0..1 progress of ``animation`` at time ``t`` (seconds)."""
    if animation.duration <= 0:
        return 1.0 if t >= animation.start else 0.0
    if t <= animation.start:
        return 0.0
    if t >= animation.start + animation.duration:
        return 1.0
    raw = (t - animation.start) / animation.duration
    easing = EASING_FUNCS.get(animation.easing, linear)
    return easing(max(0.0, min(1.0, raw)))


def _lerp(a: float, b: float, progress: float) -> float:
    return a + (b - a) * progress


def _num(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def zoom_scale(animation: Animation, t: float) -> float:
    """Scale multiplier contributed by a zoom animation."""
    if animation.type != "zoom":
        raise ValueError(f"zoom_scale called with animation type {animation.type!r}")
    progress = segment_progress(animation, t)
    return _lerp(_num(animation.from_value, 1.0), _num(animation.to_value, 1.0), progress)


def pan_position(
    animation: Animation, t: float, current: tuple[float, float]
) -> tuple[float, float]:
    """Absolute normalized center contributed by a pan animation.

    ``from``/``to`` are dicts with optional ``x``/``y`` keys; a component that
    is absent on both sides keeps ``current``.
    """
    if animation.type != "pan":
        raise ValueError(f"pan_position called with animation type {animation.type!r}")
    progress = segment_progress(animation, t)
    base_x, base_y = current
    start = _as_dict(animation.from_value)
    end = _as_dict(animation.to_value)
    from_x = _num(start.get("x"), base_x)
    to_x = _num(end.get("x"), from_x)
    from_y = _num(start.get("y"), base_y)
    to_y = _num(end.get("y"), from_y)
    return _lerp(from_x, to_x, progress), _lerp(from_y, to_y, progress)


def slide_offset(animation: Animation, t: float) -> tuple[float, float]:
    """Normalized offset added to the element position by a slide animation."""
    if animation.type != "slide":
        raise ValueError(f"slide_offset called with animation type {animation.type!r}")
    progress = segment_progress(animation, t)
    start = _as_dict(animation.from_value)
    end = _as_dict(animation.to_value)
    dx = _lerp(_num(start.get("dx"), 0.0), _num(end.get("dx"), 0.0), progress)
    dy = _lerp(_num(start.get("dy"), 0.0), _num(end.get("dy"), 0.0), progress)
    return dx, dy


def fade_opacity(animation: Animation, t: float) -> float:
    """Opacity contributed by a fade animation, clamped to 0..1."""
    if animation.type != "fade":
        raise ValueError(f"fade_opacity called with animation type {animation.type!r}")
    progress = segment_progress(animation, t)
    value = _lerp(_num(animation.from_value, 1.0), _num(animation.to_value, 1.0), progress)
    return max(0.0, min(1.0, value))


def rotate_angle(animation: Animation, t: float) -> float:
    """Absolute clockwise rotation in degrees contributed by a rotate animation."""
    if animation.type != "rotate":
        raise ValueError(f"rotate_angle called with animation type {animation.type!r}")
    progress = segment_progress(animation, t)
    return _lerp(_num(animation.from_value, 0.0), _num(animation.to_value, 0.0), progress)


@dataclass
class ElementState:
    """Resolved state of an element at one instant."""

    x: float
    y: float
    scale: float = 1.0  # multiplier over transform.scale
    rotation: float = 0.0  # absolute degrees, clockwise
    opacity: float = 1.0


def compute_state(element: RenderElement, t: float) -> ElementState:
    """Evaluate every animation of ``element`` at time ``t`` (seconds).

    Animations are applied in list order. ``pan``/``rotate``/``fade`` set
    absolute values, ``zoom`` sets a multiplier and ``slide`` adds an offset
    to the current position.
    """
    transform = element.transform
    state = ElementState(
        x=float(transform.x),
        y=float(transform.y),
        scale=1.0,
        rotation=float(transform.rotation),
        opacity=float(transform.opacity),
    )
    for animation in element.animations:
        if animation.type not in ANIMATION_TYPES:
            continue  # unsupported primitives are recorded and skipped by the renderer
        if animation.type == "zoom":
            state.scale *= zoom_scale(animation, t)
        elif animation.type == "pan":
            state.x, state.y = pan_position(animation, t, (state.x, state.y))
        elif animation.type == "slide":
            dx, dy = slide_offset(animation, t)
            state.x += dx
            state.y += dy
        elif animation.type == "fade":
            state.opacity = fade_opacity(animation, t)
        elif animation.type == "rotate":
            state.rotation = rotate_angle(animation, t)
    state.opacity = max(0.0, min(1.0, state.opacity))
    return state


def element_visible(element: RenderElement, t: float, scene_duration: float) -> bool:
    """True when ``t`` falls inside the element's timing window."""
    if t < element.timing.start:
        return False
    return t < element.timing.end(scene_duration)


def collect_unsupported_animations(element: RenderElement) -> list[str]:
    """Animation types the current motion engine cannot evaluate."""
    return [a.type for a in element.animations if a.type not in ANIMATION_TYPES]
