"""Stage 5 tests: generic renderer primitives, playback, mascot appearance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agents.video_generation.assets import AssetResolver
from agents.video_generation.encoder import detect_ffmpeg, probe_video
from agents.video_generation.models import SceneClip  # noqa: F401  (contract import)
from agents.video_generation.placeholder_assets import synthesize_scene_assets
from agents.video_generation.render_spec import (
    Animation,
    ElementTiming,
    RenderElement,
    SceneRenderSpec,
    Transform,
)
from agents.video_generation.renderer import SceneRenderer
from agents.video_generation.translator import translate_scene_plan
from agents.video_generation.stage import load_scene_plan

CANONICAL_MASCOT = "assets/channels/finance_mascot/mascot.png"
REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_MASCOT_FILE = REPO_ROOT / CANONICAL_MASCOT
REAL_PLAN_PATH = REPO_ROOT / "data" / "e2e-20261005-stage4-live-01" / "scene_plan.json"

WIDTH, HEIGHT, FPS = 160, 288, 10
BG = (0, 0, 0)


def solid_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color + (255,)).save(path, format="PNG")
    return path


def make_renderer(run_dir: Path) -> SceneRenderer:
    return SceneRenderer(
        AssetResolver(run_dir, project_root=run_dir),
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
    )


def scene_spec(elements: list[RenderElement], duration: float = 1.0) -> SceneRenderSpec:
    return SceneRenderSpec(
        scene_id="sc_test",
        order=0,
        duration_seconds=duration,
        background=BG,
        elements=elements,
    )


def mask_of(frame: Image.Image) -> np.ndarray:
    """Boolean mask of pixels that differ from the scene background."""
    array = np.asarray(frame).astype(np.int16)
    return np.abs(array - np.array(BG)).max(axis=2) > 10


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def opaque_block_mascot() -> Image.Image:
    """Deterministic mascot-like test image: three opaque color blocks."""
    img = Image.new("RGBA", (320, 480), (0, 0, 0, 0))
    img.paste((220, 30, 30, 255), (20, 20, 300, 170))
    img.paste((30, 200, 60, 255), (20, 175, 300, 320))
    img.paste((40, 60, 230, 255), (20, 325, 300, 460))
    return img


# 7 - generic image element ---------------------------------------------------


def test_generic_image_element_renders_at_expected_position(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "frame.png", (100, 100), (220, 40, 40))
    element = RenderElement(
        id="main_visual",
        type="image",
        asset="frame",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
    )
    renderer = make_renderer(run_dir)
    frame = renderer.compose_frame(scene_spec([element]), 0.0)

    mask = mask_of(frame)
    assert mask.any(), "image element produced no visible pixels"
    x0, y0, x1, y1 = bbox(mask)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    assert abs(center_x - WIDTH / 2) <= 4
    assert abs(center_y - HEIGHT / 2) <= 4

    # the block color survives compositing
    colors = np.asarray(frame)[mask]
    assert np.abs(colors.mean(axis=0) - np.array([220, 40, 40])).max() <= 6


# 8 - generic text element ----------------------------------------------------


def test_generic_text_element_renders_words(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    element = RenderElement(
        id="text_main",
        type="text",
        text="GENERIC TEXT",
        transform=Transform(x=0.5, y=0.16),
        style={"size": 0.06, "color": "#FFFFFF", "align": "center"},
    )
    renderer = make_renderer(run_dir)
    frame = renderer.compose_frame(scene_spec([element]), 0.9)

    mask = mask_of(frame)
    assert mask.any(), "text element produced no pixels"
    _x0, y0, _x1, y1 = bbox(mask)
    # the text sits in the upper part of the frame
    assert y0 < HEIGHT * 0.35
    assert y1 < HEIGHT * 0.5
    bright = (np.asarray(frame).astype(np.int16).max(axis=2) > 200).sum()
    assert bright > 20


def test_empty_text_element_fails_structural_validation(tmp_path):
    from agents.video_generation.render_spec import RenderSpecError

    element = RenderElement(id="t", type="text", text="   ")
    with pytest.raises(RenderSpecError):
        element.validate()


# 9 - zoom --------------------------------------------------------------------


def test_zoom_animation_grows_the_element(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="z",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
        animations=[
            Animation(type="zoom", start=0.0, duration=1.0, from_value=1.0, to_value=2.0)
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element])
    start_mask = mask_of(renderer.compose_frame(scene, 0.0))
    end_mask = mask_of(renderer.compose_frame(scene, 0.9))

    assert int(end_mask.sum()) > int(start_mask.sum()) * 1.5


# 10 - pan --------------------------------------------------------------------


def test_pan_animation_moves_the_element(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="p",
        type="image",
        asset="white",
        transform=Transform(x=0.25, y=0.5, scale=0.4, fit="contain"),
        animations=[
            Animation(
                type="pan",
                start=0.0,
                duration=1.0,
                from_value={"x": 0.25, "y": 0.5},
                to_value={"x": 0.75, "y": 0.5},
            )
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element])

    start_x = bbox(mask_of(renderer.compose_frame(scene, 0.0)))[0]
    end_x = bbox(mask_of(renderer.compose_frame(scene, 0.9)))[0]
    assert end_x > start_x + WIDTH * 0.25


# 11 - slide ------------------------------------------------------------------


def test_slide_animation_enters_from_outside_the_frame(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="s",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
        animations=[
            Animation(
                type="slide",
                start=0.0,
                duration=1.0,
                from_value={"dx": -1.0, "dy": 0.0},
                to_value={"dx": 0.0, "dy": 0.0},
            )
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element])

    start_mask = mask_of(renderer.compose_frame(scene, 0.0))
    end_mask = mask_of(renderer.compose_frame(scene, 0.9))
    assert int(start_mask.sum()) == 0, "element should start outside the frame"
    assert int(end_mask.sum()) > 0, "element should slide into the frame"


# 12 - fade -------------------------------------------------------------------


def test_fade_animation_controls_opacity(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="f",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
        animations=[
            Animation(type="fade", start=0.0, duration=1.0, from_value=0.0, to_value=1.0)
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element])

    start_frame = np.asarray(renderer.compose_frame(scene, 0.0))
    mid_frame = np.asarray(renderer.compose_frame(scene, 0.5))
    end_frame = np.asarray(renderer.compose_frame(scene, 0.9))

    assert start_frame.max() == 0, "fade start must be invisible"
    assert mid_frame.max() < end_frame.max(), "fade must get brighter over time"
    assert end_frame.max() > 200


# 13 - rotate -----------------------------------------------------------------


def test_rotate_animation_turns_the_element(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "bar.png", (200, 100), (255, 255, 255))
    element = RenderElement(
        id="r",
        type="image",
        asset="bar",
        transform=Transform(x=0.5, y=0.5, scale=1.0, fit="contain"),
        animations=[
            Animation(type="rotate", start=0.0, duration=1.0, from_value=0.0, to_value=90.0)
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element])

    x0a, y0a, x1a, y1a = bbox(mask_of(renderer.compose_frame(scene, 0.0)))
    x0b, y0b, x1b, y1b = bbox(mask_of(renderer.compose_frame(scene, 0.9)))
    assert (x1a - x0a) > (y1a - y0a), "wide bar should start horizontal"
    assert (y1b - y0b) > (x1b - x0b), "bar should end nearly vertical"


def test_rotation_clockwise_convention():
    element = RenderElement(
        id="r",
        type="image",
        asset="a",
        animations=[
            Animation(type="rotate", start=0.0, duration=2.0, from_value=0.0, to_value=90.0)
        ],
    )
    from agents.video_generation.motion import compute_state

    assert compute_state(element, 0.0).rotation == 0.0
    assert compute_state(element, 1.0).rotation == pytest.approx(45.0)
    assert compute_state(element, 2.0).rotation == pytest.approx(90.0)


# 16 - playable scene output --------------------------------------------------


def test_rendered_scene_clip_is_playable(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="main_visual",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.6, fit="contain"),
        animations=[
            Animation(type="zoom", start=0.0, duration=1.0, from_value=1.0, to_value=1.1)
        ],
    )
    renderer = make_renderer(run_dir)
    out = run_dir / "video" / "scene.mp4"
    result = renderer.render_scene(
        scene_spec([element], duration=1.0),
        out,
        backend_info=_backend_info(),
    )

    assert out.exists()
    header = out.read_bytes()[:64]
    assert b"ftyp" in header, "not an MP4 container"
    probe = probe_video(out)
    assert probe["valid"]
    assert probe["frames"] == 10
    assert probe["fps"] == pytest.approx(FPS)
    assert (probe["width"], probe["height"]) == (WIDTH, HEIGHT)
    assert result["duration_seconds"] == pytest.approx(1.0)


def _backend_info() -> dict:
    from agents.video_generation.encoder import resolve_backend

    return resolve_backend("auto")


# determinism of composed frames ----------------------------------------------


def test_same_inputs_produce_identical_frames(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="d",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.6, fit="contain"),
        animations=[
            Animation(type="zoom", start=0.0, duration=1.0, from_value=1.0, to_value=1.2),
            Animation(type="rotate", start=0.0, duration=1.0, from_value=0.0, to_value=12.0),
        ],
    )
    scene = scene_spec([element])
    first = np.asarray(make_renderer(run_dir).compose_frame(scene, 0.4))
    second = np.asarray(make_renderer(run_dir).compose_frame(scene, 0.4))
    assert np.array_equal(first, second)


# 6 - missing mascot render-side fallback -------------------------------------


def test_missing_mascot_renders_with_placeholder_and_note(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    element = RenderElement(
        id="mascot",
        type="mascot",
        asset=CANONICAL_MASCOT,
        transform=Transform(x=0.5, y=0.7, scale=0.45, fit="contain"),
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element], duration=1.0)

    frame = renderer.compose_frame(scene, 0.5)
    assert mask_of(frame).any(), "placeholder should still render"
    assert renderer.notes.missing_assets
    assert renderer.notes.missing_assets[0]["asset_id"] == CANONICAL_MASCOT
    entry = renderer.asset_manifest[CANONICAL_MASCOT]
    assert entry["status"] == "missing"
    assert entry["provenance"] == "missing_placeholder"
    # nothing was written into the channel asset location
    assert not (run_dir / CANONICAL_MASCOT).exists()


# generic fallbacks for unsupported primitives --------------------------------


def test_unsupported_animation_is_recorded_and_ignored(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="weird",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
        animations=[
            Animation(type="teleport", start=0.0, duration=1.0, from_value=0, to_value=1)
        ],
    )
    renderer = make_renderer(run_dir)
    frame = renderer.compose_frame(scene_spec([element]), 0.5)
    assert mask_of(frame).any(), "generic fallback must still render the image"
    assert renderer.notes.unsupported_animations
    assert renderer.notes.unsupported_animations[0]["types"] == ["teleport"]


def test_unknown_element_type_is_skipped_but_scene_continues(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    weird = RenderElement(id="odd", type="talking_head", asset="white")
    good = RenderElement(
        id="ok",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
    )
    renderer = make_renderer(run_dir)
    frame = renderer.compose_frame(scene_spec([weird, good]), 0.5)
    assert mask_of(frame).any(), "supported element must still render"
    assert renderer.notes.skipped_elements
    assert renderer.notes.skipped_elements[0]["element_id"] == "odd"


def test_element_timing_hides_element_outside_its_window(tmp_path):
    run_dir = tmp_path / "run"
    solid_png(run_dir / "assets" / "white.png", (100, 100), (255, 255, 255))
    element = RenderElement(
        id="late",
        type="image",
        asset="white",
        transform=Transform(x=0.5, y=0.5, scale=0.5, fit="contain"),
        timing=ElementTiming(start=0.5, duration=0.0),
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element], duration=1.0)
    assert int(mask_of(renderer.compose_frame(scene, 0.2)).sum()) == 0
    assert int(mask_of(renderer.compose_frame(scene, 0.7)).sum()) > 0


# 25 - mascot appearance is preserved -----------------------------------------


def _color_stats(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return frame[mask].astype(np.float64).mean(axis=0)


def _check_mascot_colors(source: Image.Image, rendered: Image.Image) -> None:
    src = np.asarray(source)
    frame = np.asarray(rendered)

    src_opaque = src[:, :, 3] >= 250
    src_rgb = src[:, :, :3][src_opaque].astype(np.float64)

    rendered_mask = np.abs(frame.astype(np.int16) - np.array(BG)).max(axis=2) > 12
    assert rendered_mask.any()

    src_mean = src_rgb.mean(axis=0)
    out_mean = _color_stats(rendered_mask, frame)
    assert np.abs(src_mean - out_mean).max() <= 10.0, (
        f"mascot colors drifted: source {src_mean} vs rendered {out_mean}"
    )

    # the distinctive block colors must still be present after rendering
    block_colors = [(220, 30, 30), (30, 200, 60), (40, 60, 230)]
    pixels = frame[rendered_mask].astype(np.int16)
    for block in block_colors:
        distance = np.abs(pixels - np.array(block)).max(axis=1)
        assert int((distance <= 25).sum()) > 0, f"lost color {block}"


def test_mascot_colors_preserved_through_zoom_and_pan(tmp_path):
    run_dir = tmp_path / "run"
    mascot_dir = run_dir / "assets" / "channels" / "finance_mascot"
    mascot_dir.mkdir(parents=True)
    source = opaque_block_mascot()
    source.save(mascot_dir / "mascot.png", format="PNG")

    element = RenderElement(
        id="mascot",
        type="mascot",
        asset=CANONICAL_MASCOT,
        transform=Transform(x=0.5, y=0.6, scale=0.5, fit="contain"),
        animations=[
            Animation(type="zoom", start=0.0, duration=1.0, from_value=1.0, to_value=1.15),
            Animation(
                type="pan",
                start=0.0,
                duration=1.0,
                from_value={"x": 0.4, "y": 0.6},
                to_value={"x": 0.6, "y": 0.6},
            ),
            Animation(type="rotate", start=0.0, duration=1.0, from_value=0.0, to_value=4.0),
        ],
    )
    renderer = make_renderer(run_dir)
    scene = scene_spec([element], duration=1.0)

    start = renderer.compose_frame(scene, 0.0)
    end = renderer.compose_frame(scene, 0.9)
    _check_mascot_colors(source, start)
    _check_mascot_colors(source, end)


@pytest.mark.skipif(
    not CANONICAL_MASCOT_FILE.exists(), reason="canonical mascot asset not present"
)
def test_canonical_mascot_colors_preserved(tmp_path):
    source = Image.open(CANONICAL_MASCOT_FILE).convert("RGBA")
    run_dir = tmp_path / "run"
    element = RenderElement(
        id="mascot",
        type="mascot",
        asset=CANONICAL_MASCOT,
        transform=Transform(x=0.5, y=0.6, scale=0.5, fit="contain"),
        animations=[
            Animation(type="zoom", start=0.0, duration=1.0, from_value=1.0, to_value=1.12),
            Animation(
                type="pan",
                start=0.0,
                duration=1.0,
                from_value={"x": 0.45, "y": 0.6},
                to_value={"x": 0.55, "y": 0.6},
            ),
        ],
    )
    renderer = SceneRenderer(
        AssetResolver(run_dir, project_root=REPO_ROOT),
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
    )
    scene = scene_spec([element], duration=1.0)
    rendered = renderer.compose_frame(scene, 0.9)

    src = np.asarray(source)
    src_rgb = src[:, :, :3][src[:, :, 3] >= 250].astype(np.float64)
    frame = np.asarray(rendered)
    rendered_mask = np.abs(frame.astype(np.int16) - np.array(BG)).max(axis=2) > 12
    if src_rgb.size == 0 or not rendered_mask.any():
        pytest.skip("canonical mascot has no opaque pixels to compare")
    assert np.abs(src_rgb.mean(axis=0) - _color_stats(rendered_mask, frame)).max() <= 10.0


def test_renderer_source_contains_no_color_transforms():
    """The renderer must never recolor, filter or remap image channels."""
    forbidden = [
        "ImageEnhance",
        "ImageFilter",
        "ImageOps",
        "colorize",
        'convert("HSV"',
        "convert('HSV'",
        'convert("LAB"',
        "convert('LAB'",
        "convert(\"L\")",
        "point(lambda",
        "split()[:3]",
    ]
    for path in Path(__file__).resolve().parent.parent.joinpath(
        "agents", "video_generation"
    ).glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains color transform {token!r}"


# real plan integration -------------------------------------------------------


@pytest.mark.skipif(not REAL_PLAN_PATH.exists(), reason="real Stage 4 artifact not present")
def test_real_plan_renders_every_scene_from_test_assets(tmp_path):
    plan = load_scene_plan(REAL_PLAN_PATH)
    spec = translate_scene_plan(plan, width=WIDTH, height=HEIGHT, fps=FPS)
    run_dir = tmp_path / "run"
    written = synthesize_scene_assets(run_dir, _scene_asset_ids(spec))
    assert len(written) == 7

    renderer = SceneRenderer(
        AssetResolver(run_dir, project_root=tmp_path), width=WIDTH, height=HEIGHT, fps=FPS
    )
    for scene in spec.ordered_scenes:
        frame = renderer.compose_frame(scene, scene.duration_seconds * 0.5)
        assert frame.size == (WIDTH, HEIGHT)
        assert mask_of(frame).any(), f"{scene.scene_id} rendered nothing"

    assert renderer.notes.missing_assets  # canonical mascot is not present
    assert not (run_dir / CANONICAL_MASCOT).exists()


def _scene_asset_ids(spec) -> list[str]:
    from agents.video_generation.translator import collect_scene_asset_ids

    return collect_scene_asset_ids(spec)


def test_detect_ffmpeg_reports_explicitly_when_absent():
    found = detect_ffmpeg()
    assert found is None or Path(found).exists()
