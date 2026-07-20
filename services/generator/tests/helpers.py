from __future__ import annotations

import json
from typing import Any


def envelope(message_type: str, seq: int, payload: dict[str, Any], timestamp_ms: int = 1_750_000_000_000) -> dict[str, Any]:
    return {
        "protocol": "pet-motion",
        "version": 1,
        "type": message_type,
        "seq": seq,
        "timestamp_ms": timestamp_ms,
        "payload": payload,
    }


def hello(seq: int = 0, session_id: str = "test-session") -> dict[str, Any]:
    return envelope(
        "hello",
        seq,
        {
            "session_id": session_id,
            "host": {"name": "unit-test-host", "version": "0.1.0", "pid": 1234},
            "requested_version": 1,
            "capabilities": ["window_surfaces"],
            "config": {
                "world_state_hz": 20,
                "plan_horizon_ms": 396,
                "plan_dt_ms": 33,
                "pet_width": 96,
                "pet_height": 96,
                "privacy": {
                    "screen_capture_enabled": False,
                    "keyboard_enabled": False,
                    "recording_enabled": False,
                },
            },
        },
    )


def world_state(
    seq: int = 1,
    *,
    session_id: str = "test-session",
    timestamp_ms: int = 1_750_000_001_000,
    click_id: str | None = None,
    allowed: bool = True,
    surface_id: str | None = "window:a:top",
    foot_x: float = 240.0,
    foot_y: float = 420.0,
) -> dict[str, Any]:
    clicks: list[dict[str, Any]] = []
    if click_id is not None:
        clicks.append(
            {
                "id": click_id,
                "button": "left",
                "x": foot_x,
                "y": foot_y - 40,
                "target": "pet",
                "timestamp_ms": timestamp_ms,
            }
        )
    pet: dict[str, Any] = {
        "x": foot_x - 48,
        "y": foot_y - 96,
        "width": 96,
        "height": 96,
        "foot_x": foot_x,
        "foot_y": foot_y,
        "vx": 0,
        "vy": 0,
        "facing": 1,
        "behavior": "idle",
        "visible": allowed,
        "user_dragging": False,
    }
    if surface_id is not None:
        pet["surface_id"] = surface_id
    return envelope(
        "world_state",
        seq,
        {
            "session_id": session_id,
            "coordinate_space": "physical_px",
            "displays": [
                {
                    "id": "display:1",
                    "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                    "work_area": {"x": 0, "y": 0, "width": 1920, "height": 1040},
                    "scale_factor": 1.25,
                    "is_primary": True,
                }
            ],
            "windows": [
                {
                    "id": "window:a",
                    "display_id": "display:1",
                    "bounds": {"x": 120, "y": 420, "width": 500, "height": 400},
                    "z_order": 0,
                    "visible": True,
                    "minimized": False,
                    "maximized": False,
                    "fullscreen": False,
                    "active": True,
                    "occluded": False,
                    "eligible": True,
                },
                {
                    "id": "window:b",
                    "display_id": "display:1",
                    "bounds": {"x": 720, "y": 300, "width": 420, "height": 500},
                    "z_order": 1,
                    "visible": True,
                    "minimized": False,
                    "maximized": False,
                    "fullscreen": False,
                    "active": False,
                    "occluded": False,
                    "eligible": True,
                },
            ],
            "surfaces": [
                {
                    "id": "window:a:top",
                    "kind": "window_top",
                    "display_id": "display:1",
                    "window_id": "window:a",
                    "x1": 120,
                    "x2": 620,
                    "y": 420,
                    "enabled": True,
                    "occluded": False,
                },
                {
                    "id": "window:b:top",
                    "kind": "window_top",
                    "display_id": "display:1",
                    "window_id": "window:b",
                    "x1": 720,
                    "x2": 1140,
                    "y": 300,
                    "enabled": True,
                    "occluded": False,
                },
                {
                    "id": "display:1:floor",
                    "kind": "work_area_floor",
                    "display_id": "display:1",
                    "x1": 0,
                    "x2": 1920,
                    "y": 1040,
                    "enabled": True,
                    "occluded": False,
                },
            ],
            "pet": pet,
            "cursor": {
                "x": 300,
                "y": 500,
                "left_down": False,
                "right_down": False,
                "middle_down": False,
                "over_pet": False,
            },
            "clicks": clicks,
            "scene": {"fullscreen_active": not allowed, "pet_allowed": allowed},
        },
        timestamp_ms,
    )


def line(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":")) + "\n"
