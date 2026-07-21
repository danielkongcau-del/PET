"""Stable backend boundary shared by procedural and future learned models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from .state import WorldState


@dataclass(frozen=True, slots=True)
class MotionPoint:
    t_ms: int
    dx: float
    dy: float
    vx: float
    vy: float
    facing: int
    lean: float
    squash: float
    bob: float
    expression: str
    bone_rotations: tuple[float, ...] | None = None
    facial_params: dict[str, float] | None = None
    # 3D skeletal pose (mutually exclusive with bone_rotations)
    root_translation: tuple[float, float, float] | None = None
    root_rotation: tuple[float, float, float, float] | None = None
    local_rotation_deltas: tuple[tuple[float, float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        self._validate_pose_encoding()

    def _validate_pose_encoding(self) -> None:
        present_3d = (
            self.root_translation is not None,
            self.root_rotation is not None,
            self.local_rotation_deltas is not None,
        )
        has_3d = any(present_3d)
        if has_3d and not all(present_3d):
            raise ValueError(
                "3D skeletal fields (root_translation, root_rotation, "
                "local_rotation_deltas) must be all-or-none"
            )
        if self.bone_rotations is not None and has_3d:
            raise ValueError(
                "bone_rotations and 3D skeletal fields are mutually exclusive"
            )

    def to_payload(self) -> dict[str, Any]:
        # Frozen dataclasses can still be corrupted through low-level mutation;
        # never serialize a partial or mixed pose even in that case.
        self._validate_pose_encoding()
        payload: dict[str, Any] = {
            "t_ms": self.t_ms,
            "dx": round(self.dx, 4),
            "dy": round(self.dy, 4),
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "facing": self.facing,
            "lean": round(self.lean, 4),
            "squash": round(self.squash, 4),
            "bob": round(self.bob, 4),
            "expression": self.expression,
        }
        if self.root_translation is not None:
            payload["root_translation"] = [round(v, 6) for v in self.root_translation]
        if self.root_rotation is not None:
            payload["root_rotation"] = [round(v, 6) for v in self.root_rotation]
        if self.local_rotation_deltas is not None:
            payload["local_rotation_deltas"] = [
                [round(v, 6) for v in q] for q in self.local_rotation_deltas
            ]
        if self.bone_rotations is not None:
            payload["bone_rotations"] = [round(value, 6) for value in self.bone_rotations]
        if self.facial_params is not None:
            payload["facial_params"] = {
                key: round(value, 6) for key, value in self.facial_params.items()
            }
        return payload


@dataclass(frozen=True, slots=True)
class MotionPlan:
    plan_id: str
    based_on_seq: int
    behavior: str
    generated_at_ms: int
    valid_until_ms: int
    dt_ms: int
    confidence: float
    seed: int
    points: tuple[MotionPoint, ...]
    target: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plan_id": self.plan_id,
            "based_on_seq": self.based_on_seq,
            "behavior": self.behavior,
            "generated_at_ms": self.generated_at_ms,
            "valid_until_ms": self.valid_until_ms,
            "dt_ms": self.dt_ms,
            "confidence": round(self.confidence, 4),
            "seed": self.seed,
            "points": [point.to_payload() for point in self.points],
        }
        if self.target is not None:
            payload["target"] = dict(self.target)
        return payload


class MotionBackend(ABC):
    """Backend interface; learned PyTorch models implement the same contract.

    Implementations may keep temporal state, but every returned point must be a
    displacement relative to ``world.pet.(foot_x, foot_y)`` from the triggering
    world-state message.  The desktop host remains authoritative for collision
    resolution and root-window placement.
    """

    name = "abstract"

    @abstractmethod
    def generate(self, world: WorldState, seed: int, generated_at_ms: int) -> MotionPlan:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, plan_id: str | None = None) -> bool:
        """Cancel temporal state, returning whether anything was active."""

    def configure_timing(self, plan_horizon_ms: int, plan_dt_ms: int) -> None:
        """Apply timing negotiated by hello, or reject an unsupported shape."""

        raise NotImplementedError(f"{self.name} does not support runtime timing negotiation")

    def close(self) -> None:
        """Release optional accelerator resources."""

    def prepare(self) -> None:
        """Warm backend resources before ready is emitted."""

    def set_skeletal_enabled(self, enabled: bool) -> None:
        """Called after hello negotiation. If false, skeletal output should be omitted."""

    def set_skeletal_3d(self, enabled_3d: bool) -> None:
        """Called when 3D quaternion capability is specifically negotiated."""

    def metrics(self) -> Mapping[str, Any]:
        return {"backend": self.name}


class TorchMotionBackend(MotionBackend):
    """Dependency-free interface marker for the future trained backend.

    This class deliberately does not import torch.  A concrete implementation
    should lazily import it during construction, load/warm its model, then return
    exactly the same ``MotionPlan`` type as the procedural backend.
    """

    name = "torch"

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    def prepare(self) -> None:
        self.warmup()
