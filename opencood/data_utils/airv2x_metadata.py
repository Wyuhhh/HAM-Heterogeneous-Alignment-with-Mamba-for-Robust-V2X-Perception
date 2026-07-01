"""Metadata helpers for the AirV2X-Perception dataset.

This module centralises high-level attributes of the AirV2X-Perception
benchmark (release year, agents, sensors, class taxonomy, etc.).  The
metadata is used to keep training configs and models consistent with the
public specification and to expose utility helpers (e.g. default modality
weights, UAV navigation priors).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class SensorRig:
    lidar: str
    cameras: str
    extras: str


@dataclass(frozen=True)
class DatasetSplits:
    train_hours: float
    val_hours: float
    test_hours: float

    @property
    def total_hours(self) -> float:
        return self.train_hours + self.val_hours + self.test_hours


@dataclass(frozen=True)
class AirV2XMetadata:
    name: str = "AirV2X-Perception"
    release_year: int = 2025
    source: str = "CARLA + AirSim cooperative simulation"
    frequency_hz: int = 5
    max_agents: int = 15
    agent_budget: Mapping[str, int] = field(
        default_factory=lambda: {"vehicle": 5, "rsu": 5, "drone": 5}
    )
    class_names: Mapping[str, int] = field(
        default_factory=lambda: {
            "car": 0,
            "motorcycle": 1,
            "bicycle": 2,
            "van": 3,
            "truck": 4,
            "bus": 5,
        }
    )
    sensor_rigs: Mapping[str, SensorRig] = field(
        default_factory=lambda: {
            "vehicle": SensorRig(
                lidar="1×64-line 360° LiDAR @20Hz",
                cameras="6× surround 1280×720 RGB",
                extras="GNSS, IMU",
            ),
            "rsu": SensorRig(
                lidar="1×64-line 360° LiDAR @20Hz",
                cameras="4× RGB (front/left/right/rear)",
                extras="GNSS",
            ),
            "drone": SensorRig(
                lidar="1×64-line 360° LiDAR @20Hz",
                cameras="1× nadir 1280×720 RGB",
                extras="GNSS, IMU",
            ),
        }
    )
    nav_strategies: Mapping[str, str] = field(
        default_factory=lambda: {
            "hover": "UAV hovers at a fixed waypoint for stationary monitoring.",
            "patrol": "UAV flies through predefined waypoints to cover large areas.",
            "escort": "UAV follows a target vehicle/convoy for persistent coverage.",
        }
    )
    scene_types: List[str] = field(
        default_factory=lambda: ["urban", "suburban", "rural"]
    )
    weather: List[str] = field(
        default_factory=lambda: ["clear", "cloudy", "foggy", "rainy"]
    )
    illumination: List[str] = field(
        default_factory=lambda: ["day", "dusk", "night"]
    )
    splits: DatasetSplits = DatasetSplits(train_hours=2.19, val_hours=1.02, test_hours=3.52)

    def num_classes(self) -> int:
        return len(self.class_names)

    def modality_weights(self, agent_type: str) -> List[float]:
        """Return default modality weights for each encoder of an agent.

        Vehicle: [LiDAR, Camera]. RSU: [LiDAR, Camera]. Drone: [LiDAR, Camera].
        If an agent has a single modality in the config we fall back to [1.0].
        """
        if agent_type == "vehicle":
            # Slightly favour LiDAR (denser) but keep camera contribution.
            return [0.6, 0.4]
        if agent_type == "rsu":
            # RSUs are static with full coverage, keep balanced weights.
            return [0.5, 0.5]
        if agent_type == "drone":
            # Downward camera dominates; LiDAR suffers from motion blur -> lower weight.
            return [0.4, 0.6]
        return [1.0]

    def nav_strategy_scaling(self, strategy: str) -> float:
        """Scaling factor to modulate drone contributions by navigation mode."""
        table = {
            "hover": 0.9,  # mostly static coverage
            "patrol": 1.0,  # baseline
            "escort": 1.1,  # prioritise escort streams for convoy safety
        }
        return table.get(strategy, 1.0)


DEFAULT_AIRV2X_METADATA = AirV2XMetadata()
