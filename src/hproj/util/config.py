from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MeasurementConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectorConfig":
        # simply unpack the dictionary; the constructor provides defaults
        return cls(**data)


@dataclass
class ProjectorConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectorConfig":
        # simply unpack the dictionary; the constructor provides defaults
        return cls(**data)


@dataclass
class ClassifierConfig:
    name: str
    paras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassifierConfig":
        # simply unpack the dictionary; the constructor provides defaults
        return cls(**data)



@dataclass
class CalibrationConfig:
    dimensions: list[int] = field(default_factory=list)
    measurements: list[MeasurementConfig] = field(default_factory=list)
    classifiers: list[ClassifierConfig] = field(default_factory=list)
    projectors: list[ProjectorConfig] = field(default_factory=list)
    subsample: int = None
    select: str = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationConfig":
        # convert nested projectors and unpack
        if "projectors" in data:
            data['projectors'] = [ProjectorConfig.from_dict(p) for p in data["projectors"]]
        if "measurements" in data:
            data['measurements'] = [MeasurementConfig.from_dict(p) for p in data["measurements"]]
        if "classifiers" in data:
            data['classifiers'] = [ClassifierConfig.from_dict(p) for p in data["classifiers"]]
        return cls(**data)


@dataclass
class SeedConfig:
    calibration: list[int] = field(default_factory=list)
    curve: list[int] = field(default_factory=list)
    evaluation: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedConfig":
        return cls(**data)


@dataclass
class Config:
    datasets: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    num_folds: int = 5

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        if "calibration" in data:
            calibration_dict = data["calibration"]
            data = {
                **data,
                "calibration": CalibrationConfig.from_dict(calibration_dict),
            }
        if "seeds" in data:
            seeds_dict = data["seeds"]
            data = {**data, "seeds": SeedConfig.from_dict(seeds_dict)}
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load a configuration from a YAML file or string path.

        Args:
            path: file path to the YAML config.

        Returns:
            Config: populated dataclass hierarchy.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
