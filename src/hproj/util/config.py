from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class MeasurementConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementConfig":
        return cls(**data)


@dataclass
class ProjectorConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectorConfig":
        return cls(**data)


@dataclass
class ClassifierConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassifierConfig":
        return cls(**data)


@dataclass
class ProjectorCalibrationConfig:
    select: Optional[str] = None
    subsample: Optional[int] = None
    dimensions: list[int] = field(default_factory=list)
    measurements: list[MeasurementConfig] = field(default_factory=list)
    projectors: list[ProjectorConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectorCalibrationConfig":
        data = dict(data)

        if "measurements" in data:
            data["measurements"] = [
                MeasurementConfig.from_dict(m) for m in data["measurements"]
            ]

        if "projectors" in data:
            data["projectors"] = [
                ProjectorConfig.from_dict(p) for p in data["projectors"]
            ]

        return cls(**data)


@dataclass
class ClassifierCalibrationConfig:
    metrics: list[str] = field(default_factory=list)
    select: Optional[str] = None
    subsample: Optional[int] = None
    dimensions: list[int] = field(default_factory=list)
    classifiers: list[ClassifierConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassifierCalibrationConfig":
        data = dict(data)

        if "classifiers" in data:
            data["classifiers"] = [
                ClassifierConfig.from_dict(c) for c in data["classifiers"]
            ]

        return cls(**data)


@dataclass
class CalibrationConfig:
    projector: ProjectorCalibrationConfig = field(default_factory=ProjectorCalibrationConfig)
    classifier: ClassifierCalibrationConfig = field(default_factory=ClassifierCalibrationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationConfig":
        data = dict(data)

        if "projector" in data:
            data["projector"] = ProjectorCalibrationConfig.from_dict(data["projector"])

        if "classifier" in data:
            data["classifier"] = ClassifierCalibrationConfig.from_dict(data["classifier"])

        return cls(**data)


@dataclass
class CalibrationSeedConfig:
    projector: list[int] = field(default_factory=list)
    classifier: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationSeedConfig":
        return cls(**data)


@dataclass
class SeedConfig:
    calibration: CalibrationSeedConfig = field(default_factory=CalibrationSeedConfig)
    curve: list[int] = field(default_factory=list)
    evaluation: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedConfig":
        data = dict(data)

        if "calibration" in data:
            data["calibration"] = CalibrationSeedConfig.from_dict(data["calibration"])

        return cls(**data)


@dataclass
class IntervalConfig:
    start: int
    stop: int
    step: int

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def as_range(self):
        return range(self.start, self.stop, self.step)


@dataclass
class CurveConfig:
    dimensions: list[int]
    subsample: int
    projectors: list[str]
    classifiers: list[str]
    measurements: list[MeasurementConfig] = field(default=MeasurementConfig)
    include_full_dimension: Optional[bool] = None
    dimension_intervals: Optional[list[tuple[int]]] = None


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurveConfig":
        data = dict(data)

        if "measurements" in data:
            data["measurements"] = [
                MeasurementConfig.from_dict(m) for m in data["measurements"]
            ]

        if "dimension_intervals" in data:
            data["dimension_intervals"] = [
                IntervalConfig.from_dict(i) for i in data['dimension_intervals']
            ]

            list_of_ranges = [
                list(i.as_range()) 
                for i in data["dimension_intervals"]
            ]
            data['dimensions'] = [r for rs in list_of_ranges for r in rs]

        return cls(**data)


@dataclass
class ThresholdsConfig:
    metrics: list[str] = field(default_factory=list)
    conditions: list[float] = field(default_factory=float)

@dataclass
class Config:
    datasets: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    num_folds: int = 5
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data)

        if "seeds" in data:
            data["seeds"] = SeedConfig.from_dict(data["seeds"])

        if "calibration" in data:
            data["calibration"] = CalibrationConfig.from_dict(data["calibration"])

        if "curve" in data:
            data["curve"] = CurveConfig.from_dict(data["curve"])

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)