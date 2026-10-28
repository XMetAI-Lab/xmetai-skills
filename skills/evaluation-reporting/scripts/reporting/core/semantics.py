"""Kinds of datasets and time semantics shared by readers and consumers."""

from enum import Enum


class TemporalKind(Enum):
    INSTANTANEOUS = "instantaneous"
    INTERVAL_ACCUMULATION = "interval_accumulation"
    SINCE_INIT_ACCUMULATION = "since_init_accumulation"
    INTERVAL_MEAN = "interval_mean"


class DataKind(Enum):
    GRIDDED_FORECAST = "gridded_forecast"
    STATION_FORECAST = "station_forecast"
    GRIDDED_OBSERVATION = "gridded_observation"
    STATION_OBSERVATION = "station_observation"
    REFERENCE = "reference"
    DERIVED = "derived"
