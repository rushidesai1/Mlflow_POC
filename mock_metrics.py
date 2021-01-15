import hashlib
import numpy as np
import pandas as pd

from typing import Dict

# data params
METRIC_CEILING = 100
NUM_RUNS = 6

# distribution params
RUNS_TO_CONVERGE = 10

# (domain, intent), sra, nd_sra
data = [
    (("HomeAutomation", "TurnOffApplianceIntent"), 97.09638504, 98.41726059),
    (("HomeAutomation", "TurnOnApplianceIntent"), 97.88362163, 99.06682851),
    (("Knowledge", "QAIntent"), 73.64100535, 87.12315078),
    (("Notifications", "SetNotificationIntent"), 99.51719854, 99.79723254),
    (("Global", "WhatTimeIntent"), 99.93474923, 99.94461141),
    (("Weather", "GetWeatherForecastIntent"), 99.60850867, 99.70823927),
    (("Music", "PlayMusicIntent"), 97.89140701, 98.91662678),
    (("Global", "SetVolumeIntent"), 99.94975672, 99.9817287),
    (("Global", "VolumeUpIntent"), 99.84027154, 99.94186961),
    (("Global", "VolumeDownIntent"), 99.95325138, 99.9765938),
    (("Routines", "InvokeRoutineIntent"), 99.68521334, 99.68619846),
    (("ToDos", "AddToListIntent"), 97.58041796, 99.12809747),
    (("Communication", "CallIntent"), 92.34842863, 96.88840201),
    (("Music", "PlayStationIntent"), 98.8949848, 99.53980727),
    (("HomeAutomation", "SetValueIntent"), 88.17185979, 98.40306741),
    (("DailyBriefing", "DailyBriefingIntent"), 98.70396644, 99.31134193),
    (("Communication", "AnnounceIntent"), 99.45622191, 99.5280811),
    (("Communication", "InstantConnectIntent"), 95.48995307, 98.4024958),
    (("Books", "ReadBookIntent"), 75.69193742, 89.57761751),
    (("HomeAutomation", "DisplayVideoFeedIntent"), 88.56425363, 96.88319643)
]


def get_hash(id: str) -> int:
    hash_value = hashlib.sha256(id.encode('utf-8'))
    return int.from_bytes(hash_value.digest(), 'big') % 10 ** 8


def get_mock_metrics(pair: str, start: Dict[str, int], num_runs: int = NUM_RUNS):
    seed = get_hash(pair)
    np.random.seed(seed)

    # hack to generate constant ratio of defective and non-defective SRA relative to metric ceiling
    sra_ratio = get_sra_ratio(start)

    sra_metrics = np.full(num_runs, start["sra"])
    nd_sra_metrics = np.full(num_runs, start["nd_sra"])
    for i in range(1, num_runs):
        # the following is a mess but its technically supposed to generate a mess so it works
        slope = (METRIC_CEILING / sra_metrics[i - 1]) * RUNS_TO_CONVERGE - 1
        new_metric = METRIC_CEILING + 1
        while (new_metric > METRIC_CEILING):
            diff = random_walk(slope, 0.2)
            new_metric = sra_metrics[i - 1] + diff
        sra_metrics[i] = new_metric
        nd_sra_metrics[i] = sra_metrics[i] + \
            (METRIC_CEILING - sra_metrics[i]) * sra_ratio

    # metrics += (np.random.rand(num_runs) * variation) - (variation / 2)
    return (sra_metrics, nd_sra_metrics)


def get_sra_ratio(start):
    sra = start["sra"]
    nd_sra = start["nd_sra"]
    return (nd_sra - sra) / (METRIC_CEILING - sra)


def random_walk(slope: float = 0.1, sigma=0.1) -> float:
    diff = slope + np.random.normal(0, slope / 2)
    return diff


def get_mock_data():
    mock_data = []
    for datum in data:
        domain, intent = datum[0]
        initial_values = {
            "sra": datum[1],
            "nd_sra": datum[2]
        }
        metrics = get_mock_metrics(f"{domain}:{intent}", initial_values)
        mock_data.append((domain, intent, metrics))

    return mock_data
