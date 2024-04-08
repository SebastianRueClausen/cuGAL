from enum import Enum
from time import time
from typing import Self
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from itertools import chain
import torch
import csv
import os


class Phase(Enum):
    FEATURE_EXTRACTION = 0
    SINKHORN = 1
    HUNGARIAN = 2
    GRADIENT = 3


@dataclass
class SinkhornProfile:
    errors: list[float] = field(default_factory=list)
    iteration_count: int = 0
    time: float = 0.0


class TimeStamp:
    def __init__(self, device: str):
        if 'cuda' in device:
            self.time = torch.cuda.Event(enable_timing=True)
            self.time.record()
        else:
            self.time = time()
        self.device = device

    def is_cuda_event(self) -> bool:
        return type(self.time) is torch.cuda.Event

    def elapsed_seconds(self, earlier: Self) -> float:
        if self.is_cuda_event():
            torch.cuda.synchronize()
            return earlier.time.elapsed_time(self.time) / 1000
        else:
            return self.time - earlier.time


@dataclass
class Profile:
    sinkhorn_profiles: list[SinkhornProfile] = field(default_factory=list)
    phase_times: dict[Phase, float] = field(default_factory=dict)
    time: float = 0.0

    def log_time(self, start_time: TimeStamp, phase: Phase):
        now = TimeStamp(start_time.device)
        prev_time = 0 if not phase in self.phase_times else self.phase_times[phase]
        self.phase_times[phase] = prev_time + now.elapsed_seconds(start_time)


def extract_phase_times(profiles: list[Profile], phase: Phase) -> list[float]:
    return [profile.phase_times[phase] for profile in profiles]


def write_phases_as_csv(profiles: list[Profile], sizes: list[int], path: str):
    phases = [Phase.SINKHORN, Phase.FEATURE_EXTRACTION,
              Phase.GRADIENT, Phase.HUNGARIAN]

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(chain([""], sizes))
        for phase in phases:
            phase_times = extract_phase_times(profiles, phase)
            writer.writerow(chain([phase.name], phase_times))


def append_phases_to_csv(profile: Profile, path: str):
    phases = [Phase.SINKHORN, Phase.FEATURE_EXTRACTION,
              Phase.GRADIENT, Phase.HUNGARIAN]
    filepath = f"{path}/times.csv"

    if os.path.isfile(filepath):
        csvfile = open(filepath, 'r', newline='')

        reader = csv.reader(csvfile)
        data = [row for row in reader]
    else:
        data = [[""]]*len(phases)

    csvfile = open(filepath, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, phase in enumerate(phases):
        phase_times = extract_phase_times([profile], phase)
        writer.writerow(data[row] + phase_times)

    print("Wrote times to ", path)


def plot_phases(profiles: list[Profile], sizes: list[int]):
    labels = ["Feature Extraction", "Sinkhorn-Knopp", "Hungarian", "Gradient"]
    plt.stackplot(
        sizes,
        extract_phase_times(profiles, Phase.FEATURE_EXTRACTION),
        extract_phase_times(profiles, Phase.SINKHORN),
        extract_phase_times(profiles, Phase.HUNGARIAN),
        extract_phase_times(profiles, Phase.GRADIENT),
        labels=labels,
        baseline='zero',
    )
    plt.xlabel("Graph size")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.show()
