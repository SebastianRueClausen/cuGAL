from enum import Enum
from time import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from itertools import chain
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


@dataclass
class Profile:
    sinkhorn_profiles: list[SinkhornProfile] = field(default_factory=list)
    phase_times: dict[Phase, float] = field(default_factory=dict)
    time: float = 0.0

    def log_time(self, start_time: float, phase: Phase):
        prev_time = 0 if not phase in self.phase_times else self.phase_times[phase]
        self.phase_times[phase] = prev_time + (time() - start_time)


def extract_phase_times(profiles: list[Profile], phase: Phase) -> list[float]:
    return [profile.phase_times[phase] for profile in profiles]    


def write_plot_phases_as_csv(profiles: list[Profile], sizes: list[int], path: str):
    phases = [Phase.SINKHORN, Phase.FEATURE_EXTRACTION, Phase.GRADIENT, Phase.HUNGARIAN]

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(chain([""], sizes))
        for phase in phases:
            phase_times = extract_phase_times(profiles, phase)
            writer.writerow(chain([phase.name], phase_times))

def append_phase_to_csv(profile: Profile, path: str):
    phases = [Phase.SINKHORN, Phase.FEATURE_EXTRACTION, Phase.GRADIENT, Phase.HUNGARIAN]
    filepath = f"{path}/times.csv"

    if os.path.isfile(filepath):
        csvfile = open(filepath, 'r', newline='')

        reader = csv.reader(csvfile)
        data = [row for row in reader]
    else: data = [[""]]*len(phases)

    csvfile = open(filepath, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row, phase in enumerate(phases):
        phase_times = extract_phase_times([profile], phase)
        writer.writerow(data[row] + phase_times)
    
    print("Wrote times to ", path)

def plot_phases(profiles: list[Profile], sizes: list[int]):
    feature_extraction_times = extract_phase_times(profiles, Phase.FEATURE_EXTRACTION)
    sinkhorn_times = extract_phase_times(profiles, Phase.SINKHORN)
    hungarian_times = extract_phase_times(profiles, Phase.HUNGARIAN)
    gradient_times = extract_phase_times(profiles, Phase.GRADIENT)
    plt.stackplot(sizes, feature_extraction_times, sinkhorn_times, hungarian_times, gradient_times)
    plt.show()