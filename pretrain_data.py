# load normalized 30 s windows from the pretraining dataset
#
#   from pretrain_data import WindowDataset
#   train = WindowDataset("train")
#   x, present = train[0]      # x: (6, 6000) float32, present: (6,) bool
#
# Plain numpy, so it works without PyTorch; it has __len__ and __getitem__,
# so it can be passed to torch.utils.data.DataLoader as-is later.
#
# Normalization: each channel is scaled with its own recording's median and
# IQR (from manifest.csv): (x - median) / IQR, then clipped to +-CLIP so a
# rare movement spike can't dominate. Median/IQR instead of mean/std because
# those spikes would drag the mean and std around. A belt channel that has a
# gap in the window is set to 0 and marked False in `present`.

import csv
from pathlib import Path

import numpy as np

from channels import respiratory_channels

CLIP = 10.0


class WindowDataset:
    def __init__(self, split, data="pretrain dataset", window_seconds=30):
        data = Path(data)
        self.data = data
        manifest = {r["recording_id"]: r for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8"))}
        self.windows = [w for w in csv.DictReader(open(data / "windows.csv", encoding="utf-8"))
                        if w["split"] == split]
        self.stats = {}
        for w in self.windows:
            rid = w["recording_id"]
            if rid not in self.stats:
                rec = manifest[rid]
                median = np.array(rec["medians"].split(";"), np.float32)
                iqr = np.array(rec["iqrs"].split(";"), np.float32)
                self.stats[rid] = (median[:, None], np.where(iqr > 0, iqr, 1)[:, None])
                self.length = window_seconds * int(rec["rate"])
        self._open = {}  # memory-mapped .npy per recording, opened on first use

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        rid = w["recording_id"]
        if rid not in self._open:
            self._open[rid] = np.load(self.data / f"{rid}.npy", mmap_mode="r")
        start = int(w["start_sample"])
        x = np.asarray(self._open[rid][:, start:start + self.length], np.float32)
        median, iqr = self.stats[rid]
        x = np.clip((x - median) / iqr, -CLIP, CLIP)
        missing = set(filter(None, w["missing"].split(";")))
        present = np.array([name not in missing for name in respiratory_channels])
        x[~present] = 0 
        return x, present
