# read Noxturnal EDF files and find the respiratory channels in them
#
# GUI exports use labels like "Nasal Pressure" and "RIP Flow"; the command-line
# (NoxCmd) exports use "NasalPressure" and "RIPFlow", and some recordings call
# the thorax belt "Chest". find_channel maps all of those to the same channel.
#
# EdfReader exists because pyedflib refuses to open the Batch 2 files: they are
# marked EDF+ but have no "EDF Annotations" channel, which pyedflib requires.
# The signal data itself is fine, so this reads it directly.

import numpy as np

respiratory_channels = ["Flow", "Nasal Pressure", "Snore", "RIP Flow", "Thorax", "Abdomen"]

# other names the same sensor has in some recordings (already normalized)
aliases = {
    "thorax": ["chest"],
}


def normalize(label):
    """Collapse case and all whitespace so 'Nasal Pressure', 'NasalPressure'
    and 'nasal  pressure ' all compare equal."""
    return "".join(label.split()).lower()


def find_channel(labels, name):
    """Index of channel `name` in the EDF's `labels`, or None if it isn't there."""
    wanted = [normalize(name)] + aliases.get(normalize(name), [])
    normalized_labels = [normalize(label) for label in labels]
    for option in wanted:
        if option in normalized_labels:
            return normalized_labels.index(option)
    return None


class EdfReader:
    """Minimal EDF/EDF+ reader with the same methods as pyedflib.EdfReader that
    our scripts use. Each channel comes back at its own sampling rate, in
    physical units (e.g. cmH2O), exactly as pyedflib would return it."""

    def __init__(self, path):
        with open(path, "rb") as f:
            head = f.read(256)
            ns = int(head[252:256])
            sig = f.read(256 * ns).decode("latin-1")
        header_bytes = int(head[184:192])
        n_records = int(head[236:244])
        record_s = float(head[244:252])

        def field(offset, width):  # one header field for every signal
            start = offset * ns
            return [sig[start + i * width: start + (i + 1) * width].strip() for i in range(ns)]

        self._labels = field(0, 16)
        self._units = field(96, 8)
        self.start = head[168:184].decode("latin-1")   # "dd.mm.yyhh.mm.ss"
        pmin, pmax = np.array(field(104, 8), float), np.array(field(112, 8), float)
        dmin, dmax = np.array(field(120, 8), float), np.array(field(128, 8), float)
        self._spr = np.array(field(216, 8), int)          # samples per data record
        self._rate = self._spr / record_s
        self._gain = (pmax - pmin) / (dmax - dmin)
        self._offset = pmin - dmin * self._gain
        self._start = np.concatenate([[0], np.cumsum(self._spr)])  # where each signal sits in a record
        # memory-mapped, so opening is instant and only the channels you read are loaded
        self._data = np.memmap(path, dtype="<i2", mode="r", offset=header_bytes,
                               shape=(n_records, int(self._spr.sum())))

    def getSignalLabels(self):
        return list(self._labels)

    def getPhysicalDimension(self, index):
        return self._units[index]

    def getFileDuration(self):
        """Length of the recording in seconds."""
        return self._data.shape[0] * float(self._spr[0] / self._rate[0])

    def getSampleFrequency(self, index):
        return float(self._rate[index])

    def readSignal(self, index):
        digital = self._data[:, self._start[index]:self._start[index + 1]].reshape(-1)
        return digital * self._gain[index] + self._offset[index]

    def close(self):
        self._data = None
