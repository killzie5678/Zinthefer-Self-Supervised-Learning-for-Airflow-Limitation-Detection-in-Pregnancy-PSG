# extract the six respiratory channels from one EDF file
import matplotlib.pyplot as plt
import glob
import pyedflib
import numpy as np
from scipy.signal import resample_poly
from math import gcd


filepath = "5031.edf"
target_rate = 200 #arbitrary val
extracted = {}
respiratory_channels = ["Flow", "Nasal Pressure", "Snore", "RIP Flow", "Thorax", "Abdomen"]

def zscore(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# def normalize(label):
#     """Collapse case and all whitespace so 'Nasal Pressure', 'NasalPressure'
#     and 'nasal  pressure ' all compare equal."""
#     return "".join(label.split()).lower()


# respiratory_lookup = {normalize(name) for name in respiratory_channels}

reader = pyedflib.EdfReader(filepath)

for item in respiratory_channels:
        
        labels = reader.getSignalLabels()
        index = labels.index(item)
        signal = reader.readSignal(index)
        rate = reader.getSampleFrequency(index)
        n_samples = int(60 * rate)
        if (rate != target_rate): 
            g = gcd(int(rate), target_rate)
            up = target_rate // g
            down = rate // g
            signal = resample_poly(signal, up, down)
            rate = target_rate; 
        normalized_signal = zscore(signal)           
        extracted[item] = {"signal": signal, "normalized_signal": normalized_signal, "rate": rate}

reader.close()
for channel_name, data in extracted.items():
    print(channel_name, "- rate:", data["rate"], "- num samples:", len(data["signal"]))
    print("   normalized mean:", np.mean(data["normalized_signal"]), "- normalized std:", np.std(data["normalized_signal"]))