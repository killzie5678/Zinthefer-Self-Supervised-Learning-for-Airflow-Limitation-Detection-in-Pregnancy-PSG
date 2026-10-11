# extract the six respiratory channels from one EDF file
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from channels import EdfReader, respiratory_channels, find_channel

edf_files = glob.glob("*.edf")
# filepath = "5031.edf"
target_rate = 200 #arbitrary val

def zscore(signal):
    return (signal - np.mean(signal)) / np.std(signal)


for filepath in edf_files:
    reader = EdfReader(filepath)
    extracted = {}  # fresh for each file, so one file's channels never end up in another's .npz

    for item in respiratory_channels:

            labels = reader.getSignalLabels()
            index = find_channel(labels, item)
            if index is None:
                print(f"WARNING: {filepath} has no {item} channel - skipped")
                continue
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

    save_dict = {}
    for channel_name, data in extracted.items():
        string_key = channel_name + "_signal"
        save_dict[string_key] = data["signal"]
        string_key = channel_name + "_normalized_signal"
        save_dict[string_key] = data["normalized_signal"]
        string_key = channel_name + "_rate"
        save_dict[string_key] = data["rate"]
                 

    output_name = filepath + "_processed.npz"
    np.savez(output_name, **save_dict)     
# loaded = np.load("5031_processed.npz") 
# print("Keys found:", list(loaded.keys()))
# print("Flow rate:", loaded["Flow_rate"])
# print("Flow signal length:", len(loaded["Flow_signal"]))