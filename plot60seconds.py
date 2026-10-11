import matplotlib.pyplot as plt
import glob
import numpy as np
from channels import EdfReader, respiratory_channels, find_channel


edf_files = glob.glob("*.edf")
print("Found files:", edf_files)

for filepath in edf_files:
    for item in respiratory_channels:
            reader = EdfReader(filepath)
            labels = reader.getSignalLabels()
            index = find_channel(labels, item)
            if index is None:
                print(f"WARNING: {filepath} has no {item} channel - skipped")
                reader.close()
                continue
            signal = reader.readSignal(index)
            rate = reader.getSampleFrequency(index)
            n_samples = int(60 * rate)
            plt.figure(figsize=(12, 4))
            plt.plot(np.arange(0, 60, 1/rate), signal[0:n_samples], color="purple")
            plt.xlabel("Time (seconds)")
            plt.ylabel(item)
            plt.title(f"{filepath}_{item}_plot")
            plt.savefig(f"{filepath}_{item}.png")
            plt.close()
            # plt.show()
            reader.close()