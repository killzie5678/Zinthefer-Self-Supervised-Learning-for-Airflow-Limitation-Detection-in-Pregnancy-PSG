import pyedflib
import glob

edf_files = glob.glob("*.edf")
print("Found files:", edf_files)
g = open("channelnamessamplerates.txt", "w")

for filepath in edf_files:
    reader = pyedflib.EdfReader(filepath)
    print(f"\n=== {filepath} ===")
    
    header = reader.getHeader()
    print("Start date:", header.get('startdate'))
    print("Patient:", header.get('patientname'))
    
    print(f"\nNumber of channels: {reader.signals_in_file}")
    print(f"{'Label':<20} {'Sample Rate (Hz)':<18} {'Physical Min':<15} {'Physical Max':<15}")
    for sig_header in reader.getSignalHeaders():
        print(f"{sig_header['label']:<20} {sig_header['sample_frequency']:<18} {sig_header['physical_min']:<15} {sig_header['physical_max']:<15}")
        g.write(str(f"{sig_header['label']:<20} {sig_header['sample_frequency']:<18}"))
        g.write("\n")

    reader.close()