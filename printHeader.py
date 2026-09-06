import pyedflib
import glob

edf_files = glob.glob("*.edf")
print("Found files:", edf_files)
respiratory_channels = ["Flow", "Nasal Pressure", "Snore", "RIP Flow", "Thorax", "Abdomen"]


def normalize(label):
    """Collapse case and all whitespace so 'Nasal Pressure', 'NasalPressure'
    and 'nasal  pressure ' all compare equal."""
    return "".join(label.split()).lower()


respiratory_lookup = {normalize(name) for name in respiratory_channels}

with open("channelnamessamplerates.txt", "w") as g, open("specificchannelnames.txt", "w") as h:
    for filepath in edf_files:
        reader = pyedflib.EdfReader(filepath)
        print(f"\n=== {filepath} ===")
        h.write(f"\n=== {filepath} ===\n")
        header = reader.getHeader()
        print("Start date:", header.get('startdate'))
        print("Patient:", header.get('patientname'))

        print(f"\nNumber of channels: {reader.signals_in_file}")
        print(f"{'Label':<20} {'Sample Rate (Hz)':<18} {'Physical Min':<15} {'Physical Max':<15}")
        for sig_header in reader.getSignalHeaders():
            print(f"{sig_header['label']:<20} {sig_header['sample_frequency']:<18} {sig_header['physical_min']:<15} {sig_header['physical_max']:<15}")
            g.write(f"{sig_header['label']:<20} {sig_header['sample_frequency']:<18}\n")
            if normalize(sig_header['label']) in respiratory_lookup:
                h.write(f"{sig_header['label']:<20} {sig_header['sample_frequency']:<18}\n")

        reader.close()
