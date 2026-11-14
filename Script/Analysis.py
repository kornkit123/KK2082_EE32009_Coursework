#Analysis of noise characteristics of D1-D6
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis
from Config import FS, DATA_DIR
from Functions import load_data
from pathlib import Path
import Functions
from scipy.ndimage import median_filter

ds, Index, Class = load_data(DATA_DIR)
d1, d6 = ds[0], ds[5]
fs = FS
PRE, POST = 40, 150
ok = (Index >= PRE) & (Index < len(d1) - POST)      # keep windows inside the recording
idx, cls = Index[ok], Class[ok]

#### ANALYSE NOISE CHARACTERISTICS OF D1-D6 ####
#checking drift
# Create a 3x2 grid of subplots before the loop
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes = axes.flatten()  # Flatten the 2D grid into a 1D list for easy indexing

i = 0
idx = 0
for idx, i in enumerate(range(0, 6)):
    dn = ds[i]
    ax = axes[idx]  # Select the current subplot (index 0 to 5)
    ax.set_title(f"Dataset {i+1}")  # Note: changed from {dn} so it doesn't print the whole array
    ax.plot(dn[:100000], lw=0.5, label=f"Dataset {i+1} raw")
    baseline = median_filter(dn[:100000], size=2501)  # about 0.1 s window
    ax.plot(baseline, "r", label="slow baseline")
    ax.legend()
    i += 1  # Don't forget to increment i!

plt.tight_layout()  # Adjusts spacing so titles and labels don't overlap
plt.show()  # Show all 6 plots at once after the loop finishes

#check if noise form D2 to D6 is white (flat) or coloured
f, Pxx = welch(d1, fs=fs, nperseg=4096)
plt.figure()
for n, sig in enumerate(ds, 1):
    f, Pxx = welch(sig, fs=fs, nperseg=4096)
    plt.semilogy(f, Pxx, label=f'D{n}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.legend()

#check if noise form D2 to D6 is Gaussian or non-Gaussian


b, a = butter(4, 4000/(fs/2), btype='high')
for n, sig in enumerate(ds, 1):
    hp = filtfilt(b, a, sig)
    print(f'D{n}: excess kurtosis {kurtosis(hp):.3f}')

plt.figure()
plt.xlabel('Amplitude after high-pass (mV)')
plt.ylabel('Probability density')
plt.title('D6 noise amplitude distribution')
plt.hist(filtfilt(b, a, d6), bins=200, density=True)
plt.show()

#visualise the data
print("d = ", len(d1), "\n", "Index = ", len(Index), "\n", "Class = ", np.unique(Class, return_counts=True))
plt.title("D1 data")
#limit plotting to first 5000 samples for better visibility (max 1440000 samples)
plt.plot(d1[:5000]); plt.plot(Index[Index < 5000], d1[Index[Index < 5000]], 'rx')
plt.show()

# 4. Average spike shape for each class (all labelled spikes in D1)
plt.figure()
for k in range(1, 6):
    W = np.array([d1[t - PRE : t + POST] for t in idx[cls == k]])
    plt.plot(np.arange(-PRE, POST), W.mean(0), label=f'class {k}')
plt.axvline(0, color='k', ls=':')
plt.xlabel('Samples from spike start')
plt.ylabel('Amplitude (mV)')
plt.title('Average spike per class (D1)')
plt.legend()
plt.show()


#plot D2 to D6
ds, Index, Class = Functions.load_data(DATA_DIR)
i = 1
for d in ds:
    print("d = ", len(d))
    plt.figure()
    plt.title(f"D{i} data")
    #limit plotting to first 5000 samples for better visibility (max 1440000 samples)
    plt.plot(d[:5000]); plt.plot(Index[Index < 5000], d[Index[Index < 5000]], 'rx')
    plt.show()
    i += 1


#Apply Noise Filtering
#fs = 25000   #25 kHz sampling frequency accroding to coursework specification
#b, a = butter(3, 2000/(fs/2), btype='low')   # try 1000-3000 Hz
#d_f = filtfilt(b, a, d)                      # zero-phase, so no timing shift
