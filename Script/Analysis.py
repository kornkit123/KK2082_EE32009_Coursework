#Analysis of noise characteristics of D1-D6
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis
from Config import FS, DATA_DIR
from Functions import load_data

ds, Index1, Class1 = load_data(DATA_DIR)
d1, d6 = ds[0], ds[5]
fs = FS
PRE, POST = 40, 150
ok = (Index1 >= PRE) & (Index1 < len(d1) - POST)      # keep windows inside the recording
idx, cls = Index1[ok], Class1[ok]

#### ANALYSE NOISE CHARACTERISTICS OF D1-D6 ####
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
print("d = ", len(d1), "\n", "Index = ", len(Index1), "\n", "Class = ", np.unique(Class1, return_counts=True))
plt.title("D1 data")
#limit plotting to first 5000 samples for better visibility (max 1440000 samples)
plt.plot(d1[:5000]); plt.plot(Index1[Index1 < 5000], d1[Index1[Index1 < 5000]], 'rx')
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

#Apply Noise Filtering
#fs = 25000   #25 kHz sampling frequency accroding to coursework specification
#b, a = butter(3, 2000/(fs/2), btype='low')   # try 1000-3000 Hz
#d_f = filtfilt(b, a, d)                      # zero-phase, so no timing shift
