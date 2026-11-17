#Analysis of noise characteristics of D1-D6
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, norm
from Config import FS, DATA_DIR
from Functions import load_data
from pathlib import Path
import Functions
from scipy.ndimage import median_filter

ds_raw, Index, Class = load_data(DATA_DIR)
fs = FS
PRE, POST = 40, 150

#### ANALYSE NOISE CHARACTERISTICS OF D1-D6 ####
# Before cleaning
fig1, axes1 = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
for j, ax in enumerate(axes1.flatten()):
    ax.plot(ds_raw[j][:100000], lw=0.5, label=f'Dataset {j + 1} raw')
    ax.plot(median_filter(ds_raw[j][:100000], size=2501), 'r', label='slow baseline')
    ax.set_title(f'Dataset {j + 1}')
    ax.legend()
fig1.tight_layout()

#clean drift 
ds = [Functions.remove_baseline(s) for s in ds_raw]    # drift removed, same as Main.py
d1, d6 = ds[0], ds[5]
ok = (Index >= PRE) & (Index < len(d1) - POST)      # keep windows inside the recording
idx, cls = Index[ok], Class[ok]

#compare before and after cleaning drift
for n, (raw, clean) in enumerate(zip(ds_raw, ds), 1):
    print(f'D{n} noise std: raw {Functions.noise_std(raw):.3f} -> cleaned {Functions.noise_std(clean):.3f}')

#detectability ceiling for each class at D4 to D6 noise
LEN = 60                                              # samples after spike start to use
ok_c = Index < len(d1) - LEN
idx_c, cls_c = Index[ok_c], Class[ok_c]
templates = {k: np.mean([d1[t:t + LEN] for t in idx_c[cls_c == k]], axis=0) for k in range(1, 6)}

Z_THR = 2.6      # assumed threshold in noise std units (about 0.5% false alarms per position)
for name, i in (('D4', 3), ('D5', 4), ('D6', 5)):
    sigma = Functions.noise_std(ds[i])
    print(f'\n{name} (noise std {sigma:.2f})')
    for k in range(1, 6):
        d = np.sqrt(np.sum(templates[k] ** 2)) / sigma
        print(f'  class {k}: detectability {d:.2f} -> best possible recall about {norm.sf(Z_THR - d):.2f}')
    for a, b in ((4, 5), (1, 2)):
        d = np.sqrt(np.sum((templates[a] - templates[b]) ** 2)) / sigma
        print(f'  classes {a} vs {b}: separation {d:.2f} -> best possible error about {norm.sf(d / 2):.2f}')

#checking after clearning drift
# Create a 3x2 grid of subplots before the loop
# After cleaning
fig2, axes2 = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
for j, ax in enumerate(axes2.flatten()):
    ax.plot(ds[j][:100000], lw=0.5, label=f'Dataset {j + 1} cleaned')
    ax.plot(median_filter(ds[j][:100000], size=2501), 'r', label='slow baseline')
    ax.set_title(f'Dataset {j + 1}')
    ax.legend()
fig2.tight_layout()

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
