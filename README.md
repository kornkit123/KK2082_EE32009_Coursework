# EE32009 Coursework: Spike Detection and Classification

Detects and classifies neuronal spikes (five neuron classes) in extracellular
recordings sampled at 25 kHz. The system is trained on the labelled recording D1
and applied to the unlabelled, noisier recordings D2-D6. For each recording it
produces an `Index` vector (spike start positions) and a `Class` vector (1-5).

## Method summary

1. **Drift removal.** A 2501-sample median filter estimates the slow baseline,
   which is subtracted from every recording (D1-D6) before anything else.
2. **Noise estimation.** The noise level of each recording is measured with a
   robust median-absolute-deviation estimate. A spectrum check showed the noise
   is white and Gaussian, so it can be simulated.
3. **Training data.** Spike, background and hard-negative windows are cut from
   clean D1. Fresh Gaussian noise at a randomly chosen D1-D6 level is added to
   every training batch, so the models never see the same noise twice.
4. **Two 1-D CNNs** (`SpikeCNN` in `Models.py`):
   - detector: is a spike starting at this sample? (65-sample window, 2 outputs)
   - classifier: which of the 5 neuron classes? (105-sample window, 5 outputs)
5. **Optional fixed low-pass layer** as the first layer of each CNN. It filters
   signal and training noise together, in training and at prediction time.
6. **Peak picking.** `find_peaks` on the detector probabilities, with a
   threshold and minimum spacing chosen per dataset (`settings` in `Main.py`).
7. **Classification** of each detected spike, then export to `.mat`.

## Project structure

```
Assets/             Figures and notes (not needed to run the code)
Data/               D1.mat ... D6.mat (not included, download from Moodle)
Script/
    Main.py         Full pipeline: train/load models, validate, export
    Functions.py    Data loading, windowing, noise, training, detection, scoring
    Models.py       SpikeCNN and the fixed low-pass layer
    Config.py       Constants (sample rate, window sizes, data path, device)
    Analysis.py     Exploratory plots (noise spectrum, Gaussian check, drift)
    requirements.txt
models/             Saved weights (created by Main.py)
submission/         Exported D2.mat ... D6.mat (created by Main.py)
```

## Requirements

- Python 3.10 or newer
- Libraries listed in `Script/requirements.txt`:
  numpy, scipy, matplotlib, scikit-learn, torch

A GPU is optional. The code uses CUDA if available and falls back to the CPU.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r Script/requirements.txt
```

Place `D1.mat` to `D6.mat` in the `Data/` folder, then check `DATA_DIR` in
`Script/Config.py`. It should point to that folder, for example:

```python
DATA_DIR = Path(__file__).parent.parent / 'Data'
```

## Running

Run everything from the project root so `models/` and `submission/` are created there:

```bash
python Script/Main.py
```

### Switches at the top of `Main.py`

| Setting | Meaning |
|---|---|
| `RETRAIN = True` | Train both models and save them to `models/` |
| `RETRAIN = False` | Load previously saved models (fast, no training) |
| `RUN_SWEEP = True` | Run the validation sweep over thresholds and spike spacing |
| `RUN_SWEEP = False` | Skip the sweep |
| `LOWPASS_HZ` | Cutoff of the fixed low-pass layer in Hz, or `None` for no filter |
| `MODEL_TAG` | Name suffix for the saved weights, so versions do not overwrite each other |
| `settings` | `{dataset: (threshold, min spike distance)}` used at export |

Typical workflow:

1. First run: `RETRAIN = True`, `RUN_SWEEP = True`. Train, then read the best
   `(threshold, distance)` per dataset from the sweep output.
2. Copy those values into `settings`.
3. Later runs: `RETRAIN = False`, `RUN_SWEEP = False` to re-export quickly.

Retrain whenever the preprocessing, model architecture (including `LOWPASS_HZ`),
noise augmentation or window sizes change. Changing only `settings` does not
need a retrain.

## Output

`submission/D2.mat` to `submission/D6.mat`, each containing only:

- `Index`: spike start positions in samples
- `Class`: class labels 1-5, one per spike

Zip the `.mat` files (directly in the ZIP, no subfolder) and upload to Moodle.
The raw signal `d` is deliberately not included.

The console also prints the spike count and per-class counts for each dataset,
which is a quick sanity check against the number of spikes expected in D1.

## Exploratory analysis

```bash
python Script/Analysis.py
```

Plots the power spectra of D1-D6 (white noise check), the amplitude histogram
of high-passed D6 (Gaussian check), the slow baseline of each recording, and
the average spike shape for each class.

## Results log

Fill in after each automarker run.

| Submission | Change | D2 P / R | D3 P / R | D4 P / R | D5 P / R | D6 P / R |
|---|---|---|---|---|---|---|
| 1 | Baseline CNNs, no drift removal | 0.970 / 0.912 | 0.958 / 0.875 | 0.944 / 0.794 | 0.322 / 0.717 | 0.132 / 0.592 |
| 2 | Drift removal | 1.0 / 0.942 | 0.996 / 0.938 | 0.995 / 0.832 | 0.979 / 0.784 | 0.915 / 0.741 |

## Notes and limitations

- Validation noise is simulated white Gaussian noise added to the last 20% of D1.
  It is optimistic compared with the real D2-D6 recordings, so the automarker
  feedback is the final check.
- Overlapping spikes closer than the minimum spacing are merged into one detection.
- Classification accuracy falls at high noise (D5, D6), mainly through confusion
  between Classes 4 and 5, and between Classes 1 and 2.
