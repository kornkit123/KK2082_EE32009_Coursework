import torch
from pathlib import Path

DATA_DIR = Path(r'C:\Users\User\Documents\Year4_Uni\EE32009\EE32009 - Coursework\Data')      # folder containing D1.mat to D6.mat, next to the scripts
FS = 25000
DET_PRE, DET_POST = 15, 50
CLS_PRE, CLS_POST = 15, 90
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TTA_OFFSETS = (0,) #s(-6, -3, 0, 3, 6)     # sample shifts used when classifying each detected spike