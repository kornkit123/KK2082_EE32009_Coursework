Changes made:
1. resweep and repick parameters
2. increase epoch from traing clf model from 40 to 80. 
    Because it hasn't finished converging at 40 epoches

D1 reference: 2432 spikes
D2: 2739 spikes, class counts [535 561 528 536 579]
D3: 2663 spikes, class counts [529 547 503 523 561]
D4: 3163 spikes, class counts [711 699 276 659 818]
D5: 2315 spikes, class counts [452 588  97 619 559]
D6: 2556 spikes, class counts [522 620 115 709 590]


Config setting
settings = {2: (0.9, 15), 3: (0.85, 15), 4: (0.93, 25), 5: (0.9, 25), 6: (0.85, 25)}
        Noise  thr   dist   Precision  Recall   F1
Data 2: 0.438  (0.9,  25),  1.000      0.950   0.974
Data 3: 0.648  (0.85, 25),  0.998      0.948   0.972
Data 4: 1.082  (0.93,  25), 0.979      0.849   0.909
Data 5: 2.073  (0.9,  25),  0.972      0.759   0.852
Data 6: 3.065  (0.85, 25)   0.897      0.704   0.789