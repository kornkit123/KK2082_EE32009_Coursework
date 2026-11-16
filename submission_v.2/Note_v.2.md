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

From Feedback Report
Dataset	Det. precision	Det. recall	F_ident	Class precision	Class recall	F_class	F_dataset (0.3·F_ident + 0.7·F_class)
D2	    1.000	        0.942	    0.970	    0.965	        0.909	        0.936	0.946
D3	    0.996	        0.938	    0.966	    0.961	        0.906	        0.932	0.942
D4	    0.995	        0.832	    0.906	    0.928	        0.776	        0.845	0.863
D5	    0.979	        0.784	    0.871	    0.819	        0.656	        0.728	0.771
D6	    0.915	        0.741	    0.819	    0.599	        0.485       	0.536	0.621
Weighted final (0.1·D2 + 0.15·D3 + 0.2·D4 + 0.25·D5 + 0.3·D6): about 0.79