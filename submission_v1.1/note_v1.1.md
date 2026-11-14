Changes made:
remove signal drift

D2: 2774 spikes, class counts [546 566 538 544 580]
D3: 2651 spikes, class counts [528 549 488 521 565]
D4: 3179 spikes, class counts [708 702 280 664 825]
D5: 2317 spikes, class counts [464 589  94 613 557]
D6: 2684 spikes, class counts [572 641 107 756 608]

config setting
Dataset	Noise	thr	   dist	Val F1
D2	   0.616	0.7	    15	0.965 (from the 0.3-0.9 run)
D3	   0.891	0.85	15	0.935
D4	   1.487	0.93	25	0.854
D5	   2.965	0.9	    25	0.792
D6	   4.415	0.85	15	0.685