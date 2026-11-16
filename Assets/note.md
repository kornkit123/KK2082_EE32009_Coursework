Submission1
after looking at D1 data 
1. notice the idx marker are at the rising edge
2. there are 5 different unique class labelled: 1, 2, 3, 4, 5 from five different types of neuron
3. most spike got undershoot -> this could be useful
4. all noise are white noise
5. D6 is gaussian noise

Submission2
After looking at signal from D1 to D6
1. notice singal has drift which get worst with higher noise level

Submission 3
1. notice classification miss mainly on neuron class3
this potentially because class3 is the weakest spike
2. Classes 4 and 5 have similar shapes and differ mainly in amplitude (2.7 against 4.3). Noise blurs the amplitude, which is why 4 and 5 swap labels, especially at D6.
3. Classes 1 and 2 have similar peaks (4.9 and 5.4). Class 1 is narrower, and Class 2 has a deeper undershoot after the peak and a bump later. Both are small features that noise hides.

plan
1. train on D1 
2. add synthetic noise to D1 during training
3. have 2 different model: 1 for classification and 1 for detecting neuron signal spike
4. remove drift
future
5. add noise filter

validating detection model
noise 0.309 | dist 15 | thr 0.9 | P 0.998 R 0.950 F1 0.973
noise 0.309 | dist 15 | thr 0.95 | P 1.000 R 0.948 F1 0.973
noise 0.309 | dist 15 | thr 0.98 | P 1.000 R 0.944 F1 0.971
noise 0.309 | dist 15 | thr 0.99 | P 1.000 R 0.934 F1 0.966
noise 0.309 | dist 25 | thr 0.9 | P 0.998 R 0.950 F1 0.973
noise 0.309 | dist 25 | thr 0.95 | P 1.000 R 0.948 F1 0.973
noise 0.309 | dist 25 | thr 0.98 | P 1.000 R 0.944 F1 0.971
noise 0.309 | dist 25 | thr 0.99 | P 1.000 R 0.934 F1 0.966
noise 0.309 | dist 40 | thr 0.9 | P 0.998 R 0.946 F1 0.971
noise 0.309 | dist 40 | thr 0.95 | P 1.000 R 0.944 F1 0.971
noise 0.309 | dist 40 | thr 0.98 | P 1.000 R 0.940 F1 0.969
noise 0.309 | dist 40 | thr 0.99 | P 1.000 R 0.930 F1 0.964
noise 0.616 | dist 15 | thr 0.9 | P 0.992 R 0.940 F1 0.965
noise 0.616 | dist 15 | thr 0.95 | P 0.994 R 0.934 F1 0.963
noise 0.616 | dist 15 | thr 0.98 | P 1.000 R 0.911 F1 0.954
noise 0.616 | dist 15 | thr 0.99 | P 1.000 R 0.877 F1 0.935
noise 0.616 | dist 25 | thr 0.9 | P 0.991 R 0.938 F1 0.964
noise 0.616 | dist 25 | thr 0.95 | P 0.994 R 0.932 F1 0.962
noise 0.616 | dist 25 | thr 0.98 | P 1.000 R 0.909 F1 0.953
noise 0.616 | dist 25 | thr 0.99 | P 1.000 R 0.875 F1 0.933
noise 0.616 | dist 40 | thr 0.9 | P 0.991 R 0.934 F1 0.962
noise 0.616 | dist 40 | thr 0.95 | P 0.994 R 0.928 F1 0.959
noise 0.616 | dist 40 | thr 0.98 | P 1.000 R 0.905 F1 0.950
noise 0.616 | dist 40 | thr 0.99 | P 1.000 R 0.871 F1 0.931
noise 0.891 | dist 15 | thr 0.9 | P 0.949 R 0.905 F1 0.927
noise 0.891 | dist 15 | thr 0.95 | P 0.980 R 0.877 F1 0.926
noise 0.891 | dist 15 | thr 0.98 | P 0.998 R 0.843 F1 0.914
noise 0.891 | dist 15 | thr 0.99 | P 1.000 R 0.817 F1 0.899
noise 0.891 | dist 25 | thr 0.9 | P 0.951 R 0.905 F1 0.928
noise 0.891 | dist 25 | thr 0.95 | P 0.980 R 0.877 F1 0.926
noise 0.891 | dist 25 | thr 0.98 | P 0.998 R 0.843 F1 0.914
noise 0.891 | dist 25 | thr 0.99 | P 1.000 R 0.817 F1 0.899
noise 0.891 | dist 40 | thr 0.9 | P 0.951 R 0.901 F1 0.926
noise 0.891 | dist 40 | thr 0.95 | P 0.980 R 0.871 F1 0.922
noise 0.891 | dist 40 | thr 0.98 | P 0.998 R 0.837 F1 0.910
noise 0.891 | dist 40 | thr 0.99 | P 1.000 R 0.811 F1 0.896
noise 1.487 | dist 15 | thr 0.9 | P 0.930 R 0.779 F1 0.848
noise 1.487 | dist 15 | thr 0.95 | P 0.967 R 0.759 F1 0.850
noise 1.487 | dist 15 | thr 0.98 | P 0.989 R 0.746 F1 0.851
noise 1.487 | dist 15 | thr 0.99 | P 0.997 R 0.738 F1 0.849
noise 1.487 | dist 25 | thr 0.9 | P 0.941 R 0.775 F1 0.850
noise 1.487 | dist 25 | thr 0.95 | P 0.977 R 0.755 F1 0.851
noise 1.487 | dist 25 | thr 0.98 | P 0.989 R 0.742 F1 0.848
noise 1.487 | dist 25 | thr 0.99 | P 0.997 R 0.736 F1 0.847
noise 1.487 | dist 40 | thr 0.9 | P 0.941 R 0.771 F1 0.847
noise 1.487 | dist 40 | thr 0.95 | P 0.976 R 0.751 F1 0.849
noise 1.487 | dist 40 | thr 0.98 | P 0.989 R 0.738 F1 0.846
noise 1.487 | dist 40 | thr 0.99 | P 0.997 R 0.732 F1 0.845
noise 2.965 | dist 15 | thr 0.9 | P 0.917 R 0.670 F1 0.774
noise 2.965 | dist 15 | thr 0.95 | P 0.958 R 0.646 F1 0.772
noise 2.965 | dist 15 | thr 0.98 | P 0.987 R 0.604 F1 0.749
noise 2.965 | dist 15 | thr 0.99 | P 1.000 R 0.584 F1 0.737
noise 2.965 | dist 25 | thr 0.9 | P 0.935 R 0.664 F1 0.776
noise 2.965 | dist 25 | thr 0.95 | P 0.967 R 0.644 F1 0.773
noise 2.965 | dist 25 | thr 0.98 | P 0.987 R 0.602 F1 0.747
noise 2.965 | dist 25 | thr 0.99 | P 1.000 R 0.584 F1 0.737
noise 2.965 | dist 40 | thr 0.9 | P 0.935 R 0.662 F1 0.775
noise 2.965 | dist 40 | thr 0.95 | P 0.967 R 0.642 F1 0.771
noise 2.965 | dist 40 | thr 0.98 | P 0.987 R 0.602 F1 0.747
noise 2.965 | dist 40 | thr 0.99 | P 1.000 R 0.584 F1 0.737
noise 4.415 | dist 15 | thr 0.9 | P 0.776 R 0.614 F1 0.685
noise 4.415 | dist 15 | thr 0.95 | P 0.857 R 0.565 F1 0.681
noise 4.415 | dist 15 | thr 0.98 | P 0.920 R 0.533 F1 0.675
noise 4.415 | dist 15 | thr 0.99 | P 0.940 R 0.471 F1 0.627
noise 4.415 | dist 25 | thr 0.9 | P 0.786 R 0.612 F1 0.688
noise 4.415 | dist 25 | thr 0.95 | P 0.862 R 0.563 F1 0.681
noise 4.415 | dist 25 | thr 0.98 | P 0.923 R 0.531 F1 0.674
noise 4.415 | dist 25 | thr 0.99 | P 0.944 R 0.471 F1 0.628
noise 4.415 | dist 40 | thr 0.9 | P 0.797 R 0.608 F1 0.689
noise 4.415 | dist 40 | thr 0.95 | P 0.868 R 0.557 F1 0.679
noise 4.415 | dist 40 | thr 0.98 | P 0.926 R 0.527 F1 0.672
noise 4.415 | dist 40 | thr 0.99 | P 0.947 R 0.469 F1 0.627

(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
noise 0.309 | dist 15 | thr 0.3 | Precision 0.971 Recall 0.952 F1 0.961
noise 0.309 | dist 15 | thr 0.4 | Precision 0.977 Recall 0.952 F1 0.964
noise 0.309 | dist 15 | thr 0.5 | Precision 0.981 Recall 0.950 F1 0.965
noise 0.309 | dist 15 | thr 0.6 | Precision 0.981 Recall 0.950 F1 0.965
noise 0.309 | dist 15 | thr 0.7 | Precision 0.987 Recall 0.950 F1 0.968
noise 0.309 | dist 15 | thr 0.8 | Precision 0.998 Recall 0.948 F1 0.972
noise 0.309 | dist 15 | thr 0.9 | Precision 1.000 Recall 0.946 F1 0.972
noise 0.309 | dist 25 | thr 0.3 | Precision 0.973 Recall 0.948 F1 0.960
noise 0.309 | dist 25 | thr 0.4 | Precision 0.979 Recall 0.948 F1 0.963
noise 0.309 | dist 25 | thr 0.5 | Precision 0.983 Recall 0.948 F1 0.965
noise 0.309 | dist 25 | thr 0.6 | Precision 0.983 Recall 0.948 F1 0.965
noise 0.309 | dist 25 | thr 0.7 | Precision 0.987 Recall 0.948 F1 0.967
noise 0.309 | dist 25 | thr 0.8 | Precision 0.998 Recall 0.946 F1 0.971
noise 0.309 | dist 25 | thr 0.9 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.309 | dist 40 | thr 0.3 | Precision 0.973 Recall 0.948 F1 0.960
noise 0.309 | dist 40 | thr 0.4 | Precision 0.979 Recall 0.948 F1 0.963
noise 0.309 | dist 40 | thr 0.5 | Precision 0.983 Recall 0.948 F1 0.965
noise 0.309 | dist 40 | thr 0.6 | Precision 0.983 Recall 0.948 F1 0.965
noise 0.309 | dist 40 | thr 0.7 | Precision 0.987 Recall 0.948 F1 0.967
noise 0.309 | dist 40 | thr 0.8 | Precision 0.998 Recall 0.946 F1 0.971
noise 0.309 | dist 40 | thr 0.9 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.616 | dist 15 | thr 0.3 | Precision 0.870 Recall 0.954 F1 0.910
noise 0.616 | dist 15 | thr 0.4 | Precision 0.903 Recall 0.952 F1 0.927
noise 0.616 | dist 15 | thr 0.5 | Precision 0.942 Recall 0.950 F1 0.946
noise 0.616 | dist 15 | thr 0.6 | Precision 0.975 Recall 0.946 F1 0.960
noise 0.616 | dist 15 | thr 0.7 | Precision 0.989 Recall 0.942 F1 0.965
noise 0.616 | dist 15 | thr 0.8 | Precision 0.994 Recall 0.934 F1 0.963
noise 0.616 | dist 15 | thr 0.9 | Precision 0.998 Recall 0.918 F1 0.956
noise 0.616 | dist 25 | thr 0.3 | Precision 0.871 Recall 0.950 F1 0.909
noise 0.616 | dist 25 | thr 0.4 | Precision 0.904 Recall 0.950 F1 0.926
noise 0.616 | dist 25 | thr 0.5 | Precision 0.944 Recall 0.946 F1 0.945
noise 0.616 | dist 25 | thr 0.6 | Precision 0.975 Recall 0.944 F1 0.959
noise 0.616 | dist 25 | thr 0.7 | Precision 0.989 Recall 0.940 F1 0.964
noise 0.616 | dist 25 | thr 0.8 | Precision 0.994 Recall 0.932 F1 0.962
noise 0.616 | dist 25 | thr 0.9 | Precision 0.998 Recall 0.915 F1 0.955
noise 0.616 | dist 40 | thr 0.3 | Precision 0.870 Recall 0.946 F1 0.906
noise 0.616 | dist 40 | thr 0.4 | Precision 0.904 Recall 0.946 F1 0.924
noise 0.616 | dist 40 | thr 0.5 | Precision 0.944 Recall 0.944 F1 0.944
noise 0.616 | dist 40 | thr 0.6 | Precision 0.975 Recall 0.942 F1 0.958
noise 0.616 | dist 40 | thr 0.7 | Precision 0.989 Recall 0.938 F1 0.963
noise 0.616 | dist 40 | thr 0.8 | Precision 0.994 Recall 0.930 F1 0.960
noise 0.616 | dist 40 | thr 0.9 | Precision 0.998 Recall 0.913 F1 0.954
noise 0.891 | dist 15 | thr 0.3 | Precision 0.633 Recall 0.954 F1 0.761
noise 0.891 | dist 15 | thr 0.4 | Precision 0.717 Recall 0.950 F1 0.817
noise 0.891 | dist 15 | thr 0.5 | Precision 0.788 Recall 0.944 F1 0.859
noise 0.891 | dist 15 | thr 0.6 | Precision 0.844 Recall 0.936 F1 0.887
noise 0.891 | dist 15 | thr 0.7 | Precision 0.890 Recall 0.924 F1 0.906
noise 0.891 | dist 15 | thr 0.8 | Precision 0.938 Recall 0.913 F1 0.926
noise 0.891 | dist 15 | thr 0.9 | Precision 0.991 Recall 0.871 F1 0.927
noise 0.891 | dist 25 | thr 0.3 | Precision 0.637 Recall 0.950 F1 0.763
noise 0.891 | dist 25 | thr 0.4 | Precision 0.720 Recall 0.946 F1 0.817
noise 0.891 | dist 25 | thr 0.5 | Precision 0.788 Recall 0.942 F1 0.858
noise 0.891 | dist 25 | thr 0.6 | Precision 0.844 Recall 0.934 F1 0.886
noise 0.891 | dist 25 | thr 0.7 | Precision 0.889 Recall 0.922 F1 0.905
noise 0.891 | dist 25 | thr 0.8 | Precision 0.938 Recall 0.911 F1 0.924
noise 0.891 | dist 25 | thr 0.9 | Precision 0.991 Recall 0.871 F1 0.927
noise 0.891 | dist 40 | thr 0.3 | Precision 0.643 Recall 0.948 F1 0.766
noise 0.891 | dist 40 | thr 0.4 | Precision 0.724 Recall 0.944 F1 0.819
noise 0.891 | dist 40 | thr 0.5 | Precision 0.788 Recall 0.940 F1 0.857
noise 0.891 | dist 40 | thr 0.6 | Precision 0.843 Recall 0.932 F1 0.885
noise 0.891 | dist 40 | thr 0.7 | Precision 0.889 Recall 0.920 F1 0.904
noise 0.891 | dist 40 | thr 0.8 | Precision 0.938 Recall 0.909 F1 0.923
noise 0.891 | dist 40 | thr 0.9 | Precision 0.991 Recall 0.869 F1 0.926
noise 1.487 | dist 15 | thr 0.3 | Precision 0.360 Recall 0.928 F1 0.519
noise 1.487 | dist 15 | thr 0.4 | Precision 0.441 Recall 0.911 F1 0.594
noise 1.487 | dist 15 | thr 0.5 | Precision 0.530 Recall 0.893 F1 0.666
noise 1.487 | dist 15 | thr 0.6 | Precision 0.634 Recall 0.875 F1 0.735
noise 1.487 | dist 15 | thr 0.7 | Precision 0.722 Recall 0.843 F1 0.778
noise 1.487 | dist 15 | thr 0.8 | Precision 0.839 Recall 0.819 F1 0.829
noise 1.487 | dist 15 | thr 0.9 | Precision 0.935 Recall 0.783 F1 0.852
noise 1.487 | dist 25 | thr 0.3 | Precision 0.372 Recall 0.918 F1 0.529
noise 1.487 | dist 25 | thr 0.4 | Precision 0.452 Recall 0.903 F1 0.603
noise 1.487 | dist 25 | thr 0.5 | Precision 0.537 Recall 0.885 F1 0.669
noise 1.487 | dist 25 | thr 0.6 | Precision 0.642 Recall 0.869 F1 0.738
noise 1.487 | dist 25 | thr 0.7 | Precision 0.727 Recall 0.837 F1 0.778
noise 1.487 | dist 25 | thr 0.8 | Precision 0.843 Recall 0.813 F1 0.828
noise 1.487 | dist 25 | thr 0.9 | Precision 0.942 Recall 0.781 F1 0.854
noise 1.487 | dist 40 | thr 0.3 | Precision 0.379 Recall 0.915 F1 0.536
noise 1.487 | dist 40 | thr 0.4 | Precision 0.459 Recall 0.901 F1 0.608
noise 1.487 | dist 40 | thr 0.5 | Precision 0.541 Recall 0.883 F1 0.671
noise 1.487 | dist 40 | thr 0.6 | Precision 0.641 Recall 0.867 F1 0.737
noise 1.487 | dist 40 | thr 0.7 | Precision 0.727 Recall 0.835 F1 0.777
noise 1.487 | dist 40 | thr 0.8 | Precision 0.843 Recall 0.811 F1 0.827
noise 1.487 | dist 40 | thr 0.9 | Precision 0.942 Recall 0.779 F1 0.852
noise 2.965 | dist 15 | thr 0.3 | Precision 0.314 Recall 0.861 F1 0.460
noise 2.965 | dist 15 | thr 0.4 | Precision 0.390 Recall 0.833 F1 0.531
noise 2.965 | dist 15 | thr 0.5 | Precision 0.488 Recall 0.807 F1 0.608
noise 2.965 | dist 15 | thr 0.6 | Precision 0.581 Recall 0.785 F1 0.668
noise 2.965 | dist 15 | thr 0.7 | Precision 0.678 Recall 0.763 F1 0.718
noise 2.965 | dist 15 | thr 0.8 | Precision 0.770 Recall 0.734 F1 0.752
noise 2.965 | dist 15 | thr 0.9 | Precision 0.901 Recall 0.698 F1 0.787
noise 2.965 | dist 25 | thr 0.3 | Precision 0.343 Recall 0.845 F1 0.488
noise 2.965 | dist 25 | thr 0.4 | Precision 0.415 Recall 0.817 F1 0.550
noise 2.965 | dist 25 | thr 0.5 | Precision 0.511 Recall 0.791 F1 0.621
noise 2.965 | dist 25 | thr 0.6 | Precision 0.603 Recall 0.769 F1 0.676
noise 2.965 | dist 25 | thr 0.7 | Precision 0.691 Recall 0.748 F1 0.719
noise 2.965 | dist 25 | thr 0.8 | Precision 0.786 Recall 0.724 F1 0.754
noise 2.965 | dist 25 | thr 0.9 | Precision 0.918 Recall 0.696 F1 0.792
noise 2.965 | dist 40 | thr 0.3 | Precision 0.358 Recall 0.843 F1 0.502
noise 2.965 | dist 40 | thr 0.4 | Precision 0.423 Recall 0.811 F1 0.556
noise 2.965 | dist 40 | thr 0.5 | Precision 0.519 Recall 0.789 F1 0.626
noise 2.965 | dist 40 | thr 0.6 | Precision 0.608 Recall 0.767 F1 0.678
noise 2.965 | dist 40 | thr 0.7 | Precision 0.693 Recall 0.746 F1 0.719
noise 2.965 | dist 40 | thr 0.8 | Precision 0.786 Recall 0.722 F1 0.753
noise 2.965 | dist 40 | thr 0.9 | Precision 0.918 Recall 0.694 F1 0.790
noise 4.415 | dist 15 | thr 0.3 | Precision 0.269 Recall 0.839 F1 0.407
noise 4.415 | dist 15 | thr 0.4 | Precision 0.342 Recall 0.797 F1 0.478
noise 4.415 | dist 15 | thr 0.5 | Precision 0.406 Recall 0.740 F1 0.524
noise 4.415 | dist 15 | thr 0.6 | Precision 0.499 Recall 0.704 F1 0.584
noise 4.415 | dist 15 | thr 0.7 | Precision 0.598 Recall 0.674 F1 0.634
noise 4.415 | dist 15 | thr 0.8 | Precision 0.710 Recall 0.636 F1 0.671
noise 4.415 | dist 15 | thr 0.9 | Precision 0.838 Recall 0.553 F1 0.667
noise 4.415 | dist 25 | thr 0.3 | Precision 0.307 Recall 0.807 F1 0.444
noise 4.415 | dist 25 | thr 0.4 | Precision 0.375 Recall 0.765 F1 0.504
noise 4.415 | dist 25 | thr 0.5 | Precision 0.438 Recall 0.716 F1 0.544
noise 4.415 | dist 25 | thr 0.6 | Precision 0.523 Recall 0.686 F1 0.594
noise 4.415 | dist 25 | thr 0.7 | Precision 0.622 Recall 0.660 F1 0.641
noise 4.415 | dist 25 | thr 0.8 | Precision 0.730 Recall 0.622 F1 0.672
noise 4.415 | dist 25 | thr 0.9 | Precision 0.852 Recall 0.543 F1 0.663
noise 4.415 | dist 40 | thr 0.3 | Precision 0.318 Recall 0.801 F1 0.455
noise 4.415 | dist 40 | thr 0.4 | Precision 0.388 Recall 0.765 F1 0.515
noise 4.415 | dist 40 | thr 0.5 | Precision 0.447 Recall 0.714 F1 0.550
noise 4.415 | dist 40 | thr 0.6 | Precision 0.529 Recall 0.686 F1 0.597
noise 4.415 | dist 40 | thr 0.7 | Precision 0.625 Recall 0.660 F1 0.642
noise 4.415 | dist 40 | thr 0.8 | Precision 0.732 Recall 0.622 F1 0.672
noise 4.415 | dist 40 | thr 0.9 | Precision 0.852 Recall 0.543 F1 0.663

(0.85, 0.9, 0.93, 0.95, 0.97, 0.98)
noise 0.309 | dist 15 | thr 0.85 | Precision 1.000 Recall 0.946 F1 0.972
noise 0.309 | dist 15 | thr 0.9 | Precision 1.000 Recall 0.946 F1 0.972
noise 0.309 | dist 15 | thr 0.93 | Precision 1.000 Recall 0.938 F1 0.968
noise 0.309 | dist 15 | thr 0.95 | Precision 1.000 Recall 0.930 F1 0.964
noise 0.309 | dist 15 | thr 0.97 | Precision 1.000 Recall 0.924 F1 0.960
noise 0.309 | dist 15 | thr 0.98 | Precision 1.000 Recall 0.911 F1 0.954
noise 0.309 | dist 25 | thr 0.85 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.309 | dist 25 | thr 0.9 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.309 | dist 25 | thr 0.93 | Precision 1.000 Recall 0.936 F1 0.967
noise 0.309 | dist 25 | thr 0.95 | Precision 1.000 Recall 0.930 F1 0.964
noise 0.309 | dist 25 | thr 0.97 | Precision 1.000 Recall 0.924 F1 0.960
noise 0.309 | dist 25 | thr 0.98 | Precision 1.000 Recall 0.911 F1 0.954
noise 0.309 | dist 40 | thr 0.85 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.309 | dist 40 | thr 0.9 | Precision 1.000 Recall 0.944 F1 0.971
noise 0.309 | dist 40 | thr 0.93 | Precision 1.000 Recall 0.936 F1 0.967
noise 0.309 | dist 40 | thr 0.95 | Precision 1.000 Recall 0.930 F1 0.964
noise 0.309 | dist 40 | thr 0.97 | Precision 1.000 Recall 0.924 F1 0.960
noise 0.309 | dist 40 | thr 0.98 | Precision 1.000 Recall 0.911 F1 0.954
noise 0.616 | dist 15 | thr 0.85 | Precision 0.996 Recall 0.928 F1 0.960
noise 0.616 | dist 15 | thr 0.9 | Precision 0.998 Recall 0.918 F1 0.956
noise 0.616 | dist 15 | thr 0.93 | Precision 1.000 Recall 0.903 F1 0.949
noise 0.616 | dist 15 | thr 0.95 | Precision 1.000 Recall 0.879 F1 0.936
noise 0.616 | dist 15 | thr 0.97 | Precision 1.000 Recall 0.837 F1 0.911
noise 0.616 | dist 15 | thr 0.98 | Precision 1.000 Recall 0.787 F1 0.881
noise 0.616 | dist 25 | thr 0.85 | Precision 0.996 Recall 0.926 F1 0.959
noise 0.616 | dist 25 | thr 0.9 | Precision 0.998 Recall 0.915 F1 0.955
noise 0.616 | dist 25 | thr 0.93 | Precision 1.000 Recall 0.901 F1 0.948
noise 0.616 | dist 25 | thr 0.95 | Precision 1.000 Recall 0.875 F1 0.933
noise 0.616 | dist 25 | thr 0.97 | Precision 1.000 Recall 0.835 F1 0.910
noise 0.616 | dist 25 | thr 0.98 | Precision 1.000 Recall 0.785 F1 0.879
noise 0.616 | dist 40 | thr 0.85 | Precision 0.996 Recall 0.924 F1 0.958
noise 0.616 | dist 40 | thr 0.9 | Precision 0.998 Recall 0.913 F1 0.954
noise 0.616 | dist 40 | thr 0.93 | Precision 1.000 Recall 0.899 F1 0.947
noise 0.616 | dist 40 | thr 0.95 | Precision 1.000 Recall 0.875 F1 0.933
noise 0.616 | dist 40 | thr 0.97 | Precision 1.000 Recall 0.835 F1 0.910
noise 0.616 | dist 40 | thr 0.98 | Precision 1.000 Recall 0.785 F1 0.879
noise 0.891 | dist 15 | thr 0.85 | Precision 0.978 Recall 0.895 F1 0.935
noise 0.891 | dist 15 | thr 0.9 | Precision 0.991 Recall 0.871 F1 0.927
noise 0.891 | dist 15 | thr 0.93 | Precision 1.000 Recall 0.849 F1 0.918
noise 0.891 | dist 15 | thr 0.95 | Precision 1.000 Recall 0.821 F1 0.902
noise 0.891 | dist 15 | thr 0.97 | Precision 1.000 Recall 0.781 F1 0.877
noise 0.891 | dist 15 | thr 0.98 | Precision 1.000 Recall 0.761 F1 0.864
noise 0.891 | dist 25 | thr 0.85 | Precision 0.978 Recall 0.895 F1 0.935
noise 0.891 | dist 25 | thr 0.9 | Precision 0.991 Recall 0.871 F1 0.927
noise 0.891 | dist 25 | thr 0.93 | Precision 1.000 Recall 0.849 F1 0.918
noise 0.891 | dist 25 | thr 0.95 | Precision 1.000 Recall 0.821 F1 0.902
noise 0.891 | dist 25 | thr 0.97 | Precision 1.000 Recall 0.781 F1 0.877
noise 0.891 | dist 25 | thr 0.98 | Precision 1.000 Recall 0.761 F1 0.864
noise 0.891 | dist 40 | thr 0.85 | Precision 0.978 Recall 0.893 F1 0.934
noise 0.891 | dist 40 | thr 0.9 | Precision 0.991 Recall 0.869 F1 0.926
noise 0.891 | dist 40 | thr 0.93 | Precision 1.000 Recall 0.847 F1 0.917
noise 0.891 | dist 40 | thr 0.95 | Precision 1.000 Recall 0.819 F1 0.900
noise 0.891 | dist 40 | thr 0.97 | Precision 1.000 Recall 0.781 F1 0.877
noise 0.891 | dist 40 | thr 0.98 | Precision 1.000 Recall 0.761 F1 0.864
noise 1.487 | dist 15 | thr 0.85 | Precision 0.887 Recall 0.809 F1 0.846
noise 1.487 | dist 15 | thr 0.9 | Precision 0.935 Recall 0.783 F1 0.852
noise 1.487 | dist 15 | thr 0.93 | Precision 0.955 Recall 0.771 F1 0.853
noise 1.487 | dist 15 | thr 0.95 | Precision 0.984 Recall 0.755 F1 0.854
noise 1.487 | dist 15 | thr 0.97 | Precision 0.995 Recall 0.744 F1 0.852
noise 1.487 | dist 15 | thr 0.98 | Precision 1.000 Recall 0.736 F1 0.848
noise 1.487 | dist 25 | thr 0.85 | Precision 0.893 Recall 0.805 F1 0.847
noise 1.487 | dist 25 | thr 0.9 | Precision 0.942 Recall 0.781 F1 0.854
noise 1.487 | dist 25 | thr 0.93 | Precision 0.960 Recall 0.769 F1 0.854
noise 1.487 | dist 25 | thr 0.95 | Precision 0.987 Recall 0.753 F1 0.854
noise 1.487 | dist 25 | thr 0.97 | Precision 0.997 Recall 0.742 F1 0.851
noise 1.487 | dist 25 | thr 0.98 | Precision 1.000 Recall 0.736 F1 0.848
noise 1.487 | dist 40 | thr 0.85 | Precision 0.893 Recall 0.803 F1 0.845
noise 1.487 | dist 40 | thr 0.9 | Precision 0.942 Recall 0.779 F1 0.852
noise 1.487 | dist 40 | thr 0.93 | Precision 0.960 Recall 0.767 F1 0.852
noise 1.487 | dist 40 | thr 0.95 | Precision 0.987 Recall 0.751 F1 0.853
noise 1.487 | dist 40 | thr 0.97 | Precision 0.997 Recall 0.740 F1 0.850
noise 1.487 | dist 40 | thr 0.98 | Precision 1.000 Recall 0.734 F1 0.847
noise 2.965 | dist 15 | thr 0.85 | Precision 0.837 Recall 0.712 F1 0.770
noise 2.965 | dist 15 | thr 0.9 | Precision 0.901 Recall 0.698 F1 0.787
noise 2.965 | dist 15 | thr 0.93 | Precision 0.922 Recall 0.668 F1 0.775
noise 2.965 | dist 15 | thr 0.95 | Precision 0.964 Recall 0.646 F1 0.773
noise 2.965 | dist 15 | thr 0.97 | Precision 0.978 Recall 0.616 F1 0.756
noise 2.965 | dist 15 | thr 0.98 | Precision 0.993 Recall 0.561 F1 0.717
noise 2.965 | dist 25 | thr 0.85 | Precision 0.854 Recall 0.708 F1 0.774
noise 2.965 | dist 25 | thr 0.9 | Precision 0.918 Recall 0.696 F1 0.792
noise 2.965 | dist 25 | thr 0.93 | Precision 0.938 Recall 0.666 F1 0.779
noise 2.965 | dist 25 | thr 0.95 | Precision 0.976 Recall 0.644 F1 0.776
noise 2.965 | dist 25 | thr 0.97 | Precision 0.984 Recall 0.614 F1 0.756
noise 2.965 | dist 25 | thr 0.98 | Precision 0.993 Recall 0.559 F1 0.716
noise 2.965 | dist 40 | thr 0.85 | Precision 0.854 Recall 0.706 F1 0.773
noise 2.965 | dist 40 | thr 0.9 | Precision 0.918 Recall 0.694 F1 0.790
noise 2.965 | dist 40 | thr 0.93 | Precision 0.938 Recall 0.664 F1 0.777
noise 2.965 | dist 40 | thr 0.95 | Precision 0.976 Recall 0.642 F1 0.774
noise 2.965 | dist 40 | thr 0.97 | Precision 0.984 Recall 0.614 F1 0.756
noise 2.965 | dist 40 | thr 0.98 | Precision 0.993 Recall 0.559 F1 0.716
noise 4.415 | dist 15 | thr 0.85 | Precision 0.784 Recall 0.608 F1 0.685
noise 4.415 | dist 15 | thr 0.9 | Precision 0.838 Recall 0.553 F1 0.667
noise 4.415 | dist 15 | thr 0.93 | Precision 0.904 Recall 0.533 F1 0.671
noise 4.415 | dist 15 | thr 0.95 | Precision 0.941 Recall 0.509 F1 0.661
noise 4.415 | dist 15 | thr 0.97 | Precision 0.945 Recall 0.453 F1 0.612
noise 4.415 | dist 15 | thr 0.98 | Precision 0.954 Recall 0.414 F1 0.578
noise 4.415 | dist 25 | thr 0.85 | Precision 0.804 Recall 0.594 F1 0.683
noise 4.415 | dist 25 | thr 0.9 | Precision 0.852 Recall 0.543 F1 0.663
noise 4.415 | dist 25 | thr 0.93 | Precision 0.919 Recall 0.523 F1 0.667
noise 4.415 | dist 25 | thr 0.95 | Precision 0.950 Recall 0.499 F1 0.654
noise 4.415 | dist 25 | thr 0.97 | Precision 0.953 Recall 0.449 F1 0.610
noise 4.415 | dist 25 | thr 0.98 | Precision 0.962 Recall 0.410 F1 0.575
noise 4.415 | dist 40 | thr 0.85 | Precision 0.804 Recall 0.594 F1 0.683
noise 4.415 | dist 40 | thr 0.9 | Precision 0.852 Recall 0.543 F1 0.663
noise 4.415 | dist 40 | thr 0.93 | Precision 0.919 Recall 0.523 F1 0.667
noise 4.415 | dist 40 | thr 0.95 | Precision 0.950 Recall 0.499 F1 0.654
noise 4.415 | dist 40 | thr 0.97 | Precision 0.953 Recall 0.449 F1 0.610
noise 4.415 | dist 40 | thr 0.98 | Precision 0.962 Recall 0.410 F1 0.575



showing class3 low recall percentage
D4 noise 1.082 | dist 40 | thr 0.95 | F_ident 0.896 F_class 0.861 F_dataset 0.871
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.259
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.97 | F_ident 0.880 F_class 0.844 F_dataset 0.854
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.138
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.98 | F_ident 0.863 F_class 0.826 F_dataset 0.837
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.034
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.99 | F_ident 0.846 F_class 0.808 F_dataset 0.820
   class 1: detector recall 0.988
   class 2: detector recall 0.968
   class 3: detector recall 0.284
   class 4: detector recall 0.944
   class 5: detector recall 0.938
D5 noise 2.073 | dist 15 | thr 0.7 | F_ident 0.776 F_class 0.632 F_dataset 0.675
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.172
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.85 | F_ident 0.834 F_class 0.675 F_dataset 0.722
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.155
   class 4: detector recall 0.935
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.9 | F_ident 0.848 F_class 0.685 F_dataset 0.734
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.112
   class 4: detector recall 0.925
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.93 | F_ident 0.849 F_class 0.692 F_dataset 0.739
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.095
   class 4: detector recall 0.907
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.95 | F_ident 0.844 F_class 0.691 F_dataset 0.737
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.060
   class 4: detector recall 0.907
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.97 | F_ident 0.845 F_class 0.692 F_dataset 0.738
   class 1: detector recall 0.964
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.860
   class 5: detector recall 0.918
D5 noise 2.073 | dist 15 | thr 0.98 | F_ident 0.833 F_class 0.685 F_dataset 0.730
   class 1: detector recall 0.940
   class 2: detector recall 0.957
   class 3: detector recall 0.034
   class 4: detector recall 0.701
   class 5: detector recall 0.907
D5 noise 2.073 | dist 15 | thr 0.99 | F_ident 0.805 F_class 0.661 F_dataset 0.704
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.276
   class 4: detector recall 0.935
   class 5: detector recall 0.938
D5 noise 2.073 | dist 25 | thr 0.7 | F_ident 0.781 F_class 0.642 F_dataset 0.684
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.164
   class 4: detector recall 0.935
   class 5: detector recall 0.928
D5 noise 2.073 | dist 25 | thr 0.85 | F_ident 0.839 F_class 0.684 F_dataset 0.731
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.147
   class 4: detector recall 0.925
   class 5: detector recall 0.928
D5 noise 2.073 | dist 25 | thr 0.9 | F_ident 0.852 F_class 0.694 F_dataset 0.741
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.103
   class 4: detector recall 0.925
   class 5: detector recall 0.928
D5 noise 2.073 | dist 25 | thr 0.93 | F_ident 0.850 F_class 0.695 F_dataset 0.741
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.095
   class 4: detector recall 0.907
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.95 | F_ident 0.846 F_class 0.690 F_dataset 0.737
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.060
   class 4: detector recall 0.907
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.97 | F_ident 0.845 F_class 0.689 F_dataset 0.736
   class 1: detector recall 0.964
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.860
   class 5: detector recall 0.907
D5 noise 2.073 | dist 25 | thr 0.98 | F_ident 0.832 F_class 0.682 F_dataset 0.727
   class 1: detector recall 0.940
   class 2: detector recall 0.957
   class 3: detector recall 0.034
   class 4: detector recall 0.701
   class 5: detector recall 0.907
D5 noise 2.073 | dist 25 | thr 0.99 | F_ident 0.805 F_class 0.661 F_dataset 0.704
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.276
   class 4: detector recall 0.935
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.7 | F_ident 0.779 F_class 0.638 F_dataset 0.680
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.164
   class 4: detector recall 0.935
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.85 | F_ident 0.838 F_class 0.681 F_dataset 0.728
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.147
   class 4: detector recall 0.925
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.9 | F_ident 0.851 F_class 0.690 F_dataset 0.738
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.103
   class 4: detector recall 0.925
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.93 | F_ident 0.849 F_class 0.691 F_dataset 0.738
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.095
   class 4: detector recall 0.907
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.95 | F_ident 0.846 F_class 0.690 F_dataset 0.737
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.060
   class 4: detector recall 0.907
   class 5: detector recall 0.918
D5 noise 2.073 | dist 40 | thr 0.97 | F_ident 0.843 F_class 0.688 F_dataset 0.734
   class 1: detector recall 0.964
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.860
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.98 | F_ident 0.831 F_class 0.681 F_dataset 0.726
   class 1: detector recall 0.940
   class 2: detector recall 0.947
   class 3: detector recall 0.034
   class 4: detector recall 0.701
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.99 | F_ident 0.804 F_class 0.659 F_dataset 0.703
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.259
   class 4: detector recall 0.860
   class 5: detector recall 0.948
D6 noise 3.065 | dist 15 | thr 0.7 | F_ident 0.737 F_class 0.508 F_dataset 0.577
   class 1: detector recall 0.940
   class 2: detector recall 0.957
   class 3: detector recall 0.138
   class 4: detector recall 0.748
   class 5: detector recall 0.928
D6 noise 3.065 | dist 15 | thr 0.85 | F_ident 0.786 F_class 0.544 F_dataset 0.616
   class 1: detector recall 0.928
   class 2: detector recall 0.957
   class 3: detector recall 0.078
   class 4: detector recall 0.692
   class 5: detector recall 0.928
D6 noise 3.065 | dist 15 | thr 0.9 | F_ident 0.784 F_class 0.551 F_dataset 0.621
   class 1: detector recall 0.880
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.682
   class 5: detector recall 0.928
D6 noise 3.065 | dist 15 | thr 0.93 | F_ident 0.778 F_class 0.552 F_dataset 0.620
   class 1: detector recall 0.867
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.636
   class 5: detector recall 0.918
D6 noise 3.065 | dist 15 | thr 0.95 | F_ident 0.777 F_class 0.549 F_dataset 0.618
   class 1: detector recall 0.819
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.523
   class 5: detector recall 0.907
D6 noise 3.065 | dist 15 | thr 0.97 | F_ident 0.757 F_class 0.532 F_dataset 0.600
   class 1: detector recall 0.807
   class 2: detector recall 0.936
   class 3: detector recall 0.043
   class 4: detector recall 0.467
   class 5: detector recall 0.897
D6 noise 3.065 | dist 15 | thr 0.98 | F_ident 0.746 F_class 0.520 F_dataset 0.588
   class 1: detector recall 0.663
   class 2: detector recall 0.904
   class 3: detector recall 0.026
   class 4: detector recall 0.402
   class 5: detector recall 0.897
D6 noise 3.065 | dist 15 | thr 0.99 | F_ident 0.709 F_class 0.496 F_dataset 0.560
   class 1: detector recall 0.964
   class 2: detector recall 0.947
   class 3: detector recall 0.233
   class 4: detector recall 0.850
   class 5: detector recall 0.928
D6 noise 3.065 | dist 25 | thr 0.7 | F_ident 0.741 F_class 0.515 F_dataset 0.583
   class 1: detector recall 0.928
   class 2: detector recall 0.947
   class 3: detector recall 0.129
   class 4: detector recall 0.748
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.85 | F_ident 0.789 F_class 0.548 F_dataset 0.620
   class 1: detector recall 0.916
   class 2: detector recall 0.947
   class 3: detector recall 0.078
   class 4: detector recall 0.692
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.9 | F_ident 0.785 F_class 0.552 F_dataset 0.622
   class 1: detector recall 0.880
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.682
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.93 | F_ident 0.780 F_class 0.552 F_dataset 0.620
   class 1: detector recall 0.867
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.636
   class 5: detector recall 0.907
D6 noise 3.065 | dist 25 | thr 0.95 | F_ident 0.777 F_class 0.548 F_dataset 0.616
   class 1: detector recall 0.819
   class 2: detector recall 0.936
   class 3: detector recall 0.043
   class 4: detector recall 0.523
   class 5: detector recall 0.897
D6 noise 3.065 | dist 25 | thr 0.97 | F_ident 0.756 F_class 0.530 F_dataset 0.598
   class 1: detector recall 0.807
   class 2: detector recall 0.936
   class 3: detector recall 0.043
   class 4: detector recall 0.467
   class 5: detector recall 0.897
D6 noise 3.065 | dist 25 | thr 0.98 | F_ident 0.746 F_class 0.520 F_dataset 0.588
   class 1: detector recall 0.663
   class 2: detector recall 0.904
   class 3: detector recall 0.026
   class 4: detector recall 0.402
   class 5: detector recall 0.897
D6 noise 3.065 | dist 25 | thr 0.99 | F_ident 0.709 F_class 0.496 F_dataset 0.560
   class 1: detector recall 0.964
   class 2: detector recall 0.947
   class 3: detector recall 0.233
   class 4: detector recall 0.850
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.7 | F_ident 0.743 F_class 0.514 F_dataset 0.583
   class 1: detector recall 0.928
   class 2: detector recall 0.947
   class 3: detector recall 0.129
   class 4: detector recall 0.748
   class 5: detector recall 0.907
D6 noise 3.065 | dist 40 | thr 0.85 | F_ident 0.788 F_class 0.546 F_dataset 0.619
   class 1: detector recall 0.916
   class 2: detector recall 0.947
   class 3: detector recall 0.078
   class 4: detector recall 0.692
   class 5: detector recall 0.907
D6 noise 3.065 | dist 40 | thr 0.9 | F_ident 0.783 F_class 0.550 F_dataset 0.620
   class 1: detector recall 0.880
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.682
   class 5: detector recall 0.907
D6 noise 3.065 | dist 40 | thr 0.93 | F_ident 0.778 F_class 0.550 F_dataset 0.619
   class 1: detector recall 0.867
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.636
   class 5: detector recall 0.907
D6 noise 3.065 | dist 40 | thr 0.95 | F_ident 0.777 F_class 0.548 F_dataset 0.616
   class 1: detector recall 0.819
   class 2: detector recall 0.936
   class 3: detector recall 0.043
   class 4: detector recall 0.523
   class 5: detector recall 0.897
D6 noise 3.065 | dist 40 | thr 0.97 | F_ident 0.756 F_class 0.530 F_dataset 0.598
   class 1: detector recall 0.807
   class 2: detector recall 0.936
   class 3: detector recall 0.043
   class 4: detector recall 0.467
   class 5: detector recall 0.897
D6 noise 3.065 | dist 40 | thr 0.98 | F_ident 0.746 F_class 0.520 F_dataset 0.588
   class 1: detector recall 0.663
   class 2: detector recall 0.904
   class 3: detector recall 0.026
   class 4: detector recall 0.402
   class 5: detector recall 0.897
D6 noise 3.065 | dist 40 | thr 0.99 | F_ident 0.709 F_class 0.496 F_dataset 0.560

improve by x3 the number of training for each class 3 signal
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.7 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.85 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.9 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.93 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.95 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.97 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.98 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D1 noise 0.222 | dist 15 | thr 0.99 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.7 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.85 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.9 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.93 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.95 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.97 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.98 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D1 noise 0.222 | dist 25 | thr 0.99 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.7 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.85 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.9 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.93 | F_ident 0.971 F_class 0.950 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.95 | F_ident 0.971 F_class 0.950 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.97 | F_ident 0.971 F_class 0.950 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.98 | F_ident 0.971 F_class 0.950 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D1 noise 0.222 | dist 40 | thr 0.99 | F_ident 0.971 F_class 0.950 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.7 | F_ident 0.973 F_class 0.953 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.85 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.9 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.93 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.95 | F_ident 0.974 F_class 0.954 F_dataset 0.960
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.97 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.98 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D2 noise 0.438 | dist 15 | thr 0.99 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.7 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.85 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.9 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.93 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.95 | F_ident 0.973 F_class 0.952 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.97 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.98 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D2 noise 0.438 | dist 25 | thr 0.99 | F_ident 0.972 F_class 0.951 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.7 | F_ident 0.971 F_class 0.948 F_dataset 0.955
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.85 | F_ident 0.972 F_class 0.949 F_dataset 0.956
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.9 | F_ident 0.972 F_class 0.949 F_dataset 0.956
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.93 | F_ident 0.971 F_class 0.948 F_dataset 0.955
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.948
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.95 | F_ident 0.971 F_class 0.948 F_dataset 0.955
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.97 | F_ident 0.970 F_class 0.947 F_dataset 0.954
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.98 | F_ident 0.970 F_class 0.947 F_dataset 0.954
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D2 noise 0.438 | dist 40 | thr 0.99 | F_ident 0.970 F_class 0.947 F_dataset 0.954
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.7 | F_ident 0.966 F_class 0.948 F_dataset 0.953
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.85 | F_ident 0.968 F_class 0.950 F_dataset 0.955
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.9 | F_ident 0.971 F_class 0.953 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.93 | F_ident 0.972 F_class 0.954 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.95 | F_ident 0.972 F_class 0.954 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.931
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.97 | F_ident 0.972 F_class 0.953 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.922
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.98 | F_ident 0.971 F_class 0.952 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.879
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 15 | thr 0.99 | F_ident 0.966 F_class 0.947 F_dataset 0.953
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.7 | F_ident 0.966 F_class 0.948 F_dataset 0.953
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.85 | F_ident 0.967 F_class 0.949 F_dataset 0.954
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.9 | F_ident 0.971 F_class 0.953 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.93 | F_ident 0.972 F_class 0.954 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.95 | F_ident 0.972 F_class 0.954 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.931
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.97 | F_ident 0.972 F_class 0.953 F_dataset 0.959
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.922
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.98 | F_ident 0.971 F_class 0.952 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.879
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D3 noise 0.648 | dist 25 | thr 0.99 | F_ident 0.966 F_class 0.947 F_dataset 0.953
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.7 | F_ident 0.964 F_class 0.946 F_dataset 0.951
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.85 | F_ident 0.966 F_class 0.948 F_dataset 0.953
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.9 | F_ident 0.970 F_class 0.951 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.93 | F_ident 0.971 F_class 0.952 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.940
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.95 | F_ident 0.971 F_class 0.952 F_dataset 0.958
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.931
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.97 | F_ident 0.970 F_class 0.951 F_dataset 0.957
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.922
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.98 | F_ident 0.969 F_class 0.950 F_dataset 0.956
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.879
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D3 noise 0.648 | dist 40 | thr 0.99 | F_ident 0.964 F_class 0.945 F_dataset 0.950
   class 1: detector recall 0.988
   class 2: detector recall 0.968
   class 3: detector recall 0.871
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.7 | F_ident 0.807 F_class 0.781 F_dataset 0.788
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.862
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.85 | F_ident 0.873 F_class 0.846 F_dataset 0.854
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.793
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.9 | F_ident 0.902 F_class 0.874 F_dataset 0.882
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.741
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.93 | F_ident 0.914 F_class 0.885 F_dataset 0.894
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.664
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.95 | F_ident 0.915 F_class 0.886 F_dataset 0.894
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.517
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.97 | F_ident 0.909 F_class 0.879 F_dataset 0.888
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.397
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.98 | F_ident 0.899 F_class 0.866 F_dataset 0.876
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.147
   class 4: detector recall 0.935
   class 5: detector recall 0.928
D4 noise 1.082 | dist 15 | thr 0.99 | F_ident 0.863 F_class 0.831 F_dataset 0.841
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.871
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.7 | F_ident 0.814 F_class 0.787 F_dataset 0.795
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.862
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.85 | F_ident 0.876 F_class 0.847 F_dataset 0.856
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.793
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.9 | F_ident 0.903 F_class 0.874 F_dataset 0.883
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.741
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.93 | F_ident 0.913 F_class 0.883 F_dataset 0.892
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.664
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.95 | F_ident 0.915 F_class 0.883 F_dataset 0.893
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.517
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.97 | F_ident 0.909 F_class 0.876 F_dataset 0.886
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.397
   class 4: detector recall 0.944
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.98 | F_ident 0.898 F_class 0.865 F_dataset 0.875
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.147
   class 4: detector recall 0.935
   class 5: detector recall 0.928
D4 noise 1.082 | dist 25 | thr 0.99 | F_ident 0.863 F_class 0.831 F_dataset 0.841
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.871
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.7 | F_ident 0.816 F_class 0.789 F_dataset 0.797
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.862
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.85 | F_ident 0.876 F_class 0.848 F_dataset 0.856
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.793
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.9 | F_ident 0.904 F_class 0.874 F_dataset 0.883
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.741
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.93 | F_ident 0.913 F_class 0.883 F_dataset 0.892
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.664
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.95 | F_ident 0.913 F_class 0.882 F_dataset 0.892
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.517
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.97 | F_ident 0.907 F_class 0.875 F_dataset 0.885
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.397
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.98 | F_ident 0.896 F_class 0.863 F_dataset 0.873
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.147
   class 4: detector recall 0.935
   class 5: detector recall 0.918
D4 noise 1.082 | dist 40 | thr 0.99 | F_ident 0.860 F_class 0.828 F_dataset 0.838
   class 1: detector recall 0.988
   class 2: detector recall 0.968
   class 3: detector recall 0.362
   class 4: detector recall 0.953
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.7 | F_ident 0.741 F_class 0.638 F_dataset 0.669
   class 1: detector recall 0.988
   class 2: detector recall 0.968
   class 3: detector recall 0.198
   class 4: detector recall 0.916
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.85 | F_ident 0.826 F_class 0.712 F_dataset 0.746
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.138
   class 4: detector recall 0.897
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.9 | F_ident 0.838 F_class 0.726 F_dataset 0.759
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.069
   class 4: detector recall 0.897
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.93 | F_ident 0.837 F_class 0.725 F_dataset 0.758
   class 1: detector recall 0.976
   class 2: detector recall 0.968
   class 3: detector recall 0.052
   class 4: detector recall 0.879
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.95 | F_ident 0.834 F_class 0.726 F_dataset 0.758
   class 1: detector recall 0.952
   class 2: detector recall 0.957
   class 3: detector recall 0.052
   class 4: detector recall 0.813
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.97 | F_ident 0.822 F_class 0.715 F_dataset 0.747
   class 1: detector recall 0.940
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.720
   class 5: detector recall 0.928
D5 noise 2.073 | dist 15 | thr 0.98 | F_ident 0.808 F_class 0.701 F_dataset 0.733
   class 1: detector recall 0.904
   class 2: detector recall 0.947
   class 3: detector recall 0.026
   class 4: detector recall 0.551
   class 5: detector recall 0.918
D5 noise 2.073 | dist 15 | thr 0.99 | F_ident 0.774 F_class 0.671 F_dataset 0.702
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.362
   class 4: detector recall 0.944
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.7 | F_ident 0.760 F_class 0.656 F_dataset 0.687
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.198
   class 4: detector recall 0.907
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.85 | F_ident 0.831 F_class 0.719 F_dataset 0.753
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.138
   class 4: detector recall 0.888
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.9 | F_ident 0.840 F_class 0.729 F_dataset 0.763
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.069
   class 4: detector recall 0.888
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.93 | F_ident 0.837 F_class 0.727 F_dataset 0.760
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.052
   class 4: detector recall 0.879
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.95 | F_ident 0.836 F_class 0.727 F_dataset 0.760
   class 1: detector recall 0.952
   class 2: detector recall 0.957
   class 3: detector recall 0.052
   class 4: detector recall 0.813
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.97 | F_ident 0.826 F_class 0.718 F_dataset 0.750
   class 1: detector recall 0.940
   class 2: detector recall 0.957
   class 3: detector recall 0.043
   class 4: detector recall 0.720
   class 5: detector recall 0.918
D5 noise 2.073 | dist 25 | thr 0.98 | F_ident 0.810 F_class 0.703 F_dataset 0.735
   class 1: detector recall 0.904
   class 2: detector recall 0.947
   class 3: detector recall 0.026
   class 4: detector recall 0.551
   class 5: detector recall 0.907
D5 noise 2.073 | dist 25 | thr 0.99 | F_ident 0.774 F_class 0.671 F_dataset 0.702
   class 1: detector recall 0.976
   class 2: detector recall 0.957
   class 3: detector recall 0.362
   class 4: detector recall 0.944
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.7 | F_ident 0.761 F_class 0.655 F_dataset 0.687
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.198
   class 4: detector recall 0.907
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.85 | F_ident 0.828 F_class 0.712 F_dataset 0.747
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.138
   class 4: detector recall 0.888
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.9 | F_ident 0.838 F_class 0.722 F_dataset 0.757
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.069
   class 4: detector recall 0.888
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.93 | F_ident 0.835 F_class 0.719 F_dataset 0.754
   class 1: detector recall 0.976
   class 2: detector recall 0.947
   class 3: detector recall 0.052
   class 4: detector recall 0.879
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.95 | F_ident 0.834 F_class 0.719 F_dataset 0.754
   class 1: detector recall 0.952
   class 2: detector recall 0.947
   class 3: detector recall 0.052
   class 4: detector recall 0.813
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.97 | F_ident 0.823 F_class 0.710 F_dataset 0.744
   class 1: detector recall 0.940
   class 2: detector recall 0.947
   class 3: detector recall 0.043
   class 4: detector recall 0.720
   class 5: detector recall 0.907
D5 noise 2.073 | dist 40 | thr 0.98 | F_ident 0.807 F_class 0.695 F_dataset 0.728
   class 1: detector recall 0.904
   class 2: detector recall 0.947
   class 3: detector recall 0.026
   class 4: detector recall 0.551
   class 5: detector recall 0.897
D5 noise 2.073 | dist 40 | thr 0.99 | F_ident 0.773 F_class 0.667 F_dataset 0.699
   class 1: detector recall 0.964
   class 2: detector recall 0.968
   class 3: detector recall 0.233
   class 4: detector recall 0.869
   class 5: detector recall 0.928
D6 noise 3.065 | dist 15 | thr 0.7 | F_ident 0.709 F_class 0.492 F_dataset 0.557
   class 1: detector recall 0.928
   class 2: detector recall 0.947
   class 3: detector recall 0.121
   class 4: detector recall 0.766
   class 5: detector recall 0.918
D6 noise 3.065 | dist 15 | thr 0.85 | F_ident 0.774 F_class 0.545 F_dataset 0.613
   class 1: detector recall 0.904
   class 2: detector recall 0.947
   class 3: detector recall 0.078
   class 4: detector recall 0.729
   class 5: detector recall 0.918
D6 noise 3.065 | dist 15 | thr 0.9 | F_ident 0.778 F_class 0.549 F_dataset 0.618
   class 1: detector recall 0.880
   class 2: detector recall 0.936
   class 3: detector recall 0.069
   class 4: detector recall 0.598
   class 5: detector recall 0.918
D6 noise 3.065 | dist 15 | thr 0.93 | F_ident 0.763 F_class 0.545 F_dataset 0.610
   class 1: detector recall 0.867
   class 2: detector recall 0.926
   class 3: detector recall 0.052
   class 4: detector recall 0.505
   class 5: detector recall 0.918
D6 noise 3.065 | dist 15 | thr 0.95 | F_ident 0.746 F_class 0.530 F_dataset 0.595
   class 1: detector recall 0.843
   class 2: detector recall 0.926
   class 3: detector recall 0.043
   class 4: detector recall 0.383
   class 5: detector recall 0.866
D6 noise 3.065 | dist 15 | thr 0.97 | F_ident 0.723 F_class 0.514 F_dataset 0.577
   class 1: detector recall 0.795
   class 2: detector recall 0.894
   class 3: detector recall 0.043
   class 4: detector recall 0.318
   class 5: detector recall 0.856
D6 noise 3.065 | dist 15 | thr 0.98 | F_ident 0.701 F_class 0.503 F_dataset 0.562
   class 1: detector recall 0.639
   class 2: detector recall 0.862
   class 3: detector recall 0.034
   class 4: detector recall 0.234
   class 5: detector recall 0.804
D6 noise 3.065 | dist 15 | thr 0.99 | F_ident 0.651 F_class 0.462 F_dataset 0.519
   class 1: detector recall 0.964
   class 2: detector recall 0.957
   class 3: detector recall 0.207
   class 4: detector recall 0.869
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.7 | F_ident 0.750 F_class 0.522 F_dataset 0.591
   class 1: detector recall 0.928
   class 2: detector recall 0.936
   class 3: detector recall 0.103
   class 4: detector recall 0.766
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.85 | F_ident 0.793 F_class 0.560 F_dataset 0.630
   class 1: detector recall 0.904
   class 2: detector recall 0.936
   class 3: detector recall 0.060
   class 4: detector recall 0.729
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.9 | F_ident 0.793 F_class 0.562 F_dataset 0.632
   class 1: detector recall 0.880
   class 2: detector recall 0.926
   class 3: detector recall 0.052
   class 4: detector recall 0.598
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.93 | F_ident 0.772 F_class 0.554 F_dataset 0.620
   class 1: detector recall 0.867
   class 2: detector recall 0.915
   class 3: detector recall 0.034
   class 4: detector recall 0.505
   class 5: detector recall 0.918
D6 noise 3.065 | dist 25 | thr 0.95 | F_ident 0.755 F_class 0.540 F_dataset 0.604
   class 1: detector recall 0.843
   class 2: detector recall 0.915
   class 3: detector recall 0.034
   class 4: detector recall 0.383
   class 5: detector recall 0.866
D6 noise 3.065 | dist 25 | thr 0.97 | F_ident 0.726 F_class 0.517 F_dataset 0.580
   class 1: detector recall 0.795
   class 2: detector recall 0.894
   class 3: detector recall 0.034
   class 4: detector recall 0.318
   class 5: detector recall 0.856
D6 noise 3.065 | dist 25 | thr 0.98 | F_ident 0.704 F_class 0.504 F_dataset 0.564
   class 1: detector recall 0.639
   class 2: detector recall 0.862
   class 3: detector recall 0.034
   class 4: detector recall 0.234
   class 5: detector recall 0.804
D6 noise 3.065 | dist 25 | thr 0.99 | F_ident 0.653 F_class 0.461 F_dataset 0.518
   class 1: detector recall 0.964
   class 2: detector recall 0.947
   class 3: detector recall 0.207
   class 4: detector recall 0.869
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.7 | F_ident 0.751 F_class 0.521 F_dataset 0.590
   class 1: detector recall 0.928
   class 2: detector recall 0.936
   class 3: detector recall 0.103
   class 4: detector recall 0.766
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.85 | F_ident 0.794 F_class 0.561 F_dataset 0.631
   class 1: detector recall 0.904
   class 2: detector recall 0.936
   class 3: detector recall 0.060
   class 4: detector recall 0.729
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.9 | F_ident 0.794 F_class 0.563 F_dataset 0.632
   class 1: detector recall 0.880
   class 2: detector recall 0.926
   class 3: detector recall 0.052
   class 4: detector recall 0.598
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.93 | F_ident 0.773 F_class 0.555 F_dataset 0.621
   class 1: detector recall 0.867
   class 2: detector recall 0.915
   class 3: detector recall 0.034
   class 4: detector recall 0.505
   class 5: detector recall 0.918
D6 noise 3.065 | dist 40 | thr 0.95 | F_ident 0.755 F_class 0.540 F_dataset 0.604
   class 1: detector recall 0.843
   class 2: detector recall 0.915
   class 3: detector recall 0.034
   class 4: detector recall 0.383
   class 5: detector recall 0.866
D6 noise 3.065 | dist 40 | thr 0.97 | F_ident 0.726 F_class 0.517 F_dataset 0.580
   class 1: detector recall 0.795
   class 2: detector recall 0.894
   class 3: detector recall 0.034
   class 4: detector recall 0.318
   class 5: detector recall 0.856
D6 noise 3.065 | dist 40 | thr 0.98 | F_ident 0.704 F_class 0.504 F_dataset 0.564
   class 1: detector recall 0.639
   class 2: detector recall 0.862
   class 3: detector recall 0.034
   class 4: detector recall 0.234
   class 5: detector recall 0.804
D6 noise 3.065 | dist 40 | thr 0.99 | F_ident 0.653 F_class 0.461 F_dataset 0.518


Conclusion setting
Dataset	Noise	thr	   dist	Val F1
D2	   0.616	0.7	    15	0.965 (from the 0.3-0.9 run)
D3	   0.891	0.85	15	0.935
D4	   1.487	0.93	25	0.854
D5	   2.965	0.9	    25	0.792
D6	   4.415	0.85	15	0.685