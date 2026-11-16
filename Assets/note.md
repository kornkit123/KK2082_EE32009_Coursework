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

Conclusion setting
Dataset	Noise	thr	   dist	Val F1
D2	   0.616	0.7	    15	0.965 (from the 0.3-0.9 run)
D3	   0.891	0.85	15	0.935
D4	   1.487	0.93	25	0.854
D5	   2.965	0.9	    25	0.792
D6	   4.415	0.85	15	0.685