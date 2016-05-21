for obs in {2..5}; do for lr in 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 0.9; do ./kitt_train.py -ta amter -ds amter_nn_00_40_alls_allt_500 -s 20 -lr $lr -i 300 -na $obs; done; done;

for obs in {2..5}; do for st in 5 7 10 15 20 30 50 100; do ./kitt_train.py -ta amter -ds amter_nn_00_40_alls_allt_500 -s $st -lr 0.5 -i 300 -na $obs; done; done;

