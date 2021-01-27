#!/bin/sh
# Tue Jan  5 17:34:25 2021 by Masaaki Nagata

DATA_DIR=./ldc

# 6099 = 4879 + 610 + 610

shuf $DATA_DIR/all.txt > $DATA_DIR/all.txt.shuf
head -610 $DATA_DIR/all.txt.shuf > $DATA_DIR/zhen_test.txt
tail -n +611 $DATA_DIR/all.txt.shuf | head -610 > $DATA_DIR/zhen_dev.txt
tail -n +1221 $DATA_DIR/all.txt.shuf > $DATA_DIR/zhen_train0.txt

head -300 $DATA_DIR/zhen_train0.txt > $DATA_DIR/zhen_train1.txt
head -600 $DATA_DIR/zhen_train0.txt > $DATA_DIR/zhen_train2.txt
head -1200 $DATA_DIR/zhen_train0.txt > $DATA_DIR/zhen_train3.txt
head -2400 $DATA_DIR/zhen_train0.txt > $DATA_DIR/zhen_train4.txt

for i in test dev train0 train1 train2 train3 train4
do
    echo $i
    wa2span_squad.py -w $DATA_DIR/zhen_$i.txt > $DATA_DIR/zhen_$i.json
done
