#!/bin/sh
# Fri Nov 29 15:13:27 2019 by Masaaki Nagata
# modified Thu Nov 26 19:44:49 2020 by Masaaki Nagata
# usage: run_devtest1.sh

# BERT program
export BERT_DIR=./bert

# BERT pretrained model
export BERT_BASE_DIR=./multi_cased_L-12_H-768_A-12

# SQuAD evaluation script
export SQUAD_DIR=.

# FineTuning data
export FT_DIR=.
export TRAIN_FILE=$FT_DIR/kftt_dev.json
export DEVTEST_FILE=$FT_DIR/kftt_devtest.json

# OUTPUT_DIR
export OUTPUT_DIR=./squad-2.0
export TRAINED_DIR=$OUTPUT_DIR/dev1

# selector
DO_TRAIN=1
DO_EVALUATE=1

date
hostname
echo $CUDA_VISIBLE_DEVICES

if [ ${DO_TRAIN} -eq 1 ]
then
    echo DO_TRAIN
    mkdir -p ${TRAINED_DIR}
    python $BERT_DIR/my-run_squad.py \
	   --do_lower_case=False \
	   --vocab_file=$BERT_BASE_DIR/vocab.txt \
	   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
	   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	   --do_train=True \
	   --train_file=$TRAIN_FILE \
	   --do_predict=True \
	   --predict_file=$DEVTEST_FILE \
	   --train_batch_size=6 \
	   --learning_rate=3e-5 \
	   --num_train_epochs=2.0 \
	   --max_seq_length=384 \
	   --max_query_length=160 \
	   --max_answer_length=15 \
	   --doc_stride=128 \
	   --output_dir=$TRAINED_DIR \
	   --version_2_with_negative=True \
	   >& $TRAINED_DIR/log.train
fi

date

if [ ${DO_EVALUATE} -eq 1 -a -f $TRAINED_DIR/checkpoint ]
then
    echo DO_EVALUATE
    python $SQUAD_DIR/evaluate-v2.0.py $DEVTEST_FILE \
    	   $TRAINED_DIR/predictions.json \
    	   > $TRAINED_DIR/predictions.score
    python $SQUAD_DIR/evaluate-v2.0.py $DEVTEST_FILE \
    	   $TRAINED_DIR/predictions.json \
	   --na-prob-file $TRAINED_DIR/null_odds.json \
    	   > $TRAINED_DIR/predictions.score.na-prob
fi


date
