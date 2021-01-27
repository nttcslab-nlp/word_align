#!/bin/sh
# Tue Jan  5 17:03:52 2021 by Masaaki Nagata

LDC_DIR=/nfs/data/LDC/LDC2015T06/data/parallel_word_aligned_treebank/
TGT_DIR=./ldc

for d in bc nw wb
do
    echo $d
    mkdir -p $TGT_DIR/$d
    for f in $LDC_DIR/$d/source/raw/*
    do
	bn=`basename $f .raw`
	bn=`basename $bn .cmn`
	echo $bn
	paste \
	    $LDC_DIR/$d/source/character_tokenized/$bn.cmn.tkn \
	    $LDC_DIR/$d/translation/tokenized/$bn.eng.tkn \
	    $LDC_DIR/$d/WA/character_aligned/$bn.wa \
	    $LDC_DIR/$d/source/raw/$bn.cmn.raw \
	    $LDC_DIR/$d/translation/raw/$bn.eng.raw \
	    > $TGT_DIR/$d/$bn.ldc
	cat $TGT_DIR/$d/$bn.ldc | convert_to_moses.py > $TGT_DIR/$d/$bn.txt
    done
    
done

cat $TGT_DIR/{bc,nw,wb}/*.txt > $TGT_DIR/all.txt

	 
