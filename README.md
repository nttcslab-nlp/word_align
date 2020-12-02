

# A Word Alignment software using multilingual BERT

This repository includes the software described in "[A Supervised Word Alignment Method based on Cross-Language Span Prediction using Multilingual BERT](https://www.aclweb.org/anthology/2020.emnlp-main.41/)" published at EMNLP-2020.

## Preparing Data

Download KFTT (Kyoto Free Translation Task) Japanese-English Alignment Data. Then expand it.

```
$ wget http://www.phontron.com/kftt/download/kftt-alignments.tar.gz
$ tar zxvf kftt-alignments.tar.gz
```

Split the data into three sets (dev, devtest, test). We used `dev` for fine-tuning and `devtest` for testing.

```
$ make_kftt_data.sh
```

Convert dev and devtest to SQuAD v2.0 json format.

```
$ wa2span_squad.py -l -w ./kftt_dev.txt > ./kftt_dev.json
$ wa2span_squad.py -l -w ./kftt_devtest.txt > ./kftt_devtest.json
```

## Preparing mulilingual BERT

Download BERT scripts from its github.

```
$ git clone https://github.com/google-research/bert
```

Then, move our 'my-run_squad.py' to the 'bert' directory. Our modification to the original run_squad.py is minimal. We simply add a code to output start and end position, which are the indexes to BERT tokens. You can check the difference by `diff run_squad.py my-run_squad.py`.

```
$ cp -p my-run_squad.py ./bert
```

Download 'BERT-Base, Multilingual Cased (New, recommended)' and unzip it.

```
$ wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
$ unzip multi_cased_L-12_H-768_A-12.zip
```

## Preparing SQuAD v2.0 evaluation script (optional)

Download the official evaluation script for SQuAD v2.0. It is useful to check the cross-language span prediction accuracy by this script.

```
$ wget https://raw.githubusercontent.com/white127/SQUAD-2.0-bidaf/master/evaluate-v2.0.py
```

## Installing TensorFlow-1.15 (optional)

For testing this software, we used anaconda3-2020.07, python-3.7.9, tensorflow-1.15.0, and cuda-10.0. It seems the original BERT does not work with TensorFlow 2.

```
$ conda create -n tensorflow python=3.7
$ conda activate tensorflow
(tensorflow) $ pip install tensorflow-gpu==1.15
```

## Fine-tuning and Run

The following script `run_devtest1.sh` do the SQuAD cross-language span predictions. It first specifies the location of BERT, input files, and output files. It then call `my-run_squad.py` and `evaluate-v2.0.py`. See the SQuAD v2.0 section of  [BERT github](https://github.com/google-research/bert) for parameters of `run_squad.py`. We used one GPU (rtx-2080ti) and lowered the batch size (train_batch_size=6) for this experiment. It took about an hour to run this script in our environment.

```
(tensorflow) $ run_devtest1.sh
```

You can obtain the cross-language span prediction accuracy using the official SQuAD V2.0 evaluation script.

```
$ cat squad-2.0/dev1/predictions.score
{
  "exact": 74.11018723455166,
  "f1": 76.00664089687204,
  "total": 16717,
  "HasAns_exact": 73.06879549189951,
  "HasAns_f1": 75.55005211497341,
  "HasAns_total": 12777,
  "NoAns_exact": 77.48730964467005,
  "NoAns_f1": 77.48730964467005,
  "NoAns_total": 3940
}
```

## Obtaining Word Alignment from Span Predictions

First, convert the indexes of start and end positions in the context, from the ones based on BERT tokens to the ones based on characters. 

```
(tensorflow) $ convert_start_end.py -q kftt_devtest.json -n ./squad-2.0/dev1/nbest_predictions.json -m 160 > charindex_nbest_predictions.json
```

Word alignment accuracy (recall, precision  and f1) can be calculated as follows. Here, `bidi-th` refers to 'bidirectional average' in our paper.

```
$ get_alignment.py -l -a ./kftt_devtest.txt -n charindex_nbest_predictions.json
f_to_e: 0.791 0.808 0.800 (8211 8041 6498)
e_to_f: 0.692 0.576 0.629 (8211 9862 5681)
bidi_int: 0.633 0.899 0.743 (8211 5781 5199)
bidi_uni: 0.850 0.576 0.687 (8211 12122 6980)
bidi_th: 0.796 0.733 0.763 (8211 8919 6535)
```

Word alignment in GIZA++ format is obtained as follows.

```
$ get_alignment.py -b -l -a ./kftt_devtest.txt -n ./charindex_nbest_predictions.json -m 160 > kftt_devtest.moses.bidi_th
```

Use your favorite software to display word alignments!

```
$ head -5 kftt_devtest.moses.bidi_th
0-1 1-0 3-1 4-0 7-9 8-10 9-7 10-4 11-4 12-4 13-5 14-6 15-6 17-12 18-14 18-15 19-14 19-15 20-15 21-14 21-15 22-14 22-15 24-2 25-2 26-2 27-16
0-0 0-1 2-3 3-3 4-3 5-4 5-6 5-7 5-8 5-9 6-4 6-6 7-11 8-12 9-12 10-12
0-0 0-1 0-2 2-6 3-4 4-4
0-0 0-1 2-3
1-2 1-13 4-4 5-4 6-4 7-5 8-6 9-8 10-7 12-9 12-10 13-12 14-13 15-14 15-15 16-16 17-17
```

A different implementation for cross-language span prediction using huggingface transformers is available in the software described in "[SpanAlign: Sentence Alignment Method based on Cross-Language Span Prediction and ILP](https://www.aclweb.org/anthology/2020.coling-main.418.pdf)" published at COLING-2020.

## License

This data is released under the NTT License, see `LICENSE.txt`.

