#!/usr/bin/env python
# Fri Nov 22 17:02:29 2019 by Masaaki Nagata
# get_alignment.py -a deen_test.text -n charindex_nbest_predictions.json -m 160

# 2019/11/25
# ファイル名に'\'がある場合に対応して rsplit('_',r)にした。
# kftt alignment の token は小文字化されていたが、LDCは違うので
# find_start_charindex を修正した。
# 2019/11/27
# s2tとt2sの単語対応をmoses形式で出力するようにした。
# 2019/12/2
# kfttのために do_lower オプションを追加

import sys
import json
import numpy as np
import argparse
from pathlib import Path

from collections import defaultdict
from collections import namedtuple

def find_start_charindex(toks, sent):
    offset = 0
    tok_to_charindex = []
    for tok in toks:
        #idx = sent.lower().find(tok, offset) # tokenが小文字なのでsentを小文字に
        idx = sent.find(tok, offset) # tokenが小文字なのでsentを小文字に
        #print (sent, offset, tok, idx)
        if idx >= 0:
            tok_to_charindex.append(idx)
            offset = idx + len(tok)
        else:
            print("token mismatch")
            print(sent, offset, tok)
            sys.exit()

    return tok_to_charindex

def count_common(a_matrix, alignment):
    n_ref = np.sum(a_matrix)
    n_sys = np.sum(alignment)
    n_common = np.sum(np.multiply(a_matrix, alignment))
    if args.verbose:
        # recall = n_common / n_ref
        # precision = n_common / n_sys
        # f1 = 2 * precision * recall / (precision + recall)
        # print('r={:.3f} p={:.3f} f1={:.3f}'.format(recall, precision, f1),
        #       end=' ')
        print('({},{},{})'.format(n_ref, n_sys, n_common))

    return n_ref, n_sys, n_common

def span_to_alignment(s_toks, t_toks, s_tok_to_charindex, t_tok_to_charindex,
                      s_uniq_tok_id_prefix, uniq_tok_id_to_keys,
                      nbest_predictions_json):
    align_matrix = np.zeros((len(s_toks), len(t_toks)), dtype=int)
    prob_matrix = np.zeros((len(s_toks), len(t_toks)), dtype=float)
    for i, s_tok in enumerate(s_toks):
        s_tok_start_char = s_tok_to_charindex[i]
        s_tok_end_char = s_tok_start_char + len(s_tok)
        if args.verbose:
            print(i, s_tok, s_tok_start_char, s_tok_end_char)

        s_uniq_tok_id = '{}_{}'.format(s_uniq_tok_id_prefix, i)
        s_key = uniq_tok_id_to_keys[s_uniq_tok_id][0]
        s_nbest_predictions = nbest_predictions_json[s_key]
        s_best_prediction = sorted(s_nbest_predictions,
                                   key=lambda x: -x['probability'])[0]
        #print(s_best_prediction)
        for j, t_tok in enumerate(t_toks):
            t_tok_start_char = t_tok_to_charindex[j]
            t_tok_end_char = t_tok_start_char + len(t_tok)
            if (s_best_prediction['start_char'] <= t_tok_start_char and
                t_tok_end_char <= s_best_prediction['end_char']):
                prob = s_best_prediction['probability']
                if args.verbose: print(j, t_tok, prob)
                align_matrix[i,j] = 1
                prob_matrix[i,j] = prob

    # print(align_matrix)
    return align_matrix, prob_matrix

def print_alignment(s_toks, t_toks, align_matrix):
    align_list = []
    for i, s_tok in enumerate(s_toks):
        for j, t_tok in enumerate(t_toks):
            if align_matrix[i,j] == 1:
                align_list.append('{}-{}'.format(i,j))
    print(' '.join(align_list))

def main(args):
    # 回答を読む
    with open(args.nbest_predictions) as fp:
        nbest_predictions_json = json.load(fp)

    # 各文の各トークンからnbestスパン候補を検索するためのキーを記録する。
    uniq_tok_id_to_keys = defaultdict(list)
    for key in nbest_predictions_json.keys():
        #(file_id, sent_id, s_lang, s_id, t_id) = key.split('_')
        (file_id, sent_id, s_lang, s_id, t_id) = key.rsplit('_',4)
        uniq_tok_id = '{}_{}_{}_{}'.format(file_id, sent_id, s_lang, s_id)
        uniq_tok_id_to_keys[uniq_tok_id].append(key)

    total_ref = 0
    total_sys = defaultdict(int)
    total_common = defaultdict(int)

    # 単語対応データを一文ずつ読む。
    alignment_file_Path = Path(args.alignments)
    file_id = alignment_file_Path.stem
    with alignment_file_Path.open() as f:
        n_pairs = 0
        for line in f:
            (f_line, e_line, a_line, f_orig, e_orig) = line.strip().split('\t')
            f_toks = f_line.strip().split(' ')
            e_toks = e_line.strip().split(' ')
            if args.do_lower:
                f_tok_to_charindex = find_start_charindex(f_toks, f_orig.lower())
                e_tok_to_charindex = find_start_charindex(e_toks, e_orig.lower())
            else:
                f_tok_to_charindex = find_start_charindex(f_toks, f_orig)
                e_tok_to_charindex = find_start_charindex(e_toks, e_orig)
            a_toks = a_line.strip().split(' ')
            a_matrix = np.zeros((len(f_toks), len(e_toks)), dtype=int)
            for a_tok in a_toks:
                (i, j) = a_tok.split('-')
                a_matrix[int(i), int(j)] = 1
            #print(a_matrix)
            
            if args.verbose: print(f_orig, e_orig)

            n_ref = np.sum(a_matrix)
            total_ref += n_ref

            f_uniq_tok_id_prefix = '{}_{}_f'.format(file_id, n_pairs)
            (f_to_e_align, f_to_e_prob) \
                = span_to_alignment(f_toks, e_toks,
                                    f_tok_to_charindex, e_tok_to_charindex,
                                    f_uniq_tok_id_prefix,
                                    uniq_tok_id_to_keys,
                                    nbest_predictions_json)

            e_uniq_tok_id_prefix = '{}_{}_e'.format(file_id, n_pairs)
            (e_to_f_align, e_to_f_prob) \
                = span_to_alignment(e_toks, f_toks,
                                    e_tok_to_charindex, f_tok_to_charindex,
                                    e_uniq_tok_id_prefix,
                                    uniq_tok_id_to_keys,
                                    nbest_predictions_json)

            bidi_align_ave = (f_to_e_prob + e_to_f_prob.transpose()) / 2.0
            bidi_align_th = np.where(bidi_align_ave > 0.4, 1, 0)

            if args.source_to_target:
                print_alignment(f_toks, e_toks, f_to_e_align)
            elif args.target_to_source:
                print_alignment(f_toks, e_toks, e_to_f_align.transpose())
            elif args.bidi_threshold:
                print_alignment(f_toks, e_toks, bidi_align_th)
            else:
                (n_ref, n_sys, n_common) = count_common(a_matrix, f_to_e_align)
                total_sys['f_to_e'] += n_sys
                total_common['f_to_e'] += n_common

                (n_ref, n_sys, n_common) \
                    = count_common(a_matrix, e_to_f_align.transpose())
                total_sys['e_to_f'] += n_sys
                total_common['e_to_f'] += n_common

                bidi_align_int = np.multiply(f_to_e_align,
                                             e_to_f_align.transpose())
                (n_ref, n_sys, n_common) = count_common(a_matrix,
                                                        bidi_align_int)
                total_sys['bidi_int'] += n_sys
                total_common['bidi_int'] += n_common

                bidi_align_uni = np.maximum(f_to_e_align,
                                            e_to_f_align.transpose())
                (n_ref, n_sys, n_common) = count_common(a_matrix,
                                                        bidi_align_uni)
                total_sys['bidi_uni'] += n_sys
                total_common['bidi_uni'] += n_common

                (n_ref, n_sys, n_common) = count_common(a_matrix, bidi_align_th)
                total_sys['bidi_th'] += n_sys
                total_common['bidi_th'] += n_common

            n_pairs += 1


    for key in total_sys.keys():
        recall = total_common[key]/total_ref
        precision = total_common[key]/total_sys[key]
        f1 = 2 * recall * precision / (recall + precision)
        print('{}: {:.3f} {:.3f} {:.3f}'.format(key, recall, precision, f1),
              end=' ')
        print('({} {} {})'.format(total_ref, total_sys[key], total_common[key]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbest_predictions', '-n',
                         help='nbest predictions json file (char index)')
    parser.add_argument('--alignments', '-a')
    parser.add_argument('--source_to_target', '-e', action='store_true')
    parser.add_argument('--target_to_source', '-f', action='store_true')
    parser.add_argument('--bidi_threshold', '-b', action='store_true')
    parser.add_argument('--max_query_length', '-m',
                        type=int, default=160)
    parser.add_argument('--do_lower', '-l', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    main(args)
# end
