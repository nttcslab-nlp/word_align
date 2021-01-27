#!/usr/bin/env python
# Fri Nov 29 15:57:05 2019 by Masaaki Nagata
# LDCの中英単語対応データからmoses形式に変換する
# convert_to_giza.py < XXX.ldc > XXX.moses

# waファイルがrejectならば捨てる。
# token mismatch が speaker tag ^[),  time stamp ^<, empty category ^* 以外なら
# その文は捨てる。

# 2019/11/29
# cmn-NG-31-111576-3460874-S1.eng.{raw,tkn} のような例に対応する。
# raw:       If [we//you//one] [take//takes] a comprehensive look at
# tokenized: If we take a comprehensive look at

import sys
import argparse
import numpy as np
import re

def find_start_charindex(toks, sent):
    offset = 0
    tok_to_charindex = []
    for tok in toks:
        idx = sent.find(tok, offset) # tokenが小文字なのでsentを小文字に
        #print (sent, offset, tok, idx)
        if idx >= 0:
            tok_to_charindex.append(idx)
            offset = idx + len(tok)
        else:
            tok_to_charindex.append(-1)
            #print("token mismatch:", tok, offset, sent, file=sys.stederr)

    return tok_to_charindex

def main(args):
    n_lines = 0
    for line in sys.stdin:
        lines = line.rstrip('\n').split('\t') # 改行を取り除き、タブで区切る
        n_lines += 1

        if lines[2] == 'rejected': # 単語対応データがない
            continue

        (f_line, e_line, a_line, f_orig, e_orig) = lines
        f_toks = f_line.strip().split(' ')
        e_toks = e_line.strip().split(' ')

        f_tok_to_charindex = find_start_charindex(f_toks, f_orig)
        e_tok_to_charindex = find_start_charindex(e_toks, e_orig)
        if args.verbose:
            if any([i == -1 for i in f_tok_to_charindex]):
                print(' '.join(f_toks), f_orig,
                      ' '.join(map(str, f_tok_to_charindex)),
                      file=sys.stderr)
            if any([i == -1 for i in e_tok_to_charindex]):
                print(' '.join(e_toks), e_orig,
                      ' '.join(map(str, e_tok_to_charindex)),
                      file=sys.stderr)

        a_toks = a_line.strip().split(' ')
        new_a_toks = []
        for token in a_toks:
            token = re.sub(r'\[\w+\]', '', token) # [DET]などを削除
            token = re.sub(r'\(\w+\)', '', token) # (GIS)などを削除
            new_a_toks.append(token)

        a_matrix = np.zeros((len(f_toks), len(e_toks)), dtype=int)
        f_indexes = set()
        e_indexes = set()
        for a_tok in new_a_toks:
            (f_seq, e_seq) = a_tok.split('-')
            for f in f_seq.split(','):
                if f:
                    f_indexes.add(int(f))
                for e in e_seq.split(','):
                    if e:
                        e_indexes.add(int(e))
                    if f and e:
                        a_matrix[int(f)-1, int(e)-1] = 1

        new_f_toks = []
        new_f_tok_to_charindex = []
        for j, f_tok in enumerate(f_toks):
            if f_tok_to_charindex[j] != -1:
                new_f_toks.append(f_tok)
                new_f_tok_to_charindex.append(f_tok_to_charindex[j])

        new_e_toks = []
        new_e_tok_to_charindex = []
        for i, e_tok in enumerate(e_toks):
            if e_tok_to_charindex[i] != -1:
                new_e_toks.append(e_tok)
                new_e_tok_to_charindex.append(e_tok_to_charindex[i])

        f_bool = (np.array(f_tok_to_charindex) != -1)
        e_bool = (np.array(e_tok_to_charindex) != -1)
        new_a_matrix = a_matrix[np.ix_(f_bool,e_bool)]
        #print(new_a_matrix)

        new_a_toks = []
        for i, f_tok in enumerate(new_f_toks):
            for j, e_tok in enumerate(new_e_toks):
                if new_a_matrix[i,j] == 1:
                    new_a_toks.append('{}-{}'.format(i,j))
                    #print(i, new_f_toks[i], j, new_e_toks[j])

        # 出力すべきデータがあれば出力する。
        if new_f_toks and new_e_toks and new_a_toks:
            new_f_line = ' '.join(new_f_toks)
            new_e_line = ' '.join(new_e_toks)
            new_a_line = ' '.join(new_a_toks)
            print("\t".join([new_f_line, new_e_line, new_a_line,
                             f_orig, e_orig]))

        # new_a_matrix = np.zeros((len(new_f_toks), len(new_e_toks)), dtype=int)
        # new_a_matrix = a_matrix[f_bool:e_bool]
        # print(new_a_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    main(args)

 
