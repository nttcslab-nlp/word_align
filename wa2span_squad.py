#!/usr/bin/env python
# Mon Nov 11 14:38:32 2019 by Masaaki Nagata
# wa2span_squad.py [-q] wa-002.txt

# import string
# string.punctuation で区切るか？
# import unicodedata で区切り文字を探す？

# "For training, each question should have exactly 1 answer."
# なので、最初のanswerを出力したらfor文を終了

# 2019/11/29
# kfttのために do_lower オプションを追加

import sys
import argparse
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

section_sep =' \u00a7 '    # use ' § ' (section sign) for section separator
context_sep =' \u00b6 '    # use ' ¶ ' (pilcrow sign) for context separator

# トークンからスパンを予測する。
def w2s_squad(f_toks, f_tok_starts, f_orig, e_toks, e_tok_starts, e_orig,
              a_matrix, sent_id):
    # スパンの始まりと終わりを求める
    # 直前のトークンと対応点の集合が同じならば同じスパン
    e_span_start_tokidx = [0]
    e_span_end_tokidx = []
    for j in range(1, len(e_toks)):
        if (a_matrix[:,j-1] != a_matrix[:,j]).any() : # 対応点が違うなら
            e_span_start_tokidx.append(j)             # 新しいスパンの始まり
            e_span_end_tokidx.append(j)
    e_span_end_tokidx.append(len(e_toks))

    e_tok_ends = []
    for i, tok in enumerate(e_toks):
        e_tok_ends.append(e_tok_starts[i] + len(tok))

    f_tok_ends = []
    for i, tok in enumerate(f_toks):
        f_tok_ends.append(f_tok_starts[i] + len(tok))

    #print(e_span_start_tokidx, e_span_end_tokidx)

    # 単語からスパンへの対応を求める。
    w2s_a_matrix = np.zeros((len(f_toks),
                              len(e_span_start_tokidx)),
                             dtype=int)
    for i, tok in enumerate(f_toks):
        for j, span_start_tokidx in enumerate(e_span_start_tokidx):
            w2s_a_matrix[i,j] = a_matrix[i, span_start_tokidx]
    #print(w2s_a_matrix)

    # SQuAD形式の単語対応
    f2e_json = {}
    if args.q_context:
        f2e_json["q_context"] = f_orig
    f2e_json["context"] = e_orig
    f2e_json["qas"] = []

    # f2e_json = {"context": e_orig,
    #            "qas": []}

    if args.verbose:
        print(f_toks, e_toks)

    # 日本語文の各スパンについて対応する英語文のスパンを表示する。
    for i, f_tok in enumerate(f_toks):
        #f_tok_text = f_tok
        f_tok_text = f_orig[f_tok_starts[i]:f_tok_ends[i]]

        # 文脈はトークン単位
        if args.context > 0:
            if i == 0:
                left_context = ''
            else:
                left_context_start = f_tok_starts[max(0, i - args.context)]
                left_context_end = f_tok_ends[i-1]
                left_context = f_orig[left_context_start:left_context_end]

            if i == len(f_toks) - 1:
                right_context = ''
            else:
                right_context_start = f_tok_starts[i+1]
                right_context_end = f_tok_ends[min(i + args.context,
                                                   len(f_toks) - 1)]
                right_context = f_orig[right_context_start:right_context_end]
            f_tok_text = left_context + context_sep + \
                         f_orig[f_tok_starts[i]:f_tok_ends[i]] + \
                         context_sep + right_context

        if args.whole:
            if i == 0:
                left_context = ''
            else:
                left_context_start = f_tok_starts[0]
                left_context_end = f_tok_ends[i-1]
                left_context = f_orig[left_context_start:left_context_end]
            if i == len(f_toks) - 1:
                right_context = ''
            else:
                right_context_start = f_tok_starts[i+1]
                right_context_end = f_tok_ends[len(f_toks) -1]
                right_context = f_orig[right_context_start:right_context_end]
            f_tok_text = left_context + context_sep + \
                         f_orig[f_tok_starts[i]:f_tok_ends[i]] + \
                         context_sep + right_context

        # if args.whole:
        #     f_tok_text = f_tok_text + section_sep + f_orig

        #print(f_tok_text)
        #continue
        
        if (w2s_a_matrix[i,:] == 0).all() :
            #f_id = '{}_{}_X'.format(sent_id, f_tok_starts[i])
            f_id = '{}_{}_X'.format(sent_id, i)
            #f_id = '{}_{}_{}'.format(sent_id, i, -1)
            qa = {"id": f_id,
                  "question": f_tok_text,
                  "answers": [],
                  "is_impossible": True}
            if args.q_context:
                qa["q_start"] = f_tok_starts[i]
            f2e_json["qas"].append(qa)
            if args.verbose:
                print(f_id, f_tok_text, f_tok_starts[i], f_tok_ends[i],
                      '<NULL>')
        else:
            for j, span_start_tokidx in enumerate(e_span_start_tokidx):
                if w2s_a_matrix[i,j] == 1:
                    span_end_tokidx = e_span_end_tokidx[j] # スパン終了単語番号
                    e_span_start = e_tok_starts[span_start_tokidx]
                    e_span_end = e_tok_ends[span_end_tokidx-1]
                    e_span_text = e_orig[e_span_start:e_span_end]
                    answer = {"text": e_span_text,
                              "answer_start": e_span_start}
                    # f_id = '{}_{}_{}'.format(sent_id,
                    #                          f_tok_starts[i], e_span_start)
                    f_id = '{}_{}_{}'.format(sent_id, i, span_start_tokidx)
                    qa = {"id": f_id,
                          "question": f_tok_text,
                          "answers": [answer],
                          "is_impossible": False}
                    #qa["answers"].append(answer)
                    if args.q_context:
                        qa["q_start"] = f_tok_starts[i]
                    f2e_json["qas"].append(qa)

                    if args.verbose:
                        print(f_id, f_tok_text, f_tok_starts[i], f_tok_ends[i],
                              e_span_text, e_span_start, e_span_end)


    return f2e_json

def find_start_pos(toks, sent):
    offset = 0
    pos_list = []
    for tok in toks:
        #idx = sent.lower().find(tok, offset) # tokenが小文字なのでsentを小文字に
        idx = sent.find(tok, offset) # tokenが小文字なのでsentを小文字に
        #print (sent, offset, tok, idx)
        if idx >= 0:
            pos_list.append(idx)
            offset = idx + len(tok)
        else:
            print("token mismatch")
            print(sent, toks, offset)
            sys.exit()

    return pos_list

def main(args):
    wa_file_Path = Path(args.wa_file)

    w2s = { "version": "v2.0",
            "data": [{"paragraphs": []}]}
    n_sent = 0
    with wa_file_Path.open() as f:
        for line in f:
            (f_line, e_line, a_line, f_orig, e_orig) = line.strip().split('\t')
            f_toks = f_line.strip().split(' ')
            e_toks = e_line.strip().split(' ')
            if args.do_lower:
                f_tok_starts = find_start_pos(f_toks, f_orig.lower())
                e_tok_starts = find_start_pos(e_toks, e_orig.lower())
            else:
                f_tok_starts = find_start_pos(f_toks, f_orig)
                e_tok_starts = find_start_pos(e_toks, e_orig)

            a_toks = a_line.strip().split(' ')
            a_matrix = np.zeros((len(f_toks), len(e_toks)), dtype=int)
            for a_tok in a_toks:
                (i, j) = a_tok.split('-')
                a_matrix[int(i), int(j)] = 1

            sent_id = '{}_{}'.format(wa_file_Path.stem, n_sent)
            # print(sent_id)
            # print(f_toks, e_toks)
            # print(a_matrix)

            f2e_json = w2s_squad(f_toks, f_tok_starts, f_orig,
                                 e_toks, e_tok_starts, e_orig,
                                 a_matrix, sent_id + '_f')
            w2s["data"][0]["paragraphs"].append(f2e_json)

            e2f_json = w2s_squad(e_toks, e_tok_starts, e_orig,
                                 f_toks, f_tok_starts, f_orig,
                                 a_matrix.transpose(), sent_id + '_e')
            w2s["data"][0]["paragraphs"].append(e2f_json)

            n_sent += 1
        # end
    # end

    print(json.dumps(w2s, ensure_ascii=False, indent=2)) # 標準出力へ


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Word alignments to SQuAD 2.0 format')
    parser.add_argument('wa_file', metavar='wa_file',
                        help='word alignment data file')
    parser.add_argument('--direction', default='both',
                        choices=['both', 'f2e', 'e2f'])
    parser.add_argument('--context', '-c', default=0, type=int)
    parser.add_argument('--whole', '-w', action='store_true')
    parser.add_argument('--q_context', '-q', action='store_true')
    parser.add_argument('--do_lower', '-l', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    status = main(args)

