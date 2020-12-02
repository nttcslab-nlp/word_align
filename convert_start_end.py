#!/usr/bin/env python
# Thu Nov 21 15:23:32 2019 by Masaaki Nagata
# modified Fri Nov 27 16:55:29 2020 by Masaaki Nagata
# convert_start_end.py -q wwa-002.json -n nbest_predictions.json \
#       		       -m 160 > nbest_predictions_charindex.json

# BERT tokenizer が [UNK] を返すことがある。
# >>> tokenizer.tokenize('岩崎泰頴')
# ['岩', '崎', '泰', '[UNK]']

import sys
import argparse
import json
import re
from collections import defaultdict

# BERT tokenizer
sys.path.append('./bert')
import tokenization

vocab_file = './multi_cased_L-12_H-768_A-12/vocab.txt'
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file,
    do_lower_case=False)

# 一つずつ文対を処理する。
id2context_tokens = defaultdict(list)
id2context_token_to_char_index = defaultdict(list)
id2question_tokens = defaultdict(list)
id2question_token_to_char_index = defaultdict(list)

def get_bert_tokens(text):
    bert_tokens = []
    for token in text.strip().split(' '):
        bert_tokens.extend(tokenizer.tokenize(token))
    return bert_tokens

def make_bert_tokens_to_char_index(text, bert_tokens):
    bert_tokens_to_char_index = []
    offset = 0
    for token in bert_tokens:
        token = re.sub(r'^##', '', token)
        index = text.find(token, offset)
        if index >= 0:
            bert_tokens_to_char_index.append(index)
            offset = index + len(token)
        elif token == '[UNK]':
            m = re.search(r'.', text[offset:]) # とりあえず任意の一文字だけ
            bert_tokens_to_char_index.append(offset + m.start())
            offset = offset + m.end()
        else:
            print("token mismatch")
            print(offset, bert_tokens, text)
            sys.exit()

    #assert offset == len(text), 'offset:{}, text:{}'.format(offset,text)
    return bert_tokens_to_char_index

def main(args):
    # 質問を読む
    with open(args.questions) as qas:
        qas_json = json.load(qas)
        for data in qas_json['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                context_tokens = get_bert_tokens(context)
                if args.verbose: print(context_tokens)

                for qa in paragraph['qas']:
                    id = qa['id']
                    question = qa['question']
                    question_tokens = get_bert_tokens(question)
                    if args.verbose: print(id, question)
                    id2context_tokens[id] = context_tokens
                    id2context_token_to_char_index[id] \
                        = make_bert_tokens_to_char_index(context,
                                                         context_tokens)
                    id2question_tokens[id] = question_tokens
                    id2question_token_to_char_index[id] \
                        = make_bert_tokens_to_char_index(question,
                                                         question_tokens)
    # 回答を読む
    #print("loading nbest_predictions")
    with open(args.nbest_predictions) as nbp:
        nbest_predictions_json = json.load(nbp)

    for id, predictions in nbest_predictions_json.items():
        if args.verbose:
            print(id)
        for prediction in predictions:
            text = prediction['text']
            start = prediction['start']
            end = prediction['end']
            context_tokens = id2context_tokens[id]
            question_tokens = id2question_tokens[id]
            if len(question_tokens) > args.max_query_length: # queryの最大長
                offset = args.max_query_length + 2
            else:
                offset = len(question_tokens) + 2
            q_tok_text = ' '.join(question_tokens)
            c_tok_text = ' '.join(context_tokens)

            #print(text, start, end)
            if start == 0 or start == -1:
                prediction['start_char'] = -1
                prediction['end_char'] = -1
                if args.verbose:
                    print(q_tok_text)
                    print(start, end)
            else:
                a_start = start - offset
                a_end = end - offset
                a_tok_text = ' '.join(context_tokens[a_start:a_end+1])
                a_text = a_tok_text.replace(' ##', '')
                a_text = a_text.replace('##', '')
                context_token_to_char_index = id2context_token_to_char_index[id]
                a_start_char_index = context_token_to_char_index[a_start]
                a_end_token_char_index = context_token_to_char_index[a_end]
                a_end_token = context_tokens[a_end]
                a_end_char_index = a_end_token_char_index \
                                   + len(a_end_token.replace('##', ''))
                prediction['start_char'] = a_start_char_index
                prediction['end_char'] = a_end_char_index
                if args.verbose:
                    print(q_tok_text, offset)
                    print(a_text, ',', a_tok_text)
                    print(c_tok_text, a_start_char_index, a_end_char_index)
    
    print(json.dumps(nbest_predictions_json,
                     ensure_ascii=False, indent=2)) # 標準出力へ

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbest_predictions', '-n',
                        help='nbest predictions json file')
    parser.add_argument('--questions', '-q',
                        metavar='questions',
                        help='context, question and answer json file')
    parser.add_argument('--max_query_length', '-m',
                        type=int, default=64)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    main(args)
# end
