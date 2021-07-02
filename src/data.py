#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy
import time
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Config
args = Config()

from utils.file_utils import save_as_pickle, load_pickle
from utils.logger import logging

def process_text(text, mode='train'):
    '''
    将数据集读成两个列表
    '''
    sents, pos, ts = [], [], []
    pos_labels = [] # 第一次运行时打印 labels，复制粘贴写在 args.label2id 里
    for i in text:
        if i == '\n':
            continue
        line = i.split('  ')[:-1]
        if mode == 'train':
            a, b = [j.split('/')[0] for j in line], [j.split('/')[1] for j in line]
            tokens, poss = [], []
            pos_labels += b
            for w in range(len(a)):
                word = a[w]
                tokens.append(word[0])
                poss.append('B-' + b[w])
                for t in word[1:]:
                    tokens.append(t)
                    poss.append('I-' + b[w])
        else:
            a = line; tokens = []; poss = []
            for w in range(len(a)):
                word = a[w]
                tokens.append(word[0])
                for t in word[1:]:
                    tokens.append(t)
            poss = [0] * len(tokens)
        ts.append(i)
        sents.append(tokens)
        pos.append(poss)
    pos_labels = list(set(pos_labels))
    # print(pos_labels)
    return sents, pos, ts

def preprocess_data(args):

    data_path = args.train_data
    logging("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf16') as f:
        text = f.readlines()
    sents, pos, text = process_text(text, 'train')
    select = list(range(len(sents)))
    random.shuffle(select)
    df_train = pd.DataFrame(data={'sents': [sents[i] for i in select[:int(0.7 * len(select))]], \
                                 'pos': [pos[i] for i in select[:int(0.7 * len(select))]], \
                                 'text': [text[i] for i in select[:int(0.7 * len(select))]]})
    df_dev = pd.DataFrame(data={'sents': [sents[i] for i in select[int(0.7 * len(select)):]], \
                                 'pos': [pos[i] for i in select[int(0.7 * len(select)):]], \
                                 'text': [text[i] for i in select[int(0.7 * len(select)):]]})

    data_path = args.test_data
    logging("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf16') as f:
        text = f.readlines()
    sents, pos, text = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'pos': pos, 'text':text})
    if not os.path.exists(args.pkl_path):
        os.mkdir(args.pkl_path)

    save_as_pickle('df_train_%d.pkl' % args.task, df_train, args.pkl_path)
    save_as_pickle('df_dev_%d.pkl' % args.task, df_dev, args.pkl_path)
    save_as_pickle('df_test_%d.pkl' % args.task, df_test, args.pkl_path)

    logging("Finished and saved!")

    return df_train, df_dev, df_test

class get_dataset(Dataset):

    def __init__(self, df, tokenizer, nolabel=False):
        self.df = df
        logging("Tokenizing data...")
        tqdm.pandas(desc='Tokenizing input')
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),axis=1)
        tqdm.pandas(desc='Tokenizing labels')
        if nolabel:
            self.df['labels'] = self.df.progress_apply(lambda x: self.add_ign(x['pos']),axis=1)
        else:
            self.df['labels'] = self.df.progress_apply(lambda x: self.convert2id(x['pos']),axis=1)
        self.df.dropna(axis=0, inplace=True)

        # print(self.df['input'], self.df['labels'], self.df['sents'], self.df['text'], self.df['pos'])
        # for i, r in self.df.iterrows():
        #     assert len(r['input']) == len(r['labels']), '%s, %s have different lengths !' % (r['input'], r['labels'])

    def add_ign(self, pos):
        return [-1] + pos + [-1]

    def convert2id(self, pos):
        if args.use_bio:
            return [-1] + [args.label2id[p.split('-')[1]] + (len(args.id2label) if p.split('-')[0]=='I' else 0) for p in pos] + [-1]
        else:
            return [-1] + [args.label2id[p.split('-')[1]] for p in pos] + [-1]

    def __len__(self,):
        return len(self.df)

    def __getitem__(self, idx):
        # print(torch.LongTensor(self.df.iloc[idx]['labels']).shape, torch.LongTensor(self.df.iloc[idx]['labels']))
        return torch.LongTensor(self.df.iloc[idx]['input']), \
                torch.LongTensor(self.df.iloc[idx]['labels'])

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        seqs = [x[0][:512] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = [x[1][:512] for x in sorted_batch]
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        # labels = list(map(lambda x: x[1], sorted_batch))
        # labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        # y_lengths = torch.LongTensor([len(x) for x in labels])

        # print(x_lengths, y_lengths)

        return seqs_padded, labels_padded, x_lengths, y_lengths


def load_dataloaders(args, mode='train'):

    if os.path.isfile(os.path.join(args.pkl_path, '%s_tokenizer.pkl' % args.bert_type)):
        tokenizer = load_pickle("%s_tokenizer.pkl" % args.bert_type, args.pkl_path)
        logging("Loaded tokenizer")
    else:
        if not os.path.exists(args.pkl_path):
            os.mkdir(args.pkl_path)
        if 'roberta' in args.bert_type:
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(args.bert_path)
        else:
            from utils.BERT.tokenization_bert import BertTokenizer as Tokenizer
            tokenizer = Tokenizer.from_pretrained(args.bert_path)
        save_as_pickle("%s_tokenizer.pkl" % args.bert_type, tokenizer, args.pkl_path)
        logging("Saved %s tokenizer at %s/%s_tokenizer.pkl" %(args.bert_type, args.pkl_path, args.bert_type))

    train_path = os.path.join(args.pkl_path, 'df_train_%d.pkl' % args.task)
    dev_path = os.path.join(args.pkl_path, 'df_dev_%d.pkl' % args.task)
    test_path = os.path.join(args.pkl_path, 'df_test_%d.pkl' % args.task)

    if os.path.isfile(train_path) and os.path.isfile(dev_path) and os.path.isfile(test_path):
        df_train = load_pickle('df_train_%d.pkl' % args.task, args.pkl_path)
        df_dev = load_pickle('df_dev_%d.pkl' % args.task, args.pkl_path)
        df_test = load_pickle('df_test_%d.pkl' % args.task, args.pkl_path)
        logging("Loaded preproccessed data.")
    else:
        df_train, df_dev, df_test = preprocess_data(args)

    # print(df_dev['sents'], df_dev['pos'], df_dev['text'])

    if mode == 'train':
        train_set = get_dataset(df_train, tokenizer=tokenizer)
        dev_set = get_dataset(df_dev, tokenizer=tokenizer)
        train_length = len(train_set); dev_length = len(dev_set); test_length = 0

        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id, label_pad_value=tokenizer.pad_token_id)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                    num_workers=0, collate_fn=PS, pin_memory=False)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, \
                                    num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = 0
    else:
        test_set = get_dataset(df_test, tokenizer=tokenizer, nolabel = True)

        train_length = 0; dev_length = 0; test_length = len(test_set)

        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id, label_pad_value=tokenizer.pad_token_id)
        train_loader = 0
        dev_loader = 0
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, \
                                    num_workers=0, collate_fn=PS, pin_memory=False)

    return train_loader, dev_loader, test_loader, train_length, dev_length, test_length
