#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from config import Config
args = Config()

from utils.file_utils import save_as_pickle, load_pickle
from utils.logger import logging

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = args.ckpt_path
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"%s_test_checkpoint_%s_%d.pth.tar" % \
                                    (args.model_choice, args.bert_type, args.use_bio))
    best_path = os.path.join(base_path,"%s_test_model_best_%s_%d.pth.tar" % \
                            (args.model_choice, args.bert_type, args.use_bio))
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logging("Loaded best model %s." % best_path)
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logging("Loaded checkpoint model %s." % checkpoint_path)
    if checkpoint != None:
    # if checkpoint != None and 0:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_f1']
        net.load_state_dict(checkpoint['state_dict'])
        # net = net.module
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logging("Loaded model and optimizer.")
    else:
        logging("Ckpt not found!")
    return start_epoch, best_pred, amp_checkpoint

def load_results(args):
    """ Loads saved results if exists """
    base_path = args.ckpt_path
    losses_path = os.path.join(base_path,"%s_test_losses_per_epoch_%s.pkl" % \
                                (args.model_choice, args.bert_type))
    accuracy_path = os.path.join(base_path,"%s_test_accuracy_per_epoch_%s.pkl" % \
                                (args.model_choice, args.bert_type))
    f1_path = os.path.join(base_path,"%s_test_f1_per_epoch_%s.pkl" % \
                                (args.model_choice, args.bert_type))
    t1_f1_path = os.path.join(base_path,"%s_test_1_f1_per_epoch_%s.pkl" % \
                                (args.model_choice, args.bert_type))
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("%s_test_losses_per_epoch_%s.pkl" % 
                                        (args.model_choice, args.bert_type), base_path)
        accuracy_per_epoch = load_pickle("%s_test_accuracy_per_epoch_%s.pkl" % \
                                        (args.model_choice, args.bert_type), base_path)
        f1_per_epoch = load_pickle("%s_test_f1_per_epoch_%s.pkl" % \
                                        (args.model_choice, args.bert_type), base_path)
        test_1_f1_per_epoch = load_pickle("%s_test_1_f1_per_epoch_%s.pkl" % \
                                        (args.model_choice, args.bert_type), base_path)
        logging("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch, test_1_f1_per_epoch = [], [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch, test_1_f1_per_epoch

def evaluate_(output, labels, ignore_idx=-1, test=0):

    # print(output.shape, labels.shape)

    if not test and labels.shape[0] > 1:

        ### ignore index 0 (padding) when calculating accuracy
        idxs = (labels != ignore_idx).squeeze()
        o_labels = torch.softmax(output, dim=-1).max(-1)[1]
        l = labels.squeeze()[idxs]
        o = o_labels[idxs]
        # print(output.shape, o.shape,l)

        if len(idxs) > 1:
            acc = (l == o).sum().item()/len(idxs)
        else:
            acc = (l == o).sum().item()
        l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
        o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()
        # print(o,l, '\n ___________________ \n ')
        # time.sleep(1)

        return acc, (o, l)
    else: # for bcz=1

        ### ignore index 0 (padding) when calculating accuracy
        idxs = (labels != ignore_idx).squeeze()
        o_labels = torch.softmax(output, dim=-1).max(-1)[1]
        l = labels.squeeze()[idxs]
        o = o_labels.squeeze()[idxs]
        # print(output.shape, o.shape, l.shape)

        if len(idxs) > 1:
            acc = (l == o).sum().item()/len(idxs)
        else:
            acc = (l == o).sum().item()
        l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
        o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()
        # print(o,l, '\n ___________________ \n ')

        return acc, (o, l)


def evaluate_results(net, test_loader, pad_id, cuda, args, test):
    logging("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, labels, _, _ = data
            attention_mask = (x != pad_id).float()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()

            _, classification_logits = net(input_ids=x, attention_mask=attention_mask, labels=labels)
            accuracy, (o, l) = evaluate_(classification_logits, labels, -1, test)
            out_labels += [str(i) for i in o]; true_labels += [str(i) for i in l]
            # print(out_labels, true_labels)
            acc += accuracy

    accuracy = acc/(i + 1)

    results = {
        "accuracy": accuracy,
        "precision": precision_score(true_labels, out_labels, average='micro'),
        "recall": recall_score(true_labels, out_labels, average='micro'),
        "f1": f1_score(true_labels, out_labels, average='micro')
    }

    logging("***** Eval results *****")
    for key in sorted(results.keys()):
        logging("  %s = %s" % (key, str(results[key])))
    return results, (out_labels, true_labels)