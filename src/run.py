import warnings
warnings.filterwarnings("ignore")

from config import Config
args = Config()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

import re
import time
import json
import torch
import random
import sklearn
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from utils.logger import logging, logging_params
from utils.file_utils import save_as_pickle, load_pickle
from utils.train_funcs import load_state, load_results, evaluate_, evaluate_results
from data import load_dataloaders


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_fit(args):

    logging_params() # 输出参数日志

    if args.fp16: # 混合精度计算提速
        from apex import amp
    else:
        amp = None

    cuda = torch.cuda.is_available() # cuda是否可用

    train_loader, val_loader, _, train_len, val_len, _ = load_dataloaders(args)

    logging("Loaded %d Training samples." % train_len)
    logging("Loaded %d Validating samples." % val_len)

    if 'baseline' in args.model_choice:
        from model import baseline as Model

    net = Model(args)

    tokenizer = load_pickle("%s_tokenizer.pkl" % args.bert_type, args.pkl_path)

    if cuda:
        net = nn.DataParallel(net)
        net.cuda()

    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=\
                                                [2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)

    if args.continue_train:
        start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)
    else:
        start_epoch, best_pred, amp_checkpoint = 0, 0, None

    if (args.fp16) and (amp is not None):
        logging("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=\
                                                    [2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)

    if args.continue_train:
        losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch, test_1_f1_per_epoch = load_results(args)
    else:
        losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch, test_1_f1_per_epoch = [], [], [], []

    logging("Starting training process...")
    logging("start_epoch: " + str(start_epoch))
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//30

    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train()

        total_loss = 0.0
        losses_per_batch = []
        total_acc = 0.0
        accuracy_per_batch = []

        for i, data in enumerate(train_loader):

            x, labels, _, _ = data

            # print(x, labels)

            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            loss, classification_logits = net(input_ids=x, attention_mask=attention_mask, labels=labels)

            loss = loss/args.gradient_acc_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # loss.backward() # 单卡
                loss.mean().backward() # 多卡

            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)

            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            # total_loss += loss.item() # 单卡
            total_loss += sum([i.item() for i in loss]) # 多卡

            total_acc += evaluate_(classification_logits, labels, ignore_idx=-1)[0]

            # print(classification_logits.shape, labels.shape, classification_logits, labels, '\n_________________\n\n')
            # time.sleep(1)

            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                logging('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()
        results, (out_labels, true_labels) = evaluate_results(net, val_loader, pad_id, cuda, args, 0)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        test_f1_per_epoch.append(results['f1'])
        # print(out_labels, true_labels)
        # test_1_f1_per_epoch.append(sklearn.metrics.f1_score([int(i) for i in true_labels], [int(i) for i in out_labels]))
        # test_1_f1_per_epoch.append(sklearn.metrics.f1_score(true_labels, out_labels))

        logging("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        logging("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        logging("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        logging("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
        # logging("Test \"1\" f1 at Epoch %d: %.7f" % (epoch + 1, test_1_f1_per_epoch[-1]))

        if test_f1_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    # 'state_dict': net.module.state_dict(),\
                    'state_dict': net.state_dict(),\
                    'best_f1': test_f1_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join(args.ckpt_path , "%s_test_model_best_%s_%d.pth.tar" \
                                % (args.model_choice, args.bert_type, args.use_bio)))

        # 直接覆盖了
        if (epoch % args.save_epoch) == 0:
            # save_as_pickle("%s_test_losses_per_epoch_%s.pkl" % \
            #                 (args.model_choice, args.bert_type), losses_per_epoch, args.ckpt_path)
            # save_as_pickle("%s_train_accuracy_per_epoch_%s.pkl" % \
            #                 (args.model_choice, args.bert_type), accuracy_per_epoch, args.ckpt_path)
            # save_as_pickle("%s_test_f1_per_epoch_%s.pkl" % \
            #                 (args.model_choice, args.bert_type), test_f1_per_epoch, args.ckpt_path)
            # save_as_pickle("%s_test_1_f1_per_epoch_%s.pkl" % \
            #                 (args.model_choice, args.bert_type), test_1_f1_per_epoch, args.ckpt_path)
            torch.save({
                    'epoch': epoch + 1,\
                    # 'state_dict': net.module.state_dict(),\
                    'state_dict': net.state_dict(),\
                    'best_f1': test_f1_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join(args.ckpt_path , "%s_test_checkpoint_%s_%d.pth.tar" \
                                % (args.model_choice, args.bert_type, args.use_bio)))

    logging("Finished Training!")

    with open(os.path.join(args.output_path, "perf.md"), "a") as f:
        f.write("### %s " % args.log_file[:-4])
        f.write("%s %s \n\n" % (args.model_choice, args.bert_type))
        f.write("| epoch | " + re.sub(",", " |", str(list(range(0, args.num_epochs)))[1:-1]) + " |")
        f.write("\n")
        f.write("| " + " -- |" * (args.num_epochs + 1))
        f.write("\n")
        f.write("| loss | " + re.sub(",", " |", str([int(i * 10000) for i in losses_per_epoch])[1:-1]) + " |")
        f.write("\n")
        f.write("| acc | " + re.sub(",", " |", str([int(i * 10000) for i in accuracy_per_epoch])[1:-1]) + " |")
        f.write("\n")
        f.write("| test_f1 | " + re.sub(",", " |", str([int(i * 10000) for i in test_f1_per_epoch])[1:-1]) + " |")
        # f.write("\n")
        # f.write("| test_1_f1 | " + re.sub(",", " |", str([int(i * 10000) for i in test_1_f1_per_epoch])[1:-1]) + " |")
        f.write("\n\n")

    return net

def infer_from_trained(args):

    logging("Loading tokenizer and model...")

    cuda = torch.cuda.is_available()
    train_loader, val_loader, _, train_len, val_len, _ = load_dataloaders(args)
    _, _, test_loader, _, _, test_len = load_dataloaders(args, "test")
    logging("Loaded %d Testing samples." % test_len)

    if 'baseline' in args.model_choice:
        from model import baseline as Model
    else:
        pass

    net = Model(args)

    tokenizer = load_pickle("%s_tokenizer.pkl" % args.bert_type, args.pkl_path)

    if cuda:
        net = nn.DataParallel(net)
        net.cuda()

    start_epoch, best_pred, amp_checkpoint = load_state(net, None, None, args, load_best=True)

    logging("Done!")

    logging("Starting infering process...")

    pad_id = tokenizer.pad_token_id

    f = open(args.res_path, 'w', encoding='utf16')
    # df_dev = load_pickle('df_dev_%d.pkl' % args.task, args.pkl_path)
    # txt = df_dev['sents']
    # t = df_dev['text']

    # print(txt, t, )
    df_test = load_pickle('df_test_%d.pkl' % args.task, args.pkl_path)
    txt = df_test['sents']
    t = df_test['text']

    logging("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        # for i, data in tqdm(enumerate(val_loader)):
        for i, data in enumerate(tqdm(test_loader)):
            x, labels, _, _ = data
            attention_mask = (x != pad_id).float()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()

            classification_logits = net(input_ids=x, attention_mask=attention_mask, labels=None)
            _, (o, l) = evaluate_(classification_logits[0], labels, -1, 1)
            if args.use_bio:
                out_labels = ['B-' + args.id2label[ii] if ii < len(args.id2label) else 'I-' + args.id2label[ii - len(args.id2label)] for ii in o]
                true_labels = ['B-' + args.id2label[ii] if ii < len(args.id2label) else 'I-' + args.id2label[ii - len(args.id2label)] for ii in l]
            else:
                out_labels = [args.id2label[ii] for ii in o]
                true_labels = [args.id2label[ii] for ii in l]
                count = 0
                res = ''
                # f.write(str(txt.iloc[i]) + '\n'  + t.iloc[i] + str(o) + '\n' + str(l) + '\n' + str(out_labels) + "\n" + str(true_labels) + "\n\n")
                for word in t.iloc[i].split('  '):
                    if '/' in word:
                        word = word.split('/')[0]
                    word_out_labels = out_labels[count:count+len(word)]
                    # print(word_out_labels, word)
                    if len(word_out_labels):
                        label = word_out_labels[0]
                        count += len(word)
                        res += '%s/%s  ' % (word, label)
                    else:
                        res += '%s/  ' % (word)
                f.write(res[:-3])
            
            # f.write(t.iloc[i] + "\n\n")
            # print(classification_logits[0].shape, labels.shape, '\n',str(txt.iloc[i]), '\n', o, t.iloc[i], str(out_labels), '\n', str(true_labels))
            # f.write(str(txt.iloc[i]) + '\n'  + t.iloc[i] + str(o) + '\n' + str(l) + '\n' + str(out_labels) + "\n" + str(true_labels) + "\n\n")
            # f.write(t.iloc[i] + str(o) + '\n' + str(out_labels) + "\n\n")
            # time.sleep(1)

    # print(out_labels, true_labels) 
    logging("Finished Infering!")


if __name__ == "__main__":

    set_seed(args.seed)

    if args.train:
        net = train_and_fit(args)

    if args.infer:
        infer_from_trained(args)