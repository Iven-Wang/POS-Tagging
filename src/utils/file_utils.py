import os
import re
import math
import json
import torch
import pickle
import torch.nn as nn
from itertools import permutations
from sklearn.metrics import precision_score, recall_score, f1_score

# 保存中间过程

def printjson(dic):
    js = json.dumps(dic, sort_keys=True, indent=1, separators=(',', ':'))
    print(js)

def load_pickle(filename, pth="./data/"):
    completeName = os.path.join(pth, filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data, pth="./data/"):
    completeName = os.path.join(pth, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def get_subject_objects(sent_):
    ### get subject, object entities by dependency tree parsing
    #sent_ = next(sents_doc.sents)
    root = sent_.root
    subject = None; objs = []; pairs = []
    for child in root.children:
        #print(child.dep_)
        if child.dep_ in ["nsubj", "nsubjpass"]:
            if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                subject = child; #print('Subject: ', child)
        elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
            objs.append(child); #print('Object ', child)
    if (subject is not None) and (len(objs) > 0):
        for a, b in permutations([subject] + [obj for obj in objs], 2):
            a_ = [w for w in a.subtree]
            b_ = [w for w in b.subtree]
            pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
            
    return pairs
