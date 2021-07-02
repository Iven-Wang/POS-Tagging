import os, sys, time, datetime, random

import transformers
transformers.logging.set_verbosity(50)

class Config(object):

    bert_type = 'bert'
    if bert_type == 'bert':
        bert_path = '/home/diske/ivenwang/data/prev_trained_model/chinese-bert-wwm-ext'
        # bert_path = '/home/diske/ivenwang/data/prev_trained_model/bert-base'
    # elif bert_type == 'roberta':
    #     bert_path = '/home/diske/ivenwang/data/prev_trained_model/roberta-base'
    # elif bert_type == 'roberta-large':
    #     bert_path = '/home/diske/ivenwang/data/prev_trained_model/roberta-large'

    task = 1
    task = 2

    if task == 1:
        id2label = ['Ax', 'Va', 'Nz', 'Nt', 'No', 'Ad', 'An', 'Vi', 'Nl', 'Dc', 'Vc', 'Nx', 'Nn', 'Vu', 'Vd', 'Ub', 'Rc', 'Vn', 'Aa', 'Ux', 'Pc', 'Vp', 'Sy', 'As', 'Ny', 'Nc', 'Vo', 'Nh', 'Ng', 'Mc', 'Ac', 'Zc', 'Ua', 'Us', 'Ns', 'Oc', 'Mo', 'Cc', 'Vx', 'Uf', 'Vt', 'Fc', 'Ec', 'Qc']
    elif task == 2:
        id2label = ['rr', 'qd', 'lb', 'ws', 'Ng', 'nz', 'ui', 'f', 'vl', 'q', 'p', 'vq', 'ul', 'nt', 'wd', 'qz', 'an', 'jd', 'df', 'j', 'ld', 'Ug', 'Ag', 'jn', 'qv', 'b', 'lv', 'ql', 's', 'ue', 'a', 'uz', 'rzw', 'qj', 'l', 'wkz', 'r', 'wf', 'in', 'dc', 'iv', 'jb', 't', 'wp', 'o', 'Tg', 'nrf', 'vu', 'ww', 'qe', 'Qg', 'nx', 'qc', 'ad', 'wu', 'id', 'la', 'Mg', 'mq', 'qb', 'wm', 'ryw', 'nr', 'u', 'ns', 'v', 'Dg', 'i', 'jv', 'rz', 'k', 'w', 'n', 'ln', 'wj', 'c', 'z', 'nrg', 'Rg', 'd', 'wt', 'h', 'ia', 'vd', 'vx', 'wky', 'vi', 'ib', 'qr', 'e', 'wyy', 'y', 'ud', 'uo', 'wyz', 'Bg', 'm', 'qt', 'vn', 'ry', 'Vg', 'us', 'tt']
    label2id = dict()
    for i in range(len(id2label)):
        label2id[id2label[i]] = i

    model_choice = 'baseline' # bert + softmax

    data_path = '../data'
    train_data = os.path.join(data_path, 'Data%d_train_utf16.tag' % task)
    test_data = os.path.join(data_path, 'Data%d_test_utf16.tag' % task)

    # output paths
    output_path = '../output/%d/' % task
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    pkl_path =  os.path.join(data_path, 'pkl/')
    ckpt_path = os.path.join(output_path, 'ckpt/')
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    log_file = 'log-{time}.log'.format(time = time.strftime('%m%d-%H:%M:%S'))
    log_path = os.path.join(output_path, log_file)
    res_path = os.path.join(output_path, log_file[:-4] + '-predict.tag')

    # training params
    batch_size = 32
    test_batch_size = batch_size

    num_epochs = 5
    save_epoch = 1
    lr = 0.00007
    max_norm = 1.0 # Clipped gradient norm
    fp16 = 0 # 1: use mixed precision ; 0: use floating point 32
    gradient_acc_steps = 1 # No. of steps of gradient accumulation
    hidden_dropout_prob = 0.1
    seed = 1

    train = 1
    infer = 1
    continue_train = 0 # continue train from last time
    use_bio = 0

    gpuid = '0,1,2,3'

    import setproctitle
    setproctitle.setproctitle("POS %s %d %d" % (model_choice, task, seed))