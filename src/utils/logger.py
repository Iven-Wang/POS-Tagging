import json
import time

from config import Config
args = Config()

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(args.log_path, 'a+') as f_log:
            f_log.write('{time}  '.format(time = time.strftime('%m/%d %H:%M:%S')) + str(s) + '\n')

def logging_params():
    arg_dict = dict()
    for k,v in Config.__dict__.items():
        if (type(v) == type('1') and '/' not in v) or type(v) == type(1) or type(v) == type(True) or type(v) == type(0.1):
            arg_dict[k] = v
    logging("Configs:\n" + json.dumps(arg_dict, indent=4))