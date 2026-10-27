from __future__ import print_function, absolute_import

import argparse
import getpass
import torch,time,os
from datetime import date, datetime
import sys
from src.utils.misc import adjust_learning_rate
import src.models as models
from PIL import Image,ImageFile
import datasets as datasets
from options import Options
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings

warnings.filterwarnings("ignore")

# 默认args中名字使用'-'替代'_'
def save_args(timestamp, user, filname, args):
    log = '# time: %s by %s\npython %s ' % (timestamp, user, filname)
    args_v = vars(args)
    for k, key in enumerate(args_v):
        key_str = key.replace('_', '-')
        value = args_v[key]
        value_str = ''
        if isinstance(value, list):
            for i, v in enumerate(value):
                value_str = value_str+str(v)
                if i < len(value)-1:
                    value_str = value_str+' '
        elif isinstance(value, str):
            value_str = '\"%s\"' % value
        else:
            value_str = str(value)
        print(key, args_v[key])
        if k < len(args_v)-1:
            if (k+1)%6 == 0:
                log = log+'--%s %s \\\n' % (key_str, value_str)
            else:
                log = log+'--%s %s ' % (key_str, value_str)
        else:
            log = log+'--%s %s\n' % (key_str, value_str)
    logfilname = 'script-%s.txt' % user
    with open(logfilname, 'a') as f:
        f.write(log)
        
def main(args):
    args.seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == 'rirci':
        dataset_func = datasets.RIRCIDataset
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
        
    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
    num_workers=args.workers, pin_memory=True)
    #print('this is length_train_loder',len(train_loader))
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
    num_workers=args.workers, pin_memory=True)

    lr = args.lr
    data_loaders = (train_loader,val_loader)
    #data_loaders = train_loader
    model = models.__dict__[args.models](datasets=data_loaders, args=args)
    print('============================ Initization Finish && Training Start =============================================')

    for epoch in range(model.args.start_epoch, model.args.epochs):
        lr = adjust_learning_rate(data_loaders, model, epoch, lr, args)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        model.record('lr',lr, epoch)        
        model.train(epoch)
        #break
        # model.validate(epoch)
        if args.freq < 0:
            model.validate(epoch)
            model.flush()
            model.save_checkpoint()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    
    args = parser.parse_args()
    
    day = date.today().strftime("%b-%d-%Y")
    now = datetime.now().strftime("%H:%M:%S")
    timestamp = day+' '+now
    save_args(timestamp, getpass.getuser(), sys.argv[0], args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('==================================== WaterMark Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    print('==> {:50}: {:<}'.format("USE GPU",os.environ['CUDA_VISIBLE_DEVICES']))
    print('==================================== Stable Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) == parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Changed Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) != parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==================================== FINISH WITHOUT ERROR =============================================')
