import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import numpy as np
from tensorboardX import SummaryWriter
from collections import OrderedDict

import torch.optim
import sys,shutil,os
import time
import src.networks as nets
from math import log10
from evaluation import AverageMeter
import pytorch_ssim as pytorch_ssim
from src.utils.osutils import mkdir_p, isfile, isdir, join
from src.utils.parallel import DataParallelModel, DataParallelCriterion
from src.utils.losses import VGGLoss




class BasicModel(object):
    def __init__(self, datasets =(None,None), models = None, args = None, **kwargs):
    #def __init__(self, datasets=None, models=None, args=None, **kwargs):
        super(BasicModel, self).__init__()
        
        self.args = args
        
        # create model
        print("==> creating model ")
        self.model = nets.__dict__[self.args.nets](args=args)
        print("==> creating model [Finish]")
       
        self.train_loader, self.val_loader = datasets
        #self.train_loader = datasets
        self.loss = torch.nn.MSELoss()
        
        self.title = args.name
        self.args.checkpoint = os.path.join(args.checkpoint, self.title)
        self.device = torch.device('cuda')
         # create checkpoint dir
        if not isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=args.lr,
                            betas=(args.beta1,args.beta2),
                            weight_decay=args.weight_decay)  
        
        if not self.args.evaluate:
            self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt')
        
        self.best_acc = 0
        self.is_best = False
        self.current_epoch = 0
        self.metric = -100000
        self.hl = 6 if self.args.hl else 1
        self.count_gpu = len(range(torch.cuda.device_count()))

        if self.args.lambda_style > 0:
            # init perception loss
            self.vggloss = VGGLoss(self.args.sltype).to(self.device)

        if self.count_gpu > 1 : # multiple
            # self.model = DataParallelModel(self.model, device_ids=range(torch.cuda.device_count()))
            # self.loss = DataParallelCriterion(self.loss, device_ids=range(torch.cuda.device_count()))
            self.model.multi_gpu()

        self.model.to(self.device)
        self.loss.to(self.device)

        print('==> Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        print('==> Total devices: %d' % (torch.cuda.device_count()))
        print('==> Current Checkpoint: %s' % (self.args.checkpoint))


    def train(self,epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossvgg = AverageMeter()
        
        # switch to train mode
        self.model.train()
        end = time.time()

        bar = Bar('Processing', max=len(self.train_loader)*self.hl)
        for _ in range(self.hl):
            for i, batches in enumerate(self.train_loader):
                # measure data loading time
                inputs = batches['image']
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)
                current_index = len(self.train_loader) * epoch + i

                if self.args.hl:
                    feeded = torch.cat([inputs,mask],dim=1)
                else:
                    feeded = inputs
                feeded = feeded.to(self.device)

                output = self.model(feeded)
                L2_loss =  self.loss(output,target) 
                
                if self.args.lambda_style > 0:
                    vgg_loss = self.vggloss(output,target,mask)
                else:
                    vgg_loss = 0

                total_loss = L2_loss + self.args.lambda_style * vgg_loss

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(L2_loss.item(), inputs.size(0))
                
                if self.args.lambda_style > 0 :
                    lossvgg.update(vgg_loss.item(), inputs.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f} | Loss VGG: {loss_vgg:.4f}'.format(
                            batch=i + 1,
                            size=len(self.train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            loss_vgg=lossvgg.avg
                            )

                if current_index % 1000 == 0:
                    print(suffix)
                
                if self.args.freq > 0 and current_index % self.args.freq == 0:
                    
                    self.validate(current_index)
                    self.flush()
                    self.save_checkpoint()
        
        self.record('train/loss_L2', losses.avg, current_index)              
        
    def validate(self, epoch):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ssimes = AverageMeter()
        psnres = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)
                
                if self.args.hl:
                    feeded = torch.cat([inputs,torch.zeros((1,4,self.args.input_size,self.args.input_size)).to(self.device)],dim=1)
                else:
                    feeded = inputs

                output = self.model(feeded)

                L2_loss = self.loss(output, target)

                psnr = 10 * log10(1 / L2_loss.item())   
                ssim = pytorch_ssim.ssim(output, target)    

                losses.update(L2_loss.item(), inputs.size(0))
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print("Epoches:%s,Losses:%.3f,PSNR:%.3f,SSIM:%.3f"%(epoch+1, losses.avg,psnres.avg,ssimes.avg))
        self.record('val/loss_L2', losses.avg, epoch)
        self.record('val/PSNR', psnres.avg, epoch)
        self.record('val/SSIM', ssimes.avg, epoch)
        
        self.metric = psnres.avg
        
    def resume(self,resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

        if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
            current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

        if self.args.start_epoch == 0:
            self.args.start_epoch = current_checkpoint['epoch']
        self.metric = current_checkpoint['best_acc']
        items = list(current_checkpoint['state_dict'].keys())

        ## restore the learning rate
        lr = self.args.lr
        for epoch in self.args.schedule:
            if epoch <= self.args.start_epoch:
                lr *= self.args.gamma
        optimizers = [getattr(self.model, attr) for attr in dir(self.model) if  attr.startswith("optimizer") and getattr(self.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))

#    def resume(self,resume_path):
#        resumePartNet(self.model, resume_path)
        
        
    def save_checkpoint(self,filename='checkpoint.pth.tar', snapshot=None):
        is_best = True if self.best_acc < self.metric else False

        if is_best:
            self.best_acc = self.metric

        state = {
                    'epoch': self.current_epoch + 1,
                    'nets': self.args.nets,
                    'state_dict': self.model.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer' : self.optimizer.state_dict() if self.optimizer else None,
                }

        filepath = os.path.join(self.args.checkpoint, filename)
        torch.save(state, filepath)

        if snapshot and state['epoch'] % snapshot == 0:
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))
        
        if is_best:
            self.best_acc = self.metric
            print('Saving Best Metric with PSNR:%s'%self.best_acc)
            if not os.path.exists(self.args.checkpoint): os.makedirs(self.args.checkpoint)
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'model_best.pth.tar'))

    def clean(self):
        self.writer.close()

    def record(self,k,v,epoch):
        self.writer.add_scalar(k, v, epoch)

    def flush(self):
        self.writer.flush()
        sys.stdout.flush()

    def norm(self,x):
        if self.args.gan_norm:
            return x*2.0 - 1.0
        else:
            return x

    def denorm(self,x):
        if self.args.gan_norm:
            return (x+1.0)/2.0
        else:
            return x

def set_dst_param(src_param, dst_param):
    dst_size = dst_param.size()
    src_size = src_param.size()
    if len(dst_size)==len(src_size) and len(dst_size)>0:
        d0 = min(dst_size[0],src_size[0])
        if len(dst_size)==1:
            dst_param[0:d0]=src_param[0:d0]
        else:
            d1 = min(dst_size[1],src_size[1])
            if len(dst_size)==2:
                dst_param[0:d0,0:d1]=src_param[0:d0,0:d1]
            else:
                d2 = min(dst_size[2],src_size[2])
                s2 = (dst_size[2]-d2)//2
                s2_1 = (src_size[2]-d2)//2
                if len(dst_size)==3:
                    dst_param[0:d0,0:d1,s2:s2+d2]=src_param[0:d0,0:d1,s2_1:s2_1+d2]
                else:
                    d3 = min(dst_size[3],src_size[3])
                    s3 = (dst_size[3]-d3)//2
                    s3_1 = (src_size[3]-d3)//2
                    if len(dst_size)==4:
                        dst_param[0:d0,0:d1,s2:s2+d2,s3:s3+d3]=src_param[0:d0,0:d1,s2_1:s2_1+d2,s3_1:s3_1+d3]
                    else:
                        d4 = min(dst_size[4],src_size[4])
                        s4 = (dst_size[4]-d4)//2
                        s4_1 = (src_size[4]-d4)//2
                        if len(dst_size)==5:
                            dst_param[0:d0,0:d1,s2:s2+d2,s3:s3+d3,s4:s4+d4]=src_param[0:d0,0:d1,s2_1:s2_1+d2,s3_1:s3_1+d3,s4_1:s4_1+d4]
                        else:
                            raise Exception('wrong dimension of src parameter: %d'%len(dst_size))
    else:
        raise Exception('inconsistent dimension: dst vs src (%d vs %d)'%(len(dst_size), len(src_size)))

def start_with(inp, strs):
    for tmp in strs:
        if inp.startswith(tmp):
            return True
    return False
    
def resumePartNet(model,model_root):
    #model_dir=os.path.join(model_root,version,fil)
    model_dir = model_root
    #model_dir1, model_dir2 = model_root.split(' ')
    src_state_dict=torch.load(model_dir, map_location=lambda storage, loc:storage)
    
#    new_state_dict = OrderedDict()
#    for k in src_state_dict['state_dict']:
#        name = k.replace('module.', '')
#        new_state_dict[name] = src_state_dict['state_dict'].setdefault(k)
#    src_state_dict['state_dict'] = new_state_dict

#    new_state_dict = OrderedDict()
#    for k in src_state_dict['state_dict']:
#        name = k.split('.')
#        name1 = name[0] + '_0.module.'
#        name2 = '.'.join(name[1:])
#        name = name1 + name2
#        new_state_dict[name] = src_state_dict['state_dict'].setdefault(k)
#    src_state_dict['state_dict'] = new_state_dict

    #src_state_dict['state_dict'] = src_state_dict['state_dict'].module
    print(src_state_dict['epoch'])
    src_state_dict = src_state_dict['state_dict']
    dst_state_dict = model.state_dict()
    for src_name,src_param in src_state_dict.items():
        strs=src_name.split('.')
        #if strs[0] in part:
        if src_name not in dst_state_dict.keys():
            print('Parameter {} does not exists!'.format(src_name))
        else:
            if isinstance(src_param,torch.nn.Parameter):
                src_param = src_param.data
            dst_param=dst_state_dict[src_name]
            if src_param.size()!=dst_param.size():
                print('inconsistent sizes for {}'.format(src_name))
                set_dst_param(src_param,dst_param)
            else:
                dst_param.copy_(src_param)
                
#    src_state_dict=torch.load(model_dir2, map_location=lambda storage, loc:storage)
#    
##    new_state_dict = OrderedDict()
##    for k in src_state_dict['state_dict']:
##        name = k.replace('module.', '')
##        new_state_dict[name] = src_state_dict['state_dict'].setdefault(k)
##    src_state_dict['state_dict'] = new_state_dict
#
#    new_state_dict = OrderedDict()
#    for k in src_state_dict['state_dict']:
#        name = k.split('.')
#        name1 = name[0] + '.module.'
#        name2 = '.'.join(name[1:])
#        name = name1 + name2
#        new_state_dict[name] = src_state_dict['state_dict'].setdefault(k)
#    src_state_dict['state_dict'] = new_state_dict
#
#    #src_state_dict['state_dict'] = src_state_dict['state_dict'].module
#    print(src_state_dict['epoch'])
#    src_state_dict = src_state_dict['state_dict']
#    dst_state_dict = model.state_dict()
#    for src_name,src_param in src_state_dict.items():
#        strs=src_name.split('.')
#        #if strs[0] in part:
#        if src_name not in dst_state_dict.keys():
#            print('Parameter {} does not exists!'.format(src_name))
#        else:
#            if isinstance(src_param,torch.nn.Parameter):
#                src_param = src_param.data
#            dst_param=dst_state_dict[src_name]
#            if src_param.size()!=dst_param.size():
#                print('inconsistent sizes for {}'.format(src_name))
#                set_dst_param(src_param,dst_param)
#            else:
#                dst_param.copy_(src_param)