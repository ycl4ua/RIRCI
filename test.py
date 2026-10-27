from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import time


def is_dic(x):
    return type(x) == type([])



def tensor2np(x, isMask=False):
    if isMask:
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x = ((x.cpu().detach()))*255
    else:
        x = x.cpu().detach()
        mean = 0
        std = 1
        x = (x * std + mean)*255
		
    return x.numpy().transpose(0,2,3,1).astype(np.uint8)

def save_output(inputs, preds, save_dir, img_fn, extra_infos=None,  verbose=False, alpha=0.5):
    outs = []
    image, bg_gt,mask_gt = inputs['I'], inputs['bg'], inputs['mask']
    image = cv2.cvtColor(tensor2np(image)[0], cv2.COLOR_RGB2BGR)
    bg_gt = cv2.cvtColor(tensor2np(bg_gt)[0], cv2.COLOR_RGB2BGR)
    mask_gt = tensor2np(mask_gt, isMask=True)[0]

    bg_pred,mask_preds = preds['bg'], preds['mask']
    bgr, bgi, bgc = preds['bgr'], preds['bgi'], preds['bgc']
    bg_pred = cv2.cvtColor(tensor2np(bg_pred)[0], cv2.COLOR_RGB2BGR)
    mask_preds = [tensor2np(m, isMask=True)[0] for m in mask_preds]
    mask_pred = mask_preds[0]
    outs = [image, bg_gt, bg_pred,cv2.cvtColor(tensor2np(bgr)[0], cv2.COLOR_RGB2BGR), cv2.cvtColor(tensor2np(bgi)[0], cv2.COLOR_RGB2BGR), cv2.cvtColor(tensor2np(bgc)[0], cv2.COLOR_RGB2BGR)] #, main_mask]
    outimg = np.concatenate(outs, axis=1)
	
    if verbose==True:
        cv2.imshow("out",outimg)
        cv2.waitKey(0)
    else:
        psnr = extra_infos['psnr']
        rmsew = extra_infos['rmsew']
        f1 = extra_infos['f1']

        img_fn = os.path.split(img_fn)[-1]
        out_fn = os.path.join(save_dir, "{}_psnr_{:.2f}_rmsew_{:.2f}_f1_{:.4f}{}".format(os.path.splitext(img_fn)[0],psnr,rmsew, f1, os.path.splitext(img_fn)[1]))
        cv2.imwrite(out_fn, outimg)





def main(args):
    args.dataset = args.dataset.lower()
    if args.dataset == 'rirci':
        dataset_func = datasets.RIRCIDataset
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    val_loader = torch.utils.data.DataLoader(dataset_func(args.eval_split,args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    data_loaders = (None,val_loader)

    Machine = models.__dict__[args.models](datasets=data_loaders, args=args)

    
    model = Machine
    model.model.eval()
    print("==> testing VM model ")
    rmses = AverageMeter()
    rmsews = AverageMeter()
    ssimesx = AverageMeter()
    psnresx = AverageMeter()
    maskIoU = AverageMeter()
    maskF1 = AverageMeter()
    prime_maskIoU = AverageMeter()
    prime_maskF1 = AverageMeter()
    processTime = AverageMeter()

    prediction_dir = os.path.join(args.checkpoint, 'results')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    
    save_flag = True
    with torch.no_grad():
        for i, batches in enumerate(model.val_loader):


            inputs = batches['image'].to(model.device)
            target = batches['target'].to(model.device)
            mask = batches['mask'].float().to(model.device)
            img_path = batches['img_path']

            outputs = model.model(model.norm(inputs))

            imoutput, immask= outputs
            immask = immask[0]
            
            imcoarse = None
            imoutput_restoration = imoutput[1]
            imoutput_inpainting = imoutput[2]
            imoutput_cleancomp = imoutput[3]
            imoutput = imoutput[0] if is_dic(imoutput) else imoutput

            imfinal = model.denorm(imoutput*immask + model.norm(inputs)*(1-immask))

            eps = 1e-6
            

            imfinal =imoutput*immask + model.norm(inputs)*(1-immask)
            imfinal_r =imoutput_restoration*immask + model.norm(inputs)*(1-immask)
            imfinal_i = imoutput_inpainting*immask + model.norm(inputs)*(1-immask)
            imfinal_c = imoutput_cleancomp*immask + model.norm(inputs)*(1-immask)
            psnrx = 10 * log10(1 / F.mse_loss(imfinal,target).item())       
            final_np = (imfinal.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            target_np = (target.detach().cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            ssimx = pytorch_ssim.ssim(imfinal, target)
            
            
            
            rmsex = compute_RMSE(imfinal, target, mask, is_w=False)
            rmsewx = compute_RMSE(imfinal, target, mask, is_w=True)
            rmses.update(rmsex, inputs.size(0))
            rmsews.update(rmsewx, inputs.size(0))
            psnresx.update(psnrx, inputs.size(0))
            ssimesx.update(ssimx, inputs.size(0))
            
            iou = compute_IoU(immask, mask)
            maskIoU.update(iou)
            f1 = FScore(immask, mask).item()
            maskF1.update(f1, inputs.size(0))

            if save_flag:
                save_output(
                    inputs={'I':inputs, 'bg':target,  'mask':mask}, 
                    preds={'bg':imfinal, 'mask':[immask], 'bgr':imfinal_r, 'bgi':imfinal_i, 'bgc':imfinal_c}, 
                    save_dir=prediction_dir, 
                    img_fn=img_path[0], 
                    extra_infos={"psnr":psnrx, "rmsew":rmsewx, "f1":f1},
                    verbose=False
                )
            if i % 100 == 0:
                print("Batch[%d/%d]| PSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(i,len(model.val_loader),psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("Total:\nPSNR:%.4f | SSIM:%.4f | RMSE:%.4f | RMSEw:%.4f | primeIoU:%.4f, primeF1:%.4f | maskIoU:%.4f | maskF1:%.4f | time:%.2f"
                %(psnresx.avg,ssimesx.avg, rmses.avg, rmsews.avg, prime_maskIoU.avg, prime_maskF1.avg, maskIoU.avg, maskF1.avg, processTime.avg))
    print("DONE.\n")


if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())
    
