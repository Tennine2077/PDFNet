import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from soc_metrics import soc_metrics
import datetime
import cv2
import numpy as np
import ttach as tta
import time
import sys
sys.path.append("..")
from dataloaders.Mydataset import MyDataset,GOSNormalize,get_files
from models.PDFNet import *
from Train_VIDIF import get_args_parser

sys.argv = ['run.py']

TEST_MODE = False
# TEST_MODE = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser('HQD evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.device = device
args.DEBUG = TEST_MODE

model,model_name = build_model(args)

model.load_state_dict(torch.load(r'/home/PDFNet/PDFNet/checkpoints/PDFNet_Best.pth',map_location='cpu'),strict=False)
model = model.to(device)

test_dir = {
    # "HRSOD":r"/home/DATA/HRSOD_test/images",
    # "UHRSOD":r"/home/DATA/UHRSD_TE_2K/images"
    # "DIV2K":r"/home/DATA/DIV2K/images",
    # "Flickr2K":r"/home/DATA/Flickr2K/images",
    # "COIFT":r'/home/DATA/COIFT/images',
    "DIS-VD":r'/home/DATA/DIS-DATA/DIS-VD/images',
    # "DIS-TE1":r'/home/DATA/DIS-DATA/DIS-TE1/images',
    # "DIS-TE2":r'/home/DATA/DIS-DATA/DIS-TE2/images',
    # "DIS-TE3":r'/home/DATA/DIS-DATA/DIS-TE3/images',
    # "DIS-TE4":r'/home/DATA/DIS-DATA/DIS-TE4/images'
            }

test_time = str(datetime.datetime.today()).replace(' ','_').replace(':','_')[:-7]
save_dir = None
file_name = f"HRSOD_{model_name}{test_time}"
if save_dir is None:
    save_dir = f'/home/your_results/{file_name}'
if not TEST_MODE:
    os.makedirs(save_dir,exist_ok=True)

to_pil = transforms.ToPILImage()

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1,1.25], interpolation='bilinear', align_corners=False),
    ]
)

starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

for keys in test_dir:
    if not TEST_MODE:
        os.makedirs(f'{save_dir}/{keys}',exist_ok=True)
    test_datatset = MyDataset(root=test_dir[keys],transform=[
        GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
        chached=False,size=[1024,1024],use_gt=False)

    BATCH_SIZE = 1
    test_loader = DataLoader(dataset=test_datatset,batch_size=BATCH_SIZE,shuffle=False,num_workers=8,persistent_workers=True)

    model.eval()
    with torch.no_grad():
        iter_pbar = tqdm(total=len(test_loader))
        all_time = torch.zeros(len(test_loader))   
        for i,data in enumerate(test_loader):
            image_name, image_size, inputs, gt,= data['image_name'], data['image_size'], data['image'],data['gt']
            inputs,gt = inputs.to(device),gt.to(device)
            depth=data['depth'].to(device)
            mask = []
            for transformer in transforms:  
                rgb_trans = transformer.augment_image(inputs)
                depth_trans = transformer.augment_image(depth)
                starter.record()
                pred_grad_sigmoid,pred_grad = model.inference(rgb_trans,depth_trans)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                all_time[i] = curr_time
                deaug_mask = transformer.deaugment_mask(pred_grad)
                mask.append(deaug_mask)
            prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
            prediction = prediction.sigmoid()

            for k in range(inputs.shape[0]):

                save_name = image_name[k].split('/')[-1].split('.pt')[0].replace('.jpg','.png')
                w_,h_ = image_size
                prediction = to_pil(prediction.squeeze(0).cpu())
                prediction = prediction.resize((h_,w_), Image.BILINEAR)
                if not TEST_MODE:
                    prediction.save(f'{save_dir}/{keys}/{save_name}')
                if TEST_MODE:
                    break
                iter_pbar.update()
            if TEST_MODE:
                break
        if TEST_MODE:
            break
        iter_pbar.close()
        torch.cuda.empty_cache()
        mean_time = all_time.mean()
        print(f"inference time:{mean_time}ms/iter, FPS:{1000/mean_time}")

soc_metrics(file_name)