import argparse
import torch
import numpy as np
from pathlib import Path
from dataloaders import Mydataset as Mydataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from torch.autograd import Variable
import utiles
import os
from timm.scheduler import create_scheduler
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter
from models.PDFNet import build_model
import random
from torch.cuda.amp import autocast, GradScaler
import shutil

def copy_allfiles(src,dest,not_case = ["valid_sample","runs"]):
  for root, dirs, files in os.walk(src):
    # 计算目标文件夹中的对应路径
    relative_path = os.path.relpath(root, src)
    flag = 0
    for not_case_ in not_case:
        if not_case_ in relative_path:
            flag = 1
            break
    if flag == 1:
        continue
    target_subfolder = os.path.join(dest, relative_path)
    # 创建对应的子文件夹
    os.makedirs(target_subfolder, exist_ok=True)
    # 复制文件
    for file_name in files:
        source_file = os.path.join(root, file_name)
        target_file = os.path.join(target_subfolder, file_name)
        shutil.copy(source_file, target_file)

def get_files(PATH):
    file_lan = []
    for filepath,dirnames,filenames in os.walk(PATH):
        for filename in filenames:
            file_lan.append(os.path.join(filepath,filename))
    return file_lan

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def main(args):
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    seed = args.seed
    setup_seed(seed)

    dataset_train= Mydataset.build_dataset(is_train=True, args=args)
    dataset_val = Mydataset.build_dataset(is_train=False, args=args)

    data_loader_train = DataLoader(dataset=dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,persistent_workers=True)
    data_loader_val = DataLoader(dataset=dataset_val,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,persistent_workers=True)

    print(f"Creating model: {args.model}")
    model,model_name = build_model(args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        model_dict = model.state_dict()
        checkpoint_model =  {k: v for k, v in checkpoint_model.items() if k in model_dict  and v.shape == model_dict[k].shape}
        model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', round(n_parameters/1024/1024),"M")
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % (args.batch_size * args.update_freq))
    print("Update frequent = %d" % args.update_freq)

    optimizer = utiles.build_optimizer(args, model)
    lr_scheduler, EPOCH = create_scheduler(args, optimizer)

    train_time = str(datetime.datetime.today()).replace(' ','_').replace(':','_')[:-7]
    if not args.DEBUG:
        writer = SummaryWriter(log_dir='runs/'+args.model+train_time)

    this_checkpoints_dir = None
    train_time = str(datetime.datetime.today()).replace(' ','_').replace(':','_')[:-7]
    if this_checkpoints_dir is None:
        this_checkpoints_dir = f'{args.checkpoints_save_path}/{model_name}{train_time}'
    if not args.DEBUG:
        os.makedirs(this_checkpoints_dir,exist_ok=True)

    best_valid_f1 = -1
    best_valid_mae = -1
    iter_pbar = tqdm(total=len(data_loader_train))
    mean_epoch_time = 0
    rest_time = 0

    tmp_f1,tmp_mae = 0,0

    scaler = GradScaler()
    if not args.DEBUG:
        os.makedirs(f'valid_sample/{args.model}{train_time}',exist_ok=True)
        if args.COPY:
            os.makedirs(f'valid_sample/{args.model}{train_time}/project_copy', exist_ok=True)
            copy_allfiles(os.getcwd(),f'valid_sample/{args.model}{train_time}/project_copy')
    large_loss_name_list = []
    for epoch in range(EPOCH):
        if epoch < args.finetune_epoch and args.finetune_epoch > 0:
            lr_scheduler.step(epoch)
            continue

        loss_list = []
        model.train()
        epoch_loss,epoch_R_loss = 0,0
        epoch_target_loss = 0
        epoch_starttime = datetime.datetime.now()
        iters = 0
        for i,data in enumerate(data_loader_train):
            
            # if args.is_break:
            # break
            if epoch % 2 == 1 and args.update_half:
                if not data['image_name'] in large_loss_name_list:
                    iter_pbar.update()
                    continue
                else:
                    large_loss_name_list.remove(data['image_name'])
            inputs, gt, labels = data['image'],data['gt'],data['label']
            depth,depth_large = data['depth'],data['depth_large']
            if args.device != 'cpu':
                inputs_v, gt_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(gt.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)
                depth_v = Variable(depth.to(device), requires_grad=False)
                depth_large_v = Variable(depth_large.to(device), requires_grad=False)
            else:
                inputs_v, gt_v, labels_v = Variable(inputs, requires_grad=False), Variable(gt, requires_grad=False), Variable(labels, requires_grad=False)
                depth_v = Variable(depth, requires_grad=False)
                depth_large_v = Variable(depth_large, requires_grad=False)
            with autocast():
                outputs = model(inputs_v,depth_v,gt_v,depth_large_v)
                pred,loss,target_loss = outputs[0],outputs[1],outputs[2]
                loss = loss / args.update_freq
            scaler.scale(loss).backward()
            iters += 1
            if iters % args.update_freq == 0 or iters==len(data_loader_train):    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            with torch.no_grad():
                loss_list.append({"image_name":data['image_name'],'loss':float(target_loss.cpu().detach())})
                epoch_loss+=float(loss.cpu().detach().item()*args.update_freq)
                epoch_target_loss+=float(target_loss.cpu().detach())
                iter_pbar.update()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                if args.eval_metric == 'F1':
                    iter_pbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, mean epoch time: {mean_epoch_time}s, rest time: {rest_time/3600:.2f}h, '
                                                +f'Train loss: {epoch_loss/(i+1):.4f}, '
                                                +f'target loss: {epoch_target_loss/(i+1):.4f}, '
                                                +f'##'
                                                +f'Valid F1: {tmp_f1:.4f}, '
                                                +f'mae: {tmp_mae:.4f},'
                                                +f'##'
                                                +f'best F1: {best_valid_f1:.4f}, '
                                                +f'LR:{lr:.8f}'
                                            )
                elif args.eval_metric == 'MAE':
                    iter_pbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, mean epoch time: {mean_epoch_time}s, rest time: {rest_time/3600:.2f}h, '
                                                +f'Train loss: {epoch_loss/(i+1):.4f}, '
                                                +f'target loss: {epoch_target_loss/(i+1):.4f}, '
                                                +f'##'
                                                +f'Valid F1: {tmp_f1:.4f}, '
                                                +f'mae: {tmp_mae:.4f},'
                                                +f'##'
                                                +f'best MAE: {best_valid_mae:.4f}, '
                                                +f'LR:{lr:.8f}'
                                            )
                del outputs, loss, inputs_v, gt_v, labels_v, pred, target_loss
                gc.collect()
        torch.cuda.empty_cache()
        if args.eval:
            tmp_f1,tmp_mae,best_valid,inputs_k,gt_k,pred_grad_k = utiles.eval(this_checkpoints_dir,model,epoch,dataset_val,data_loader_val,best_valid_f1,train_time,args)
            if args.eval_metric == 'F1':
                best_valid_f1 = best_valid
            elif args.eval_metric == 'MAE':
                best_valid_mae = best_valid
            if not args.DEBUG:
                writer.add_scalar('Train/loss',epoch_loss/(iters),epoch+1)
                writer.add_scalar('Train/target_loss',epoch_target_loss/(iters),epoch+1)
                writer.add_scalar('Valid/F1',tmp_f1,epoch+1)
                writer.add_scalar('Valid/mae',tmp_mae,epoch+1)
                writer.add_scalar('Lr',lr,epoch+1)
                writer.add_image('Image/image',np.array(inputs_k.cpu().detach().permute(1,2,0)),dataformats='HWC',global_step=epoch+1)
                writer.add_image('Image/GT',np.array(gt_k.cpu().detach().permute(1,2,0)),dataformats='HWC',global_step=epoch+1)
                writer.add_image('Image/Pred',np.array(pred_grad_k.cpu().detach().permute(1,2,0)),dataformats='HWC',global_step=epoch+1)
            

        else:
            if not args.DEBUG:
                writer.add_scalar('Train/loss',epoch_loss/(iters),epoch+1)
                writer.add_scalar('Train/target_loss',epoch_target_loss/(iters),epoch+1)
        epoch_endtime = datetime.datetime.now()
        mean_epoch_time = (epoch_endtime-epoch_starttime).seconds
        rest_time = (EPOCH-epoch-1)*mean_epoch_time
        iter_pbar.reset()
        lr_scheduler.step(epoch)
        torch.cuda.empty_cache()
        torch.save(model.state_dict(),f'{this_checkpoints_dir}/LAST.pth')
        large_loss_list = sorted(loss_list, key = lambda i: i['loss'], reverse=True)
        large_loss_name_list = [item['image_name'] for item in large_loss_list][:loss_list.__len__()//2]
    if not args.DEBUG:
        torch.save(model.state_dict(),f'{this_checkpoints_dir}/{epoch}_last_F1_{tmp_f1:.6f}_mae_{tmp_mae:.6f}_{args.model}.pth')
    iter_pbar.close()