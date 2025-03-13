from torch import optim as optim
import torch
from torch.autograd import Variable
import numpy as np
from metric_tools.F1torch import f1score_torch
import torch.nn.functional as F
import torch.nn as nn
import os
import glob
import torchvision.transforms as transforms

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src

def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs

def build_optimizer(args, model):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return build_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args)
    )


def build_optimizer_v2(model,
                    opt: str = 'sgd',
                    lr = None,
                    weight_decay: float = 0.,
                    momentum: float = 0.9,
                    **kwargs):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_args = dict(weight_decay=weight_decay, **kwargs)

    opt_lower = opt.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=momentum, lr=lr, nesterov=True, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr, **opt_args)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def keep_n_files(directory, n):
    # 获取目录下所有文件的路径和最后修改时间
    files = [(file_path, os.path.getmtime(file_path)) for file_path in glob.glob(os.path.join(directory, '*'))]
    
    # 按照最后修改时间从近到远排序
    files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除多余的文件，保留最近的n个文件
    for file_path, _ in files[n:]:
        os.remove(file_path)


def eval_cycle(this_checkpoints_dir,model,epoch,test_datatset,test_loader,best_valid_mae,args):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mybins = np.arange(0,256)
        val_num = test_datatset.__len__()
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))
        ACC = np.zeros((val_num))
        valid_iter = 0
        for i,data in enumerate(test_loader):
            name, inputs, gt, labels = data['image_name'], data['image'],data['gt'], data['label']
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
            pred_grad,transition,pred_cls = model.inference_cycle(inputs_v)
            for k in range(inputs.shape[0]):
                inputs_k = inputs[k]
                gt_k = gt[k]
                pred_grad_k = pred_grad[k].cpu().detach()
                pre, rec, f1 = f1score_torch(pred_grad_k,gt_k)
                mae = nn.L1Loss()(pred_grad_k,gt_k.float())
                PRE[valid_iter+k,:] = pre.cpu().detach()
                REC[valid_iter+k,:] = rec.cpu().detach()
                F1[valid_iter+k,:] = f1.cpu().detach()
                MAE[valid_iter+k] = mae.cpu().detach()
                ACC[valid_iter+k] = model.acc(pred_cls,labels_v+1).cpu().detach()
                print(f'{valid_iter+k+1}/{val_num}, image_name: '+name[k])
            valid_iter += inputs.shape[0]
            # break
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)
        tmp_f1 = np.amax(f1_m)
        tmp_mae = np.mean(MAE)
        tmp_acc = np.mean(ACC)
        if tmp_mae < best_valid_mae or best_valid_mae < 0:
            best_valid_mae = tmp_mae
            if args.checkpoints_save_path:
                torch.save(model.state_dict(),f'{this_checkpoints_dir}/{epoch}_best_MAE_{best_valid_mae:.6f}_{args.model}.pth')
                keep_n_files(this_checkpoints_dir,n=3)
        torch.cuda.empty_cache()
    
    return tmp_f1,tmp_mae,tmp_acc,best_valid_mae,inputs_k,gt_k,pred_grad_k,transition

def eval(this_checkpoints_dir,model,epoch,test_datatset,test_loader,best_valid,train_time,args):
    to_pil = transforms.ToPILImage()
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mybins = np.arange(0,256)
        val_num = test_datatset.__len__()
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))
        valid_iter = 0
        if args.COPY:
            os.makedirs(f'valid_sample/{args.model}{train_time}/{epoch+1}',exist_ok=True)
        for i,data in enumerate(test_loader):
            name, inputs, gt, labels = data['image_name'], data['image'],data['gt'], data['label']
            depth = data['depth']
            if args.device!='cpu':
                inputs_v, labels_v = Variable(inputs.to(args.device), requires_grad=False), Variable(labels.to(args.device), requires_grad=False)
                depth_v = Variable(depth.to(args.device), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
                depth_v = Variable(depth, requires_grad=False)
            pred_grad,_ = model.inference(inputs_v,depth_v)
            for k in range(inputs.shape[0]):
                inputs_k = inputs[k]
                gt_k = gt[k]
                pred_grad_k = pred_grad[k].cpu().detach()
                pred_grad_k = F.upsample(pred_grad_k[None,...],size=gt_k.shape[1:],mode='bilinear')[0]
                prediction = to_pil(pred_grad_k.squeeze(0).cpu())
                gt_name = name[k].split(r'/')[-1]
                if args.COPY:
                    prediction.save(f'valid_sample/{args.model}{train_time}/{epoch+1}/{gt_name}')
                pre, rec, f1 = f1score_torch(pred_grad_k,gt_k)
                mae = nn.L1Loss()(pred_grad_k,gt_k.float())
                PRE[valid_iter+k,:] = pre.cpu().detach()
                REC[valid_iter+k,:] = rec.cpu().detach()
                F1[valid_iter+k,:] = f1.cpu().detach()
                MAE[valid_iter+k] = mae.cpu().detach()
                print(f'{valid_iter+k+1}/{val_num}, image_name: '+name[k])
            valid_iter += inputs.shape[0]
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)
        tmp_f1 = np.amax(f1_m)
        tmp_mae = np.mean(MAE)
        if args.eval_metric == 'F1':
            if tmp_f1 >= best_valid or best_valid < 0:
                best_valid = tmp_f1
                if args.checkpoints_save_path and not args.DEBUG:
                    torch.save(model.state_dict(),f'{this_checkpoints_dir}/{epoch+1}_best_f1_{best_valid:.6f}_mae_{tmp_mae:.6f}_{args.model}.pth')
                    keep_n_files(this_checkpoints_dir,n=3)
        elif args.eval_metric == 'MAE':
            if tmp_mae <= best_valid or best_valid < 0:
                best_valid = tmp_mae
                if args.checkpoints_save_path and not args.DEBUG:
                    torch.save(model.state_dict(),f'{this_checkpoints_dir}/{epoch+1}_best_mae_{best_valid:.6f}_f1_{tmp_f1:.6f}_{args.model}.pth')
                    keep_n_files(this_checkpoints_dir,n=3)
        if args.COPY:
            os.rename(f'valid_sample/{args.model}{train_time}/{epoch+1}',f'valid_sample/{args.model}{train_time}/{epoch+1}_f1_{tmp_f1:.6f}_mae_{tmp_mae:.6f}')
        torch.cuda.empty_cache()
    
    return tmp_f1,tmp_mae,best_valid,inputs_k,gt_k,pred_grad_k