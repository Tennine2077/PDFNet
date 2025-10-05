# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy.random as random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, eps=1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps  # 防止log(0)的微小常数

    def forward(self, pred, target):
        # 将输入转换为float32计算关键部分
        pred = pred.float()
        target = target.float()
        
        # 添加eps防止log(0)并提升数值稳定性
        diff_log = torch.log(target + self.eps) - torch.log(pred + self.eps)
        loss = torch.sqrt(
            (diff_log ** 2).mean() - self.lambd * (diff_log.mean() ** 2) + self.eps
        )
        return loss

class IntegrityPriorLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon

        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        
        self.sobel_x.weight.data = sobel_kernel_x
        self.sobel_y.weight.data = sobel_kernel_y
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, mask, depth_map, gt):
        #对FP计算与均值的差距，越远loss越高
        #对FN计算与均值的差距，越近loss越高
        py = gt*mask + (1-gt)*(1-mask)
        FP = (1-py)*mask
        FN = (1-py)*gt
        logP = -torch.log(py + self.epsilon)
        diff = (depth_map-((depth_map*gt).sum()/(gt.sum()+self.epsilon)))**2
        FPdiff = (diff)*FP
        FNdiff = (1-diff)*FN
        vareight = (FPdiff+FNdiff)*py    
        variance = logP * vareight  # [B,1]
        variance_loss = torch.mean(variance)

        grad_x = abs(self.sobel_x(depth_map))  # [B,1,H,W]
        grad_y = abs(self.sobel_y(depth_map))  # [B,1,H,W]
        
        masked_grad_x = grad_x * logP
        masked_grad_y = grad_y * logP
        
        grad = (masked_grad_x + masked_grad_y)
        grad_loss = torch.mean(grad)

        total_loss = variance_loss + grad_loss
        return total_loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - (1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2

def circular_highPassFiltering(img, ratio):
    device = img.device
    batch_size,_,height,width = img.shape
    sigma = (height * (ratio[...,None,None])) / 4
    center_h = height // 2
    center_w = width // 2
    grid_y, grid_x = torch.meshgrid(torch.arange(-center_h, height - center_h),
                                    torch.arange(-center_w, width - center_w))
    grid_y = grid_y[None,None,...].repeat(batch_size, 1, 1, 1).to(device)
    grid_x = grid_x[None,None,...].repeat(batch_size, 1, 1, 1).to(device)
    # 按照二维高斯分布公式计算每个位置的值
    gaussian_values = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    gmin = gaussian_values.flatten(-2).min(dim=-1)[0][...,None,None]
    gmax = gaussian_values.flatten(-2).max(dim=-1)[0][...,None,None]
    decreasing_matrix = (gaussian_values-gmin) / (gmax-gmin)  # 根据归一化距离计算灰度值
    mask = ((0.5-decreasing_matrix)*100).sigmoid()
    fft = torch.fft.fft2(img)
    fft_shift = torch.fft.fftshift(fft,dim=(2,3))
    fft_shift = torch.mul(fft_shift, mask)
    idft_shift = torch.fft.ifftshift(fft_shift,dim=(2,3))
    ifimg = torch.fft.ifft2(idft_shift)
    ifimg = torch.abs(ifimg)
    ifmin = ifimg.flatten(-2).min(dim=-1)[0][...,None,None]
    ifmax = ifimg.flatten(-2).max(dim=-1)[0][...,None,None]
    ifimg = (ifimg-ifmin) / (ifmax-ifmin)  # 根据归一化距离计算灰度值
    return mask,ifimg

def _upsample_like(src,tar,mode='bilinear'):
    if mode == 'bilinear':
        src = F.upsample(src,size=tar.shape[2:],mode=mode,align_corners=True)
    else:
        src = F.upsample(src,size=tar.shape[2:],mode=mode)
    return src

def _upsample_(src,size,mode='bilinear'):
    if mode == 'bilinear':
        src = F.upsample(src,size=size,mode=mode,align_corners=True)
    else:
        src = F.upsample(src,size=size,mode=mode)
    return src

def patchfy(x,p=4,c=4):
    h = w = x.shape[2] // p
    x = x.reshape(shape=(x.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(x.shape[0], h * w, p**2 * c))
    return x
    
def unpatchfy(x,p=4,c=4):
    h = w = round(x.shape[1]**0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    x = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return x


# def structure_loss(pred, mask):
#     size = 15
#     pad = size//2
#     M_edge = mask
#     N_edge = 1 - M_edge
#     for i in range(2):
#         M_edge = abs(torch.nn.functional.max_pool2d(M_edge, kernel_size=size, stride=1, padding=pad))
#         N_edge = abs(torch.nn.functional.max_pool2d(N_edge, kernel_size=size, stride=1, padding=pad))
#     edge = M_edge + N_edge - 1
#     edge = abs(torch.nn.functional.avg_pool2d(edge, kernel_size=size, stride=1, padding=pad))
#     weit  = 1+2.5*edge

#     wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
#     pred  = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou  = 1-(inter+1)/(union-inter+1)
#     return (wbce+wiou).mean()

# def structure_loss(pred, mask):
#     wbce  = F.binary_cross_entropy_with_logits(pred, mask)


#     pred  = torch.sigmoid(pred)
#     inter = ((pred * mask)).sum(dim=(2, 3))

#     union = ((pred + mask)).sum(dim=(2, 3))
#     wiou  = 1-(inter+1e-6)/(union-inter+1e-6)

#     return (wbce+wiou.mean())

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1e-6)/(union-inter+1e-6)

    return (wbce+wiou).mean()

def iou_loss(pred, mask):
    eps = 1e-6
    inter = (pred * mask).sum(dim=(2, 3)) #交集
    union = (pred + mask).sum(dim=(2, 3)) - inter #并集-交集
    iou = 1 - (inter + eps) / (union + eps)
    return iou.mean()

def dice_loss(pred, mask):
    eps = 1e-6
    N = pred.size()[0]
    pred_flat = pred.view(N,-1)
    mask_flat = mask.view(N,-1)

    intersection = (pred_flat * mask_flat).sum(1)
    dice_coefficient = (2. * intersection + eps) / (pred_flat.sum(1) + mask_flat.sum(1) + eps)
    dice_loss_value = 1 - dice_coefficient.sum()/N
    return dice_loss_value

class LargeK(nn.Module):
    """ LargeK Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim):
        super().__init__()
        self.channel_split = nn.Conv2d(dim,dim*3,kernel_size=1)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=7, dilation=2, padding=6, groups=dim) # depthwise conv
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, dilation=4, padding=12, groups=dim) # depthwise conv
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=7, dilation=8, padding=24, groups=dim) # depthwise conv
        self.channel_mix = nn.Conv2d(dim*3,dim,kernel_size=1)

    def forward(self, x):
        x = self.channel_split(x)
        x1,x2,x3 = torch.chunk(x,3,dim=1)
        x1 = self.dwconv1(x1)
        x2 = self.dwconv2(x2)
        x3 = self.dwconv3(x3)
        x = torch.cat([x1,x2,x3],dim=1)
        x = self.channel_mix(x)
        return x

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    use_multi_head = True
    if q.size() == 3 and k.size() == 3:
        use_multi_head = False
        q, k = q[:,None,...], k[:,None,...]
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos
    if not use_multi_head:
        q, k = q[:,0], k[:,0]
    return q, k

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        # w1(x) -> (batch_size, seq_len, intermediate_size)
        # w3(x) -> (batch_size, seq_len, intermediate_size)
        # w2(*) -> (batch_size, seq_len, hidden_size)
    	return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, data_format="channels_first") -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.data_format = data_format
  
    def _norm(self, hidden_states):
        if self.data_format == "channels_first":
            variance = hidden_states.pow(2).mean(dim=(1), keepdim=True)  # 在高和宽维度上计算均值
        elif self.data_format == "channels_last":
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)
  
    def forward(self, hidden_states):
        if self.data_format == "channels_first":
            return self.weight[..., None, None] * self._norm(hidden_states.float()).type_as(hidden_states)
        elif self.data_format == "channels_last":
            return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes* scale * scale, kernel_size=1, padding = pad)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()
        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x
    
class REsampling(nn.Module):
    def __init__(self, scale):
        super(REsampling, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        N, C, H, W = x.size()
        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x
    
class Dcrop(nn.Module):
    def __init__(self,inplanes,cropscale=2):
        super(Dcrop, self).__init__()
        self.conv = nn.Conv2d(inplanes*cropscale*cropscale, inplanes*cropscale*cropscale, kernel_size=3, padding = 1)
        self.cropscale = cropscale
    
    def forward(self, x):
        B,C,H,W = x.size()
        x_permuted = x.permute(0, 2, 3, 1) 
        x_permuted = x_permuted.contiguous().view((B, H, W//self.cropscale, C*self.cropscale))
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((B, W//self.cropscale, H//self.cropscale, C*self.cropscale*self.cropscale))
        x = x_permuted.permute(0, 3, 2, 1)
        x = self.conv(x)+x
        return x
    
def show_gray_images(images, m=8, alpha=3, cmap='coolwarm',save_path=None):
    if len(images.size()) == 2:
        plt.imshow(images, cmap=cmap)
        plt.axis('off')
    else:
        n, h, w = images.shape
        if n == 1:
            plt.imshow(images[0], cmap=cmap)
            plt.axis('off')
        else:
            if m > n: m = n
            num_rows = (n + m - 1) // m
            fig, axes = plt.subplots(num_rows, m, figsize=(m * 2*alpha, num_rows * 2*alpha))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(num_rows):
                for j in range(m):
                    idx = i*m + j
                    if m == 1 or num_rows == 1:
                        axes[idx].imshow(images[idx], cmap=cmap)
                        axes[idx].axis('off')
                    elif idx < n:
                        axes[i, j].imshow(images[idx], cmap=cmap)
                        axes[i, j].axis('off')
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    


