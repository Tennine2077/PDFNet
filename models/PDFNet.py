import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from .swin_transformer import SwinB
from .utils import RMSNorm,SwiGLU,\
    structure_loss,show_gray_images,_upsample_,_upsample_like,\
    SSIMLoss,IntegrityPriorLoss,SiLogLoss
from timm.models.layers import trunc_normal_

def make_crs(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), RMSNorm(out_dim), nn.SiLU(inplace=True))

class PDF_depth_decoder(nn.Module):
    def __init__(self, args,raw_ch=3,out_ch=1):
        super(PDF_depth_decoder, self).__init__()
        
        emb_dim = 128
        self.Decoder = nn.ModuleList()
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2,emb_dim*2),make_crs(emb_dim*2,emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2,emb_dim*2),make_crs(emb_dim*2,emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2,emb_dim*2),make_crs(emb_dim*2,emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2,emb_dim*2),make_crs(emb_dim*2,emb_dim)))

        self.shallow = nn.Sequential(nn.Conv2d(raw_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.upsample1 = make_crs(emb_dim,emb_dim)
        self.upsample2 = make_crs(emb_dim,emb_dim)

        self.Bside = nn.ModuleList()
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))

    def forward(self,img,img_feature):

        L1_feature,L2_feature,L3_feature,L4_feature,global_feature = img_feature

        De_L4 = self.Decoder[0](torch.cat([global_feature,L4_feature],dim=1))

        De_L3 = self.Decoder[1](torch.cat([_upsample_like(De_L4,L3_feature),L3_feature],dim=1))

        De_L2 = self.Decoder[2](torch.cat([_upsample_like(De_L3,L2_feature),L2_feature],dim=1))

        De_L1 = self.Decoder[3](torch.cat([_upsample_like(De_L2,L1_feature),L1_feature],dim=1))

        shallow = self.shallow(img)
        final_output = De_L1 + _upsample_like(shallow, De_L1)
        final_output = self.upsample1(_upsample_(final_output,[final_output.shape[-2]*2,final_output.shape[-1]*2]))
        final_output = _upsample_(final_output + _upsample_like(shallow, final_output),[final_output.shape[-2]*2,final_output.shape[-1]*2])
        final_output = self.upsample2(final_output)

        final_output = self.Bside[0](final_output)

        side_1 = self.Bside[1](De_L1)
        side_2 = self.Bside[2](De_L2)
        side_3 = self.Bside[3](De_L3)
        side_4 = self.Bside[4](De_L4)

        return [final_output,side_1,side_2,side_3,side_4]

class CoA(nn.Module):
    def __init__(self, emb_dim=128):
        super(CoA, self).__init__()
        self.Att = nn.MultiheadAttention(emb_dim,1,bias=False,batch_first=True,dropout=0.1)
        self.Norm1 = RMSNorm(emb_dim,data_format='channels_last')
        self.drop1 = nn.Dropout(0.1)
        self.FFN = SwiGLU(emb_dim,emb_dim)
        self.Norm2 = RMSNorm(emb_dim,data_format='channels_last')
        self.drop2 = nn.Dropout(0.1)

    def forward(self,q,kv):
        res = q
        KV_feature = self.Att(q, kv, kv)[0]
        KV_feature = self.Norm1(self.drop1(KV_feature)) + res
        res = KV_feature
        KV_feature = self.FFN(KV_feature)
        KV_feature = self.Norm2(self.drop2(KV_feature)) + res
        return KV_feature


class FSE(nn.Module):
    def __init__(self, img_dim=128, depth_dim=128, patch_dim=128, emb_dim=128, pool_ratio=[1,1,1], patch_ratio=4):
        super(FSE, self).__init__()

        self.patch_ratio = patch_ratio
        self.pool_ratio = pool_ratio
        self.I_channelswich = make_crs(img_dim,emb_dim)
        self.P_channelswich = make_crs(patch_dim,emb_dim)
        self.D_channelswich = make_crs(depth_dim,emb_dim)

        self.IP = CoA(emb_dim)
        self.PI = CoA(emb_dim)

        self.ID = CoA(emb_dim)
        self.DI = CoA(emb_dim)

    @torch.no_grad()
    def split(self, x: torch.Tensor, patch_ratio: int = 8) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        B,C,H,W = x.shape
        patch_stride = H//patch_ratio
        patch_size = H//patch_ratio
        # patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = patch_ratio

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    @torch.no_grad()
    def merge(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]
                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output
    
    def get_boundary(self,pred):
        # return torch.ones_like(pred)
        if pred.shape[-2]//8 % 2 == 0:
            return abs(pred.sigmoid()-F.avg_pool2d(pred.sigmoid(),kernel_size=(pred.shape[-2]//8+1,pred.shape[-1]//8+1),stride=1,padding=(pred.shape[-2]//8//2,pred.shape[-1]//8//2)))
        else:
            return abs(pred.sigmoid()-F.avg_pool2d(pred.sigmoid(),kernel_size=(pred.shape[-2]//8,pred.shape[-1]//8),stride=1,padding=(pred.shape[-2]//8//2,pred.shape[-1]//8//2)))

    def BIS(self,pred):
        if pred.shape[-2]//8 % 2 == 0:
            boundary = 2*self.get_boundary(pred.sigmoid())
            return boundary, F.relu(pred.sigmoid()-5*boundary)
        else:
            boundary = 2*self.get_boundary(pred.sigmoid())
            return boundary, F.relu(pred.sigmoid()-5*boundary)

    def forward(self,img,depth,patch,last_pred):
        boundary,integrity = self.BIS(last_pred)
        img = img * _upsample_like(last_pred.sigmoid(),img)
        depth = depth * _upsample_like(last_pred.sigmoid(),depth)
        patch = patch * _upsample_like(last_pred.sigmoid(),patch)
        pi,pd,pp = self.pool_ratio
        B,C,img_H,img_W = img.size()
        img_cs = self.I_channelswich(img)
        pool_img_cs = F.adaptive_avg_pool2d(img_cs,output_size=[img_H//pi,img_W//pi])
        img_cs = rearrange(img_cs, 'b c h w -> b (h w) c')
        pool_img_cs = rearrange(pool_img_cs, 'b c h w -> b (h w) c')
        B,C,depth_H,depth_W = depth.size()

        #give depth the integrity prior
        integrity = _upsample_like(integrity,depth)
        last_pred_sigmoid = _upsample_like(last_pred,depth).sigmoid()
        enhance_depth = depth*(last_pred_sigmoid + integrity)
        depth_cs = self.D_channelswich(enhance_depth)
        pool_depth_cs = F.adaptive_avg_pool2d(depth_cs,output_size=[depth_H//pd,depth_W//pd])
        pool_depth_cs = rearrange(pool_depth_cs, 'b c h w -> b (h w) c')
        B,C,patch_H,patch_W = patch.size()

        #select the boundary patches to select patches
        patch_batch = self.split(patch,patch_ratio=self.patch_ratio)
        boundary_batch = self.split(boundary,patch_ratio=self.patch_ratio)
        boundary_score = boundary_batch.mean(dim=[2,3])[...,None,None]
        select_patch = patch_batch * (1+5*boundary_score)
        select_patch = self.merge(select_patch,batch_size=B)

        patch_cs = self.P_channelswich(select_patch)
        pool_patch_cs = F.adaptive_avg_pool2d(patch_cs,output_size=[patch_H//pp,patch_W//pp])
        pool_patch_cs = rearrange(pool_patch_cs, 'b c h w -> b (h w) c')
        
        patch_feature = self.PI(pool_patch_cs, torch.cat([pool_img_cs,pool_depth_cs],dim=1))
        img_feature = self.IP(img_cs,patch_feature)

        depth_feature = self.DI(pool_depth_cs, torch.cat([pool_img_cs,pool_patch_cs],dim=1))
        img_feature = self.ID(img_feature,depth_feature)

        patch_feature = rearrange(patch_feature, 'b (h w) c -> b c h w',h=patch_H//pp)
        depth_feature = rearrange(depth_feature, 'b (h w) c -> b c h w',h=depth_H//pd)
        img_feature = rearrange(img_feature, 'b (h w) c -> b c h w',h=img_H)

        depth_feature = _upsample_like(depth_feature,depth)
        patch_feature = _upsample_like(patch_feature,patch)

        return img_feature + rearrange(img_cs, 'b (h w) c -> b c h w',h=img_H), depth_feature + depth_cs, patch_feature + patch_cs


class PDF_decoder(nn.Module):
    def __init__(self, args,raw_ch=3,out_ch=1):
        super(PDF_decoder, self).__init__()
        self.args = args
        emb_dim = args.emb
        self.patch_ratio = 8

        self.FSE_mix = nn.ModuleList()
        self.FSE_mix.append(FSE(emb_dim*2,emb_dim*2,emb_dim*2,
                                         emb_dim,pool_ratio=[1,1,1],patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2,emb_dim*2,emb_dim*2,
                                         emb_dim,pool_ratio=[1,1,1],patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2,emb_dim*2,emb_dim*2,
                                         emb_dim,pool_ratio=[2,2,2],patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2,emb_dim*2,emb_dim*2,
                                         emb_dim,pool_ratio=[2,2,2],patch_ratio=self.patch_ratio))

        self.shallow = nn.Sequential(nn.Conv2d(raw_ch*2, emb_dim, kernel_size=4, stride=4),make_crs(emb_dim,emb_dim))
        self.upsample1 = nn.Sequential(make_crs(emb_dim,emb_dim))
        self.upsample2 = nn.Sequential(make_crs(emb_dim,emb_dim))

        self.channel_mix = nn.ModuleList()
        self.channel_mix.append(make_crs(emb_dim*3,emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3,emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3,emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3,emb_dim))

        self.Bside = nn.ModuleList()
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))
        self.Bside.append(nn.Conv2d(emb_dim,out_ch,3,padding=1))

    def forward(self,img,depth,img_feature,depth_feature,patch_img_feature):
        B,C,H,W = img.size()
        side_5 = self.Bside[5](_upsample_like(img_feature[4],patch_img_feature[4]) + _upsample_like(depth_feature[4],patch_img_feature[4]) + patch_img_feature[4])

        img_L4,depth_L4,patch_L4 = self.FSE_mix[0](torch.cat([img_feature[4],img_feature[3]],dim=1),
                                torch.cat([depth_feature[4],depth_feature[3]],dim=1),
                                torch.cat([patch_img_feature[4],patch_img_feature[3]],dim=1),side_5)
        mix_L4 = self.channel_mix[3](torch.cat([_upsample_like(img_L4,patch_L4),_upsample_like(depth_L4,patch_L4),
                                                patch_L4],dim=1))
        side_4 = self.Bside[4](mix_L4)
        img_L3,depth_L3,patch_L3 = self.FSE_mix[1](torch.cat([_upsample_like(img_L4,img_feature[2]),img_feature[2]],dim=1),
                                torch.cat([_upsample_like(depth_L4,depth_feature[2]),depth_feature[2]],dim=1),
                                torch.cat([_upsample_like(patch_L4,patch_img_feature[2]),patch_img_feature[2]],dim=1),side_4)
        mix_L3 = self.channel_mix[2](torch.cat([_upsample_like(img_L3,patch_L3),_upsample_like(depth_L3,patch_L3),
                                                patch_L3],dim=1))
        side_3 = self.Bside[3](mix_L3)
        img_L2,depth_L2,patch_L2 = self.FSE_mix[2](torch.cat([_upsample_like(img_L3,img_feature[1]),img_feature[1]],dim=1),
                                torch.cat([_upsample_like(depth_L3,depth_feature[1]),depth_feature[1]],dim=1),
                                torch.cat([_upsample_like(patch_L3,patch_img_feature[1]),patch_img_feature[1]],dim=1),side_3)
        mix_L2 = self.channel_mix[1](torch.cat([_upsample_like(img_L2,patch_L2),_upsample_like(depth_L2,patch_L2),
                                                patch_L2],dim=1))
        side_2 = self.Bside[2](mix_L2)
        img_L1,depth_L1,patch_L1 = self.FSE_mix[3](torch.cat([_upsample_like(img_L2,img_feature[0]),img_feature[0]],dim=1),
                                torch.cat([_upsample_like(depth_L2,depth_feature[0]),depth_feature[0]],dim=1),
                                torch.cat([_upsample_like(patch_L2,patch_img_feature[0]),patch_img_feature[0]],dim=1),side_2)
        mix_L1 = self.channel_mix[0](torch.cat([_upsample_like(img_L1,patch_L1),_upsample_like(depth_L1,patch_L1),
                                                patch_L1],dim=1))
        side_1 = self.Bside[1](mix_L1)

        shallow = self.shallow(_upsample_(torch.cat([img,depth],dim=1),[H*4,W*4]))
        final_output = _upsample_(mix_L1,[mix_L1.shape[-2]*2,mix_L1.shape[-1]*2]) + _upsample_(shallow,[mix_L1.shape[-2]*2,mix_L1.shape[-1]*2])
        final_output = self.upsample1(final_output)
        final_output =  _upsample_(final_output,[final_output.shape[-2]*2,final_output.shape[-1]*2]) + shallow
        final_output = self.upsample2(final_output)

        final_output = self.Bside[0](final_output)

        return [final_output,side_1,side_2,side_3,side_4,side_5]

class PDFNet_process(nn.Module):
    def __init__(self, encoder, decoder, depth_decoder, device, args):
        super().__init__()
        self.patch_ratio = 8
        self.device = device
        self.raw_ch = 3
        emb = args.emb
        self.Glob = nn.Sequential(make_crs(emb,emb))
        self.decoder = decoder
        self.depth_decoder = depth_decoder
        self.decoder.patch_ratio = self.patch_ratio
        self.args=args

        self.channel_mix = make_crs(emb*4,emb)
        self.channel_mix4 = make_crs(args.back_bone_channels_stage4,emb)
        self.channel_mix3 = make_crs(args.back_bone_channels_stage3,emb)
        self.channel_mix2 = make_crs(args.back_bone_channels_stage2,emb)
        self.channel_mix1 = make_crs(args.back_bone_channels_stage1,emb)

        self.apply(self._init_weights)

        self.encoder = encoder
        self.SSIMLoss = SSIMLoss()
        self.SiLogLoss = SiLogLoss().to(device)
        self.IntegrityPriorLoss = IntegrityPriorLoss().to(device)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def loss_compute(self,Pred,GT):
        loss = 0
        for i in range(len(Pred)):
            if Pred[i].shape[2:] != GT.shape[2:]:
                up_pred = F.interpolate(Pred[i],size=GT.shape[2:],mode='bilinear')
            else:
                up_pred = Pred[i]
            if i == 0:
                target_loss = structure_loss(up_pred,GT) + self.SSIMLoss(up_pred.sigmoid(),GT) * 0.5
                loss = loss + target_loss
            else:
                loss = loss + (structure_loss(up_pred,GT) + self.SSIMLoss(up_pred.sigmoid(),GT) * 0.5) * 0.5
        return loss, target_loss
    
    def Integrity_Loss(self,Pred,depth,gt):
        loss = 0
        for i in range(len(Pred)):
            if Pred[i].shape[2:] != depth.shape[2:]:
                up_pred = F.interpolate(Pred[i],size=depth.shape[2:],mode='bilinear')
            else:
                up_pred = Pred[i]
            if i == 0:
                target_loss = self.IntegrityPriorLoss(up_pred.sigmoid(),depth,gt)
                loss = loss + target_loss
            else:
                loss = loss + (self.IntegrityPriorLoss(up_pred.sigmoid(),depth,gt)) * 0.5
        return loss, target_loss

    def depth_loss(self,Pred,GT):
        loss = 0
        for i in range(len(Pred)):
            if Pred[i].shape[2:] != GT.shape[2:]:
                up_pred = F.interpolate(Pred[i],size=GT.shape[2:],mode='bilinear')
            else:
                up_pred = Pred[i]
            if i == 0:
                target_loss = self.SiLogLoss(up_pred.sigmoid(),GT)
                loss = loss + target_loss
            else:
                loss = loss + (self.SiLogLoss(up_pred.sigmoid(),GT)) * 0.5
        return loss, target_loss

    def encode(self,x,encoder):
        latent_I1,latent_I2,latent_I3,latent_I4 = encoder(x)

        latent_I1 = self.channel_mix1(latent_I1)
        latent_I2 = self.channel_mix2(latent_I2)
        latent_I3 = self.channel_mix3(latent_I3)
        latent_I4 = self.channel_mix4(latent_I4)
        x_glob = self.Glob(self.channel_mix(torch.cat([_upsample_like(latent_I1,latent_I4),
                                                       _upsample_like(latent_I2,latent_I4),
                                                       _upsample_like(latent_I3,latent_I4),
                                                       latent_I4],dim=1)))

        return latent_I1,latent_I2,latent_I3,latent_I4,x_glob

    @torch.no_grad()
    def split(self, x: torch.Tensor, patch_size: int = 256, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    @torch.no_grad()
    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if padding > 0:
                    if j != 0:
                        output = output[..., padding:, :]
                    if i != 0:
                        output = output[..., :, padding:]
                    if j != steps - 1:
                        output = output[..., :-padding, :]
                    if i != steps - 1:
                        output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def forward(self,img,depth,gt,depth_gt):
        depth = (depth-depth.min())/(depth.max()-depth.min())
        depth_gt = (depth_gt-depth_gt.min())/(depth_gt.max()-depth_gt.min())
        B,C,H,W = img.size()
        RIMG,RDEPTH,RGT = img, depth, gt
        if RDEPTH.shape[1] == 1:
            RDEPTH = RDEPTH.repeat(1,3,1,1)
        down_ratio = 2
        patch_ratio = self.patch_ratio
        Down_RIMG = _upsample_(RIMG,[RIMG.shape[-2]//down_ratio,RIMG.shape[-1]//down_ratio])
        Down_RDEPTH = _upsample_(RDEPTH,[RDEPTH.shape[-2]//down_ratio,RDEPTH.shape[-1]//down_ratio])
        Down_img_depth = torch.cat([Down_RIMG,Down_RDEPTH],dim=0)

        latent_I1,latent_I2,latent_I3,latent_I4,x_glob = self.encode(Down_img_depth,self.encoder)
        Depth_latent_I1,Depth_latent_I2,Depth_latent_I3,Depth_latent_I4,Depth_x_glob = latent_I1[B:2*B],latent_I2[B:2*B],latent_I3[B:2*B],latent_I4[B:2*B],x_glob[B:2*B]
        latent_I1,latent_I2,latent_I3,latent_I4,x_glob = latent_I1[:B],latent_I2[:B],latent_I3[:B],latent_I4[:B],x_glob[:B]
        
        patch_img = self.split(RIMG,patch_size=RIMG.shape[-2]//patch_ratio,overlap_ratio=0.)
        patch_latent_I1,patch_latent_I2,patch_latent_I3,patch_latent_I4,patch_x_glob = self.encode(patch_img,self.encoder)
        
        patch_latent_I1 = self.merge(patch_latent_I1,batch_size=B,padding=0)
        patch_latent_I2 = self.merge(patch_latent_I2,batch_size=B,padding=0)
        patch_latent_I3 = self.merge(patch_latent_I3,batch_size=B,padding=0)
        patch_latent_I4 = self.merge(patch_latent_I4,batch_size=B,padding=0)
        patch_x_glob = self.merge(patch_x_glob,batch_size=B,padding=0)

        pred_m = self.decoder(RIMG,RDEPTH,
                            [latent_I1,latent_I2,latent_I3,latent_I4,x_glob],
                            [Depth_latent_I1,Depth_latent_I2,Depth_latent_I3,Depth_latent_I4,Depth_x_glob],
                            [patch_latent_I1,patch_latent_I2,patch_latent_I3,patch_latent_I4,patch_x_glob])
        
        pred_depth = self.depth_decoder(RIMG,[latent_I1+Depth_latent_I1+_upsample_like(patch_latent_I1,latent_I1),
                                              latent_I2+Depth_latent_I2+_upsample_like(patch_latent_I2,latent_I2),
                                              latent_I3+Depth_latent_I3+_upsample_like(patch_latent_I3,latent_I3),
                                              latent_I4+Depth_latent_I4+_upsample_like(patch_latent_I4,latent_I4),
                                              x_glob+Depth_x_glob+_upsample_like(patch_x_glob,x_glob)])

        loss, target_loss = self.loss_compute(pred_m,RGT)
        integrity_loss,_ = self.Integrity_Loss(pred_m,depth_gt,RGT)
        depth_loss,_ = self.depth_loss(pred_depth,depth_gt)

        loss = loss + integrity_loss/2 + depth_loss/10

        if self.args.DEBUG:
            print(pred_m[0].shape)
            H,W = RIMG.shape[-2],RIMG.shape[-1]
            Show_X = torch.cat([RIMG.reshape([-1,H,W])[:3].cpu().detach(),
                                RDEPTH.reshape([-1,H,W])[:1].cpu().detach(),
                                RGT.reshape([-1,H,W])[:1].cpu().detach(),
                                pred_m[0].sigmoid().reshape([-1,H,W])[:1].cpu().detach(),
                                _upsample_like(pred_depth[0],pred_m[0]).sigmoid().reshape([-1,H,W])[:1].cpu().detach(),],dim=0)
            show_gray_images(Show_X,m=RIMG.shape[0]*4,alpha=1.5,cmap='gray')
        return [i.sigmoid() for i in pred_m], loss, target_loss
    
    @torch.no_grad()
    def inference(self,img,depth):
        depth = (depth-depth.min())/(depth.max()-depth.min())
        B,C,H,W = img.size()
        RIMG,RDEPTH = img, depth
        if RDEPTH.shape[1] == 1:
            RDEPTH = RDEPTH.repeat(1,3,1,1)
        down_ratio = 2
        patch_ratio = self.patch_ratio
        Down_RIMG = _upsample_(RIMG,[RIMG.shape[-2]//down_ratio,RIMG.shape[-1]//down_ratio])
        Down_RDEPTH = _upsample_(RDEPTH,[RDEPTH.shape[-2]//down_ratio,RDEPTH.shape[-1]//down_ratio])
        Down_img_depth = torch.cat([Down_RIMG,Down_RDEPTH],dim=0)

        latent_I1,latent_I2,latent_I3,latent_I4,x_glob = self.encode(Down_img_depth,self.encoder)
        Depth_latent_I1,Depth_latent_I2,Depth_latent_I3,Depth_latent_I4,Depth_x_glob = latent_I1[B:2*B],latent_I2[B:2*B],latent_I3[B:2*B],latent_I4[B:2*B],x_glob[B:2*B]
        latent_I1,latent_I2,latent_I3,latent_I4,x_glob = latent_I1[:B],latent_I2[:B],latent_I3[:B],latent_I4[:B],x_glob[:B]
        
        patch_img = self.split(RIMG,patch_size=RIMG.shape[-2]//patch_ratio,overlap_ratio=0.)
        patch_latent_I1,patch_latent_I2,patch_latent_I3,patch_latent_I4,patch_x_glob = self.encode(patch_img,self.encoder)
        
        patch_latent_I1 = self.merge(patch_latent_I1,batch_size=B,padding=0)
        patch_latent_I2 = self.merge(patch_latent_I2,batch_size=B,padding=0)
        patch_latent_I3 = self.merge(patch_latent_I3,batch_size=B,padding=0)
        patch_latent_I4 = self.merge(patch_latent_I4,batch_size=B,padding=0)
        patch_x_glob = self.merge(patch_x_glob,batch_size=B,padding=0)

        pred_m = self.decoder(RIMG,RDEPTH,
                            [latent_I1,latent_I2,latent_I3,latent_I4,x_glob],
                            [Depth_latent_I1,Depth_latent_I2,Depth_latent_I3,Depth_latent_I4,Depth_x_glob],
                            [patch_latent_I1,patch_latent_I2,patch_latent_I3,patch_latent_I4,patch_x_glob])
        
        return pred_m[0].sigmoid(),pred_m[0]

def build_model(args):
    if args.back_bone == 'PDFNet_swinB':
        return PDFNet_process(encoder=SwinB(args=args,in_chans=3,pretrained=True),
                           decoder=PDF_decoder(args=args),depth_decoder=PDF_depth_decoder(args=args),
                           device=args.device, args=args),args.model