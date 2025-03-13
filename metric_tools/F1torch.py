import torch

def f1score_torch(pd_raw,gt_raw):

	pd = pd_raw[0]*255
	gt = gt_raw[0]*255
	gtNum = torch.sum((gt>128).float()*1) ## number of ground truth pixels

	pp = pd[gt>128]
	nn = pd[gt<=128]

	pp_hist =torch.histc(pp,bins=255,min=0,max=255)
	nn_hist = torch.histc(nn,bins=255,min=0,max=255)


	pp_hist_flip = torch.flipud(pp_hist)
	nn_hist_flip = torch.flipud(nn_hist)

	pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
	nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

	precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)#torch.divide(pp_hist_flip_cum,torch.sum(torch.sum(pp_hist_flip_cum, nn_hist_flip_cum), 1e-4))
	recall = (pp_hist_flip_cum)/(gtNum + 1e-4)
	f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-8)

	return torch.reshape(precision,(1,precision.shape[0])),torch.reshape(recall,(1,recall.shape[0])),torch.reshape(f1,(1,f1.shape[0]))