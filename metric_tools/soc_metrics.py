import os
import cv2
from tqdm import tqdm
try:
    import metrics as M
except:
    from .metrics import metrics as M
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

_EPS = 1e-16
_TYPE = np.float64

def get_files(path,name='.pkl'):
    file_lan = []
    for filepath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            if name not in os.path.join(filepath,filename):
                file_lan.append(os.path.join(filepath,filename))
    return file_lan

def once_compute(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE):
    gt_path = os.path.join(gt_root, gt_name)
    pred_path = os.path.join(pred_root, gt_name)
    # print(gt_path,pred_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gtsize = gt.shape
    predsize = pred.shape
    if gtsize[0] == predsize[1] and gtsize[1] == predsize[0] and gtsize[0] != gtsize[1]:
        print(pred_path)
    if predsize[0] != gtsize[0] and predsize[1] != gtsize[1]:
        pred = cv2.resize(pred, (gtsize[1], gtsize[0]))
    precisions,recalls = FM.step(pred=pred, gt=gt)
    wfm = WFM.step(pred=pred, gt=gt)
    mae = MAE.step(pred=pred, gt=gt)
    sm = SM.step(pred=pred, gt=gt)
    em = EM.step(pred=pred, gt=gt)
    return {'precisions':precisions,
            'recalls':recalls,
            'wfm':wfm,
            # 'wfm':0,
            'mae':mae,
            # 'mae':0,
            'sm':sm,
            # 'sm':0,
            'em':em,
            # 'em':0,
            }

def once_get(gt_root,pred_root,FM,WFM,SM,EM,MAE,testdir,i,n_jobs):
    gt_name_list = get_files(pred_root)
    gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])
    # print(gt_root,pred_root)
    # for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
    results = Parallel(n_jobs=n_jobs)(delayed(once_compute)(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE) for gt_name in tqdm(gt_name_list, total=len(gt_name_list)))
    precisions,recalls,wfm,sm,em,mae = [],[],[],[],[],[]
    for result in results:
        precisions.append([result['precisions']])
        recalls.append([result['recalls']])
        wfm.append([result['wfm']])
        mae.append([result['mae']]) 
        sm.append([result['sm']])
        em.append([result['em']])
        
    # print(np.array(fm, dtype=_TYPE).shape)
    precisions = np.array(precisions, dtype=_TYPE)
    recalls = np.array(recalls, dtype=_TYPE)
    precision = precisions.mean(axis=0)
    recall = recalls.mean(axis=0)
    fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + _EPS)
    # print(fm.shape)
    wfm = np.mean(np.array(wfm, dtype=_TYPE))
    mae = np.mean(np.array(mae, dtype=_TYPE))
    sm = np.mean(np.array(sm, dtype=_TYPE))
    em = np.mean(np.array(em, dtype=_TYPE), axis=0)
    onefile = pd.DataFrame()
    results = {'maxFm':fmeasure.max(),
        'wFmeasure':wfm,
        'MAE':mae, 
        'Smeasure:':sm, 
        'meanEm':em.mean(),
        }
    results = pd.DataFrame.from_dict([results]).T
    onefile = pd.concat([onefile,results])
    print(
        'testdir:', testdir+'##'+str(i), ', ',
        'maxFm:', fmeasure.max().round(3),'; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'Smeasure:', sm.round(3), '; ',
        'meanEm:', em.mean().round(3), '; ',
        sep=' '
    )
    # onefile.to_csv(args.testdir+str(i)+".csv")
    # allfile = pd.concat([allfile.T,onefile.T]).T
    return onefile

def soc_metrics(testdir):
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

#<<<<<<< HEAD:codes/soc_eval.py

    gt_roots = [
        # r'/home/DATA/HRSOD_test/masks',
        # r'/home/DATA/UHRSD_TE_2K/masks',
        r'/home/DATA/DIS-DATA/DIS-VD/masks',
                # r'/home/DATA/DIS-DATA/DIS-TE1/masks',
                # r'/home/DATA/DIS-DATA/DIS-TE2/masks',
                # r'/home/DATA/DIS-DATA/DIS-TE3/masks',
                # r'/home/DATA/DIS-DATA/DIS-TE4/masks',
                ]
    
    
    n_jobs=12

    cycle_roots = [
        # r"/home/4090-2/MVANET/saved_model/Model_20/DIS_VD",
        # f'/home/your_results/HRSOD-TE/{testdir}/HRSOD_test',
        # f'/home/your_results/UHRSD-TE/{testdir}/UHRSD-TE',
        # f'/home/your_results/{testdir}/HRSOD',
        # f'/home/your_results/{testdir}/UHRSOD',
        f'/home/your_results/{testdir}/DIS-VD',
                #   f'/home/your_results/{testdir}/DIS-TE1',
                #   f'/home/your_results/{testdir}/DIS-TE2',
                #   f'/home/your_results/{testdir}/DIS-TE3',
                #   f'/home/your_results/{testdir}/DIS-TE4',
                  ]

    allfile = pd.DataFrame()
    for i in range(gt_roots.__len__()):
        gt_root = gt_roots[i]
        pred_root = cycle_roots[i]
        onefile = once_get(gt_root,pred_root,FM,WFM,SM,EM,MAE,testdir,i,n_jobs)
        allfile = pd.concat([allfile.T,onefile.T]).T
    allfile.to_csv('/home/your_results/'+testdir+"/ALL.csv")



if __name__ == '__main__':
    # soc_metrics('BiRefNet-UH')
    # soc_metrics('Inspy-UH')
    # soc_metrics('PGNet-UH')
    soc_metrics('HRSOD_VIDP-d2-p8-loss_ablation-baseline2025-03-13_14_40_48')