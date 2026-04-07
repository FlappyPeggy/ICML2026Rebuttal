import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import gc
import torch.multiprocessing as mp
import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import setup_seed, meanstd, NUM_CLASSES, METRIC_OOD, fpr95, get_seed, SEMANTIC_TEMPLATE, NGLABEL, get_clip_fea, demetabatch, get_available_gpus
from utils.utils_1k import PART_LABEL, CLASS2PART_MAPPING, get_text_from_datapath_withref, ALL_LABEL
from utils.datasets import load_data, save_npz
from model.clip_wrapper import CLIPWrapper
from model.model import structural_score, upsample_position_embedding
# optimization is hardware‑related!!!!!!
# please save your work and close other files beforehand to avoid potential data loss!!!!!!
# donot use tor detection performance, computational cost only (execution order may differ, do not mixing them)
# from model.model_fast import structural_score, upsample_position_embedding 
from glob import glob


# from model.clip_seg import CLIPSeg
# # from model.clip_seg_fast import CLIPSegTensor # donot use tor detection performance, computational cost only

# def init_worker(gpu_main, gpu_id):
#     global model, device
#     device = torch.device(f'cuda:{gpu_id}')
#     model = CLIPSeg(device=gpu_id)

# def seg_func(tasks):
#     available, masks, features, label = [], [], [], []
#     for (I, T, t) in (tasks if isinstance(tasks, list) else [tasks]):
#         try:
#             a, m, _, f, l = model(I, T, None if t is None else ALL_LABEL[t])
#             available.append(a), masks.append(m.numpy());features.append(f.numpy());label.append(ALL_LABEL.index(l))
#         except:
#             available.append(False), masks.append(np.zeros((1, 1, 1)));features.append(np.zeros((1, 512)));label.append(-1)
#     return available, masks, features, np.array(label)


@torch.no_grad()
def infer_my_loader(net, loader, text_features, text_part_features, CLASS2PARTMAPPING=None, which_dataset=None, neg_features=None, use_image_fea=False, geo_detector=None, agg_g_score=True):
    dataloader, metaloader = loader
    total = 0
    score_L, score_L_star, score_G, score_G_star = [], [], [], []
    
    # for (_, images, _), (_, all_data, gt) in zip(dataloader, metaloader):
    #     _orifea = net.get_normed_fea(images, cls_only=True)
    for _, all_data, gt in metaloader:
        _orifea = all_data['fea'].to(net.device) @ net.visual.proj
        total += len(gt)
        fea = F.normalize(_orifea, dim=-1, p=2).float()
        pred_G, pred_naive = F.softmax((fea @ text_features.t()).float(), dim=-1)[:, :NUM_CLASSES[which_dataset]].max(1)
        if not agg_g_score:
            g_score = pred_G.cpu().numpy().tolist()
            score_G.extend(g_score)
            score_G_star.extend(g_score)
        
        # part clip
        for segfea, orifea, pred, pred_ in zip(all_data['sfea'], fea.cpu(), all_data['pred'], pred_naive.cpu()):
            part_cls_start, part_cls_end = CLASS2PARTMAPPING[pred_.item() if pred == -1 else pred.item()]
            score_mat = (F.normalize(torch.cat([segfea,orifea.unsqueeze(0)], dim=0).to(net.device), dim=-1, p=2).float() @ torch.cat([text_part_features, neg_features], dim=0).t()).float()
            prob_mat = F.softmax(score_mat[:, :len(text_part_features)], dim=1)[:, part_cls_start:part_cls_end]
            if agg_g_score:
                score_G.append(prob_mat[-1].mean().item())
            if pred == -1:
                score_L.append(prob_mat[-1].mean().item())
            else:
                score_L.append(torch.diagonal(prob_mat[:-2]).mean().item())
        
            prob_mat = F.softmax(score_mat, dim=1)
            prob_mat = prob_mat[:, part_cls_start:part_cls_end] - 0.5*prob_mat[:, len(text_part_features):].amax(1,keepdim=True)    
            if agg_g_score:
                score_G_star.append(prob_mat[-1].mean().item())
            if pred == -1:
                score_L_star.append(prob_mat[-1].mean().item())
            else:
                score_L_star.append(torch.diagonal(prob_mat[:-2]).mean().item())
    base_score = (np.array(score_G)*np.array(score_L)).tolist()
    base_score_star = (np.array(score_G_star)*np.array(score_L_star)).tolist()
    if geo_detector is None:
        return total, base_score, base_score_star
    
    ###################################
    ########  geom detector   #########
    ###################################
    geo_score = []
    goodmask_list = []
    assert dataloader is not None
    for batch_idx, ((_, images, _), (_, all_data, gts)) in enumerate(zip(dataloader, metaloader)):
        goodmask = torch.tensor([(sm.sum()!=0 and sm.numel()>=28**2) for sm in all_data['smask']])
        goodmask_list.append(goodmask)
        goodmask = goodmask.nonzero(as_tuple=True)[0]
        if len(goodmask):
            all_data['fea'] = all_data['fea'][goodmask];all_data['pred'] = all_data['pred'][goodmask];all_data['sood'] = all_data['sood'][goodmask];all_data['smask'] = [all_data['smask'][goodidx] for goodidx in goodmask];all_data['sfea'] = [all_data['sfea'][goodidx] for goodidx in goodmask]
            images = images[goodmask]
            
            clip_fea, clip_attn = get_clip_fea(net, batch=images)
            # geo_score.extend(geo_detector.detect_fast(demetabatch(all_data, gts), clip_fea=clip_fea, clip_attn_squeeze1=clip_attn))
            geo_score.extend(geo_detector.detect(demetabatch(all_data, gts), clip_fea=clip_fea, clip_attn_squeeze1=clip_attn))
    
    goodmask_list = torch.cat(goodmask_list, dim=0).numpy()
    geo_score = np.array(geo_score)
    geo_score_ = np.ones(len(goodmask_list))*geo_score.min()
    geo_score_[goodmask_list] = geo_score
    geo_score_ = geo_score_.tolist()
    
    return total, base_score, base_score_star, geo_score_
    
@torch.no_grad()
def evaluation(net, trainloader, dataloader, ori_text, part_text, which_dataset, ood_sets=None, 
               geo_cfg=None, alpha=0.2):
    net.eval()
    prompted_text_with_part = [[template.format(t) for t in part_text] for template in SEMANTIC_TEMPLATE]
    text_features = F.normalize(net.encode_text(clip.tokenize([f'a photo of a {t}' for t in ori_text]).to(net.device)), dim=1, p=2).float()
    
    num_cls_to_ensem = len(prompted_text_with_part[0])
    prompted_text_with_part = clip.tokenize([x for sub in prompted_text_with_part for x in sub]).to(net.device)
    text_part_features = torch.cat([net.encode_text(chunk)[None].cpu() for chunk in prompted_text_with_part.view(-1, num_cls_to_ensem, prompted_text_with_part.size(-1))], dim=0)
    text_part_features = F.normalize(text_part_features.mean(0), dim=1, p=2).to(net.device).float()

    neg_prompted_text = [[template.format(t) for t in NGLABEL] for template in SEMANTIC_TEMPLATE]
    neg_text_features = net.encode_text(clip.tokenize([x for sub in neg_prompted_text for x in sub]).to(net.device))
    neg_text_features = F.normalize(neg_text_features.view(-1, len(neg_prompted_text[0]), neg_text_features.size(-1)).mean(0), dim=1, p=2).float()
    
    res, fpr, hyp = [], [], []
    midname = f'mergegeomzoom'
    if trainloader[0] is None or os.path.exists(f'./scores/geo_score_id_.npz'):
        geo_score = np.load(f'./scores/geo_score_id_.npz')
        geo_detector=None
        my_score_geo = geo_score['geo'].tolist()
    else:
        geo_detector = structural_score(
            net,
            metaloader=trainloader[1], 
            dataloader=trainloader[0],
            pe_token=upsample_position_embedding(net.visual.positional_embedding, (224 // 16, 224 // 16))[1:],
            which_dataset=which_dataset,
            **geo_cfg)
    
    my_score = infer_my_loader(net, dataloader, text_features, text_part_features, CLASS2PARTMAPPING=CLASS2PART_MAPPING, which_dataset=which_dataset, neg_features=neg_text_features, geo_detector=geo_detector)
    my_score_base = my_score[1]
    my_score_base_star = my_score[2]
    total = my_score[0]
    if geo_detector is not None:
        geo_score = my_score[3]
        np.savez(f'./scores/geo_score_id.npz', geo=np.array(geo_score))        
        
    del dataloader
    gt_list_all = [0] * total
    lenth = [total]
    my_score_base_, my_score_base_star_, my_score_geo_ = [], [], []
    for ood_idx, (ood_dataset, ood_metaset) in enumerate(zip(ood_sets[0], ood_sets[1])):
        lenth.append(lenth[-1])
        my_score_ = infer_my_loader(net, (ood_dataset, ood_metaset),  text_features, text_part_features, CLASS2PARTMAPPING=CLASS2PART_MAPPING, which_dataset=which_dataset, neg_features=neg_text_features, geo_detector=geo_detector)
        my_score_base_.extend(my_score_[1])
        my_score_base_star_.extend(my_score_[2])
        total = my_score_[0]
        if os.path.exists(f'./scores/geo_score_id.npz'):
            geo_score = np.load(f'./scores/geo_score_ood{ood_idx}.npz')
            my_score_geo_.extend(geo_score['geo'].tolist())
        else:
            my_score_geo_.extend(my_score_[3])
            np.savez(f'./scores/geo_score_ood{ood_idx}_.npz', geo=np.array(my_score_[2]))
        
        lenth[-1] += total
        gt_list_all += [1]*total
    del ood_sets
    ########################################################################################################################
    my_score_base_all = meanstd(-np.array(my_score_base+my_score_base_))
    my_score_base_star_all = meanstd(-np.array(my_score_base_star+my_score_base_star_))
    my_score_geo_all = meanstd(-np.array(my_score_geo+my_score_geo_))
    my_score = my_score_base_all + alpha*my_score_geo_all
    my_score_star = my_score_base_star_all + alpha*my_score_geo_all
    
    res = sum([METRIC_OOD(y_true=np.append(gt_list_all[:lenth[0]], gt_list_all[lenth[i]:lenth[i + 1]]),
                y_score=np.append(my_score[:lenth[0]], my_score[lenth[i]:lenth[i + 1]]))*100 for i in range(len(lenth) - 1)])/4
    fpr = sum([fpr95(y_true=np.append(gt_list_all[:lenth[0]], gt_list_all[lenth[i]:lenth[i + 1]]),
                y_score=np.append(my_score[:lenth[0]], my_score[lenth[i]:lenth[i + 1]]))*100 for i in range(len(lenth) - 1)])/4
    res_star = sum([METRIC_OOD(y_true=np.append(gt_list_all[:lenth[0]], gt_list_all[lenth[i]:lenth[i + 1]]),
                y_score=np.append(my_score_star[:lenth[0]], my_score_star[lenth[i]:lenth[i + 1]]))*100 for i in range(len(lenth) - 1)])/4
    fpr_star = sum([fpr95(y_true=np.append(gt_list_all[:lenth[0]], gt_list_all[lenth[i]:lenth[i + 1]]),
                y_score=np.append(my_score_star[:lenth[0]], my_score_star[lenth[i]:lenth[i + 1]]))*100 for i in range(len(lenth) - 1)])/4
    
    gc.collect()
    return res, fpr, res_star, fpr_star


if __name__ == '__main__':
    seed = 2025
    ckpt_path = 'ckpt/'
    batch_size = 7
    objectnet_asid = False
    test_batch_size = 32
    which_dataset = 'imagenet1000'
    alpha = 0.5 # 0.2
    save_by_class = False
    use_meta=True
    GEO_CFG = {
        'coreset_k'          : 1,
        'n_point_per_part'   : 4, 
    }
    
    mp.set_start_method('spawn', force=True)
    gpu_main = 0
    # bs_for_seg_mp = len(get_available_gpus(12))
    # batch_size = bs_for_seg_mp
    # test_batch_size = bs_for_seg_mp
    # gpu_main = get_available_gpus(12)[0]
    torch.cuda.set_device(gpu_main)
    torch.backends.cudnn.benchmark = True
    assert len(PART_LABEL.keys()) == NUM_CLASSES[which_dataset]
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs('scores', exist_ok=True)
    
    if not use_meta:
        setup_seed(seed)
        (_, _, all_set_path), _, train_dataloader_noaug, in_dataloader, ood_dataloader = load_data(which_dataset=which_dataset, data_source='processed', hard_id=objectnet_asid, batch_size=batch_size, test_batch_size=test_batch_size, multi_crop=False)
    setup_seed(seed)
    (_, _, all_set_path), val_metaloader, train_metaloader_noaug, in_metaloader, ood_metaloader = load_data(which_dataset=which_dataset, data_source='feature', hard_id=objectnet_asid, batch_size=batch_size, test_batch_size=test_batch_size)
    
    ori_text, part_text = get_text_from_datapath_withref(PART_LABEL)
    train_dataloader_noaug, in_dataloader, ood_dataloader = None, None, [None, None, None, None]
    # pre_load_loader = all_set_path, ori_text, part_text, train_metaloader_noaug, in_metaloader, ood_metaloader, train_dataloader_noaug, in_dataloader, ood_dataloader, get_seed()
    # setup_seed(seed)
    
    main_device = torch.device(f"cuda:{gpu_main}")
    # all_set_path, ori_text, part_text, train_metaloader_noaug, in_metaloader, ood_metaloader, train_dataloader_noaug, in_dataloader, ood_dataloader, state = pre_load_loader
    # setup_seed(state=state)
    
    net = CLIPWrapper(clip.load("ViT-B/16", device=main_device)[0])
    net.eval()
    print(evaluation(
        net, (train_dataloader_noaug, train_metaloader_noaug), (in_dataloader, in_metaloader), ori_text, part_text,
        which_dataset, (ood_dataloader, ood_metaloader),
        geo_cfg=GEO_CFG, alpha=alpha))
    ##############################
    # gpu_ids = get_available_gpus(12)
    # pool = [mp.Pool(
    #         processes=1,
    #         initializer=init_worker,
    #         initargs=(main_device, gpu)
    #     ) for gpu in gpu_ids]
    # torch.cuda.set_device(gpu_main)
    # for l, n in zip(
    #         [train_dataloader_noaug, in_dataloader, *ood_dataloader, ],
    #         [ 'trainnoaug', 'test', *['ood_' + str(i) for i in range(len(all_set_path['ood']))], ]):
    #     p = all_set_path[n] if len(n.split('_')) == 1 else all_set_path[n.split('_')[0]]
    #     if n.split('_')[0] == 'ood':
    #         p = p[int(n.split('_')[1])]
    #     os.makedirs(p, exist_ok=True)
    #     guid, im_list, fea_list, label_list, block_idx, seg_mask, seg_fea, seg_available, label_pred = None, [], [], [], 0, [], [], [], []

    #     for idx, (ori_im, images, labels) in enumerate(tqdm(l)):
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #         with torch.no_grad():
    #             pre_proj = net.pool_visual(
    #                 net.visual.ln_post(
    #                     net.visual.transformer.resblocks[
    #                         net.visual.transformer.layers - 1
    #                         ](net.visual(images.to(main_device), 224, 224, guid=None)[0])[0].permute(1, 0, 2)
    #                 ),
    #                 use_cls_token=True,
    #                 guid=None).cpu()
            
    #         results = []
    #         tasks = [(img, PART_LABEL, gt) for img, gt in zip(ori_im, labels)] if ('OOD' not in p and 'ID' not in p) else [(img, PART_LABEL, None) for img in ori_im]
    #         tasks_len = (len(tasks)-1) // len(gpu_ids) + 1
    #         [results.append(p.apply_async(seg_func, (tasks[tasks_len * i:min(tasks_len * (i + 1), len(tasks))], ))) for i, p in enumerate(pool)]
                
    #         for res in results:
    #             available_list, masks_list, feature_list, label_pred_list = res.get()
    #             seg_available.extend(available_list);seg_mask.extend(masks_list);seg_fea.extend(feature_list);label_pred.extend(label_pred_list)
    #         del results
            
    #         label_list.extend(labels.numpy().tolist() if 'ood' not in n else [-1]*len(labels))
    #         fea_list.extend(pre_proj.numpy().tolist())

    #         if len(label_list)>1000:
    #             save_npz(p, f"{n.replace(n.split('_')[0]+'_','')+'_' if save_by_class and n.split('_')[0]!='ood' else ''}{block_idx:02d}", fea_list, label_list, label_pred, availableseg=seg_available, maskseg=seg_mask, feaseg=seg_fea)
    #             block_idx += 1
    #             fea_list, label_list, seg_mask, seg_fea, seg_available, label_pred = [], [], [], [], [], []
                
    #     if len(label_list):
    #         save_npz(p, f"{n.replace(n.split('_')[0]+'_','')+'_' if save_by_class and n.split('_')[0]!='ood' else ''}{block_idx:02d}", fea_list, label_list, label_pred, availableseg=seg_available, maskseg=seg_mask, feaseg=seg_fea)
    # for p in pool:
    #     p.close()
    #     p.join()
    # raise False

