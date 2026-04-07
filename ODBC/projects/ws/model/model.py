import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import EPS, INF, fps_sampling, pairwise_cosine_similarity, hungarian_match, transform_points_affine, fit_ransac, STD, MEAN, idx_list, NUM_CLASSES
from utils.utils_1k import PART_LABEL
from tqdm import tqdm
from multiprocessing import Pool
import os
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from concurrent.futures import ThreadPoolExecutor

_CONTEXT_LENGTH = 77
MAX_PARTS = max([len(v) for v in PART_LABEL.values()])

def upsample_position_embedding(embed, new_size):
  first = embed[:1, :]
  embed = embed[1:, :]
  n = embed.size(0)
  d = embed.size(1)
  size = int(np.sqrt(n))
  if size * size != n:
    raise ValueError(f'The size of embed {n} is not a perfect square number.')
  embed = embed.permute(1, 0)
  embed = embed.view(1, d, size, size).contiguous()
  embed =  F.interpolate(
      embed,
      size=new_size,
      mode='bilinear',
  )
  embed = embed.view(d, -1).contiguous()
  embed = embed.permute(1, 0)
  embed = torch.cat([first, embed], 0)
  embed = nn.parameter.Parameter(embed.half())
  return embed


class CustomBlock(nn.Module):
  def __init__(self, block):
    super().__init__()
    for k, v in vars(block).items():
      setattr(self, k, v)

  def attention(self, x):
    self.attn_mask = (
        self.attn_mask.to(dtype=x.dtype, device=x.device)
        if self.attn_mask is not None
        else None
    )
    self.attn = self.attn.to(dtype=x.dtype, device=x.device)
    # Setting need_weights to True also returns the attention weights
    return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
    
  def forward(self, x):
    attn_output, attn_weight = self.attention(self.ln_1(x))    
    x = x + attn_output
    x = x + self.mlp(self.ln_2(x))
    return x, attn_weight

def detect_single(data, coord, queries_masks, clip_fea, query_labels, ):
  clip_fea = clip_fea[:, 1:]
  HW = int(clip_fea.size(1)**0.5)
  assert HW**2==clip_fea.size(1)
  out_list = []
  for q_mask, q_feat, label in zip(queries_masks, clip_fea, query_labels):
    q_valid_parts = torch.nonzero(q_mask.sum(dim=1) > 0, as_tuple=False).squeeze(1)
    if q_valid_parts.numel() == 0 or label == -1 or label not in data:
      out_list.append(0.)
      continue

    best_geom, best_sc = 0., 0.
    
    cand_masks_list  = data[label][0]
    cand_feats_list  = data[label][1]
    cand_part_num    = cand_masks_list[0].shape[0]

    
    for c_mask, c_feat in zip(cand_masks_list, cand_feats_list):
      matched_q_pts, matched_c_pts, matched_s, matched_label, per_part_allpair_means, per_part_allpair_sc, per_part_allpair_pos = [], [], [], [], [], [], [[], []]
      c_valid_parts = torch.nonzero(c_mask.sum(dim=1) > 0, as_tuple=False).squeeze(1)

      
      for p in range(cand_part_num):
        if p not in q_valid_parts or p not in c_valid_parts:
          continue
        q_tok_idx = torch.nonzero(q_mask[p] > 0, as_tuple=False).squeeze(1)
        c_tok_idx = torch.nonzero(c_mask[p] > 0, as_tuple=False).squeeze(1)
        qF = q_feat[q_tok_idx]
        cF = c_feat[c_tok_idx]
        qC = coord[q_tok_idx]
        cC = coord[c_tok_idx]

        sim = pairwise_cosine_similarity(qF, cF)
        r, c = hungarian_match(sim)
        per_part_allpair_sc.append(sim);per_part_allpair_pos[0].append(qC);per_part_allpair_pos[1].append(cC)
        if r.numel() == 0:
          continue

        matched_q_pts.append(qC[r])
        matched_c_pts.append(cC[c])
        matched_label.extend([p]*len(r))
        matched_s.append(sim[r, c])

      if len(matched_label) < 3:
        continue

      pts_src = torch.cat(matched_q_pts, dim=0)
      pts_dst = torch.cat(matched_c_pts, dim=0)
      pts_label = torch.tensor(matched_label).long()  
      s = torch.cat(matched_s, dim=0)  
      
      Trans_M, inmask, residuals = fit_ransac(pts_src, pts_dst, pts_label)
      if Trans_M is None or inmask.sum() == 0:
        continue
      
      for sc_all_mat, pos_all_mat_q, pos_all_mat_c in zip(per_part_allpair_sc, per_part_allpair_pos[0], per_part_allpair_pos[1]):
        per_part_allpair_means.append(float((sc_all_mat*(-(transform_points_affine(Trans_M, pos_all_mat_q)[:, None] - pos_all_mat_c[None]).pow(2).sum(-1)/784).exp()).mean().item()))
        
      geom_means_per_part, sc_means_per_part, offset = [], [], 0
      for p in c_valid_parts:
        if p not in q_valid_parts:
          continue
        nq = int((q_mask[p] > 0).sum().item())
        nc = int((c_mask[p] > 0).sum().item())
        blk = min(nq, nc)
        if blk == 0: 
          continue
        sl = slice(offset, offset+blk)
        offset += blk
        
        geom_res_ = -residuals[sl].float().pow(2)
        geom_means_per_part.append(float((geom_res_*0.5).exp().mean().item()))
        sc_means_per_part.append(float((s[sl].float()*(geom_res_/784).exp()).mean().item()))
      
      if len(geom_means_per_part) == 0:
        continue
      
      geo_val = sum(geom_means_per_part)/len(q_valid_parts)
      if geo_val > best_geom:
        all_part_allpair_means = (np.array(per_part_allpair_means).sum()/len(q_valid_parts)).tolist() if len(per_part_allpair_means)>0 else 0.
        best_sc = (sum(sc_means_per_part)/len(q_valid_parts) if len(sc_means_per_part) > 0 else 0.)- all_part_allpair_means
        best_geom = geo_val

    out_list.append(best_sc)

  return out_list
  

class structural_score:
  def __init__(self, net, metaloader, dataloader, pe_token, coreset_k=-1, n_point_per_part=4,  patch_size=16, net_nms=None, use_nms=[False, False], which_dataset='', let_it_go=False, **kwargs):
    self.coreset_k                          = coreset_k
    self.n_point_per_part                   = n_point_per_part
    self.patch_size                         = patch_size
    self.pe_token                           = pe_token.cpu()
    self.net                                = net
    self.net_nms                            = net_nms
    self.which_dataset                      = which_dataset
    self.let_it_go                          = let_it_go
    
    self.coord = None
    self.transform = [transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),]), transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN['clip'], std=STD['clip'])])]
    self.data, fea_list, attn_list = self._todatalist(metaloader, dataloader, datasetname=which_dataset)
    
    import pickle
    if False and os.path.exists('/data//temp1000_ccs_data.pkl'):
      with open('/data//temp1000_ccs_data.pkl', 'rb') as f:
        self.data = pickle.load(f)
    else:
      self.data = self.sapmling_feature_mask(self.data, clip_feas=fea_list, clip_attn_squeeze1=attn_list, training=True)
      # try:
      #   with open('/data//temp1000_ccs_data.pkl', 'wb') as f:
      #     pickle.dump(self.data, f)
      # except:
    torch.cuda.empty_cache()
   
  def _todatalist(self, metaloader, dataloader, datasetname=''): 
    data, imgs = [], []
    feas, attns = [], []
    labels, masks, crops, goods, mask_ = [], [], [], [], []
    for _, all_data, gts in metaloader:#metaloader
      for pred, smask, sfea, gt in zip(all_data['pred'], all_data['smask'], all_data['sfea'], gts):
        mask = smask.amax(0)
        good = mask.mean()>0.1 or (gt in idx_list and '1000' in self.which_dataset)
        goods.append(good)
        if good:
          if gt in idx_list and '1000' in self.which_dataset:
            crop = (slice(None, None), slice(0, 224), slice(0, 224))
            mask = smask.view(-1, 224, 224).to(smask.dtype)
          else:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            crop = (slice(None, None), slice(ys.min().item(), ys.max().item() + 1), slice(xs.min().item(), xs.max().item() + 1))
            mask = F.interpolate(smask[crop].unsqueeze(1).float(), size=(224, 224), mode="nearest").to(smask.dtype).squeeze(1)
          crops.append(crop)
          masks.append(mask.amax(0))
          data.append({'gt':str(gt.item()),
                    'pred':str(pred.item()),
                    'smask':mask,
                    'sfea':sfea})
        else:
          crops.append(None);masks.append(None)
        
      labels.extend(gts.numpy().tolist())
    labels_all = torch.tensor(labels)
    labels = labels_all[torch.tensor(goods)]
    all_label = torch.unique(labels)
    goods = torch.tensor(goods)
    
    cnt = 0
    for oribatch, stdbatch, _ in dataloader:
      for ori_img in oribatch:
        if goods[cnt]:
          if (labels_all[cnt] in idx_list and '1000' in self.which_dataset):
            prompted_image = np.array(self.transform[0](ori_img)).astype(np.uint8)
            imgs.append(self.transform[1](Image.fromarray(prompted_image)).to(self.net.dtype))
          else:
            c = crops[cnt]
            prompted_image = cv2.resize(np.array(self.transform[0](ori_img))[c[1], c[2]].astype(np.uint8), (224, 224), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(prompted_image.copy(), (15,15), 0)
            mask = np.where(masks[cnt].numpy()>0.5, 1., 0.)[:, :, None]
            imgs.append(self.transform[1](Image.fromarray((prompted_image * mask+blurred * (1 - mask)).astype(np.uint8))).to(self.net.dtype))
        cnt+=1
        
    imgs = torch.stack(imgs, dim=0)
    for imgs_batch in imgs.split(32, dim=0):
      tok, attn = self.net.visual.transformer.resblocks[self.net.visual.transformer.layers - 1](self.net.visual(imgs_batch.to(self.net.device), 224, 224, guid=None)[0])
      attns.append(attn.mean(1)[:, 1:].cpu())
      feas.append(self.net.visual.ln_post(tok.permute(1, 0, 2)).cpu())
    del imgs
    feas = torch.cat(feas, dim=0)
    attns = torch.cat(attns, dim=0)
    
    if self.coreset_k<100:
      self.coreset_k = round(self.coreset_k)*2e6//100000*NUM_CLASSES[self.which_dataset]
    
    if 0<self.coreset_k < len(data):
      if os.path.exists(f'./coreset{datasetname}.npz'):
        coreset_idx = np.load(f'./coreset{datasetname}.npz')['coreset_idx']
        if len(coreset_idx) != self.coreset_k:
          coreset_idx = None
      else:
        coreset_idx = None
      if coreset_idx is None:
        label_per_class = self.coreset_k//len(all_label)
        assert  self.coreset_k==label_per_class*len(all_label)
        coreset_idx = [fps_sampling(feas[:, 0], k=label_per_class, mask=labels==cur_label) for cur_label in all_label]
        coreset_idx = torch.cat(coreset_idx, dim=0)
        np.savez(f'./coreset{datasetname}.npz', coreset_idx=coreset_idx.numpy())
      data = [data[idx] for idx in coreset_idx]
      feas = [feas[idx] for idx in coreset_idx]
      attns = [attns[idx] for idx in coreset_idx]
    return data, feas, attns
            
  def _get_coord(self, N=196):
    if self.coord is None:
      HW = int(N**0.5)
      assert HW**2==N
      HW = torch.arange(HW, dtype=torch.float16)
      gx, gy = torch.meshgrid(HW, HW, indexing='xy') 
      self.coord = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    
  def sapmling_feature_mask(self, data_list, clip_feas=None, clip_attn_squeeze1=None, training=False):
    if not isinstance(data_list, list):
      data_list = [data_list]
      
    self._get_coord()
    extracted_feature = {} if training else []
    for data, clip_fea, attn in zip(data_list, clip_feas, clip_attn_squeeze1):
      patch_token, class_token = clip_fea[1:], clip_fea[:1]
      
      if training and data['gt'] not in extracted_feature:
        extracted_feature[data['gt']] = [[], [], ]
      
      mask_part2token = F.avg_pool2d(data['smask'][:-1].unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size).squeeze(1).view(-1, 196)>0.1 # any value>0
      sampling_number = mask_part2token.sum(-1).clamp_max(self.n_point_per_part).to(torch.long) 
      sampling_mask = torch.zeros_like(mask_part2token, dtype=torch.bool) 

      rank = pairwise_cosine_similarity(data['sfea'][:-1], patch_token@self.net.visual.proj.cpu()).masked_fill(~mask_part2token, -INF)  # (P,C) X (N,C).T ->(P,N)
      
      for p, (k, top1) in enumerate(zip(sampling_number, rank.argmax(dim=-1))):
        if k <= 0:
          continue
        selected_idx = np.argpartition(-rank, k)[:k]
        if selected_idx.numel() > 0:
            sampling_mask[p, selected_idx] = True
      
      if training:
        if (sampling_mask.sum(dim=1) > 0).sum()>1:
          extracted_feature[data['gt']][0].append(sampling_mask)
          extracted_feature[data['gt']][1].append(patch_token)
      else:
        extracted_feature.append(sampling_mask)
    # the part below could be skip
    if training: # and not self.let_it_go:
      for data, clip_fea, attn in zip(data_list, clip_feas, clip_attn_squeeze1):
        if len(extracted_feature[data['gt']][0]): 
          continue
        
        patch_token, class_token = clip_fea[1:], clip_fea[:1]
        mask_part2token = F.avg_pool2d(data['smask'][:-1].unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size).squeeze(1).view(-1, 196)>0 #[P, N]
        sampling_number = mask_part2token.sum(-1).clamp_max(self.n_point_per_part).to(torch.long)  # (P,)
        sampling_mask = torch.zeros_like(mask_part2token, dtype=torch.bool)  # (P,N)

        rank = pairwise_cosine_similarity(data['sfea'][:-1], patch_token@self.net.visual.proj.cpu()).masked_fill(~mask_part2token, -INF)
        for p, (k, top1) in enumerate(zip(sampling_number, rank.argmax(dim=-1))):
          if k <= 0:
            continue
          selected_idx = np.argpartition(-rank, k)[:k]
          if selected_idx.numel() > 0:
              sampling_mask[p, selected_idx] = True

        if (sampling_mask.sum(dim=1) > 0).sum():
          extracted_feature[data['gt']][0].append(sampling_mask)
          extracted_feature[data['gt']][1].append(patch_token) # 
          
      for data, clip_fea, attn in zip(data_list, clip_feas, clip_attn_squeeze1):
        if len(extracted_feature[data['gt']][0]): 
          continue
        
        patch_token, class_token = clip_fea[1:], clip_fea[:1]
        mask_part2token = torch.ones((max(1, data['smask'].size(0)-1), 196), dtype=bool, device=data['smask'].device) #[P, N]
        sampling_number = mask_part2token.sum(-1).clamp_max(self.n_point_per_part).to(torch.long)  # (P,)
        sampling_mask = torch.zeros_like(mask_part2token, dtype=torch.bool)  # (P,N)

        rank = pairwise_cosine_similarity(data['sfea'][:-1] if len(data['sfea'])>1 else data['sfea'], patch_token@self.net.visual.proj.cpu()).masked_fill(~mask_part2token, -INF)
        
        for p, (k, top1) in enumerate(zip(sampling_number, rank.argmax(dim=-1))):
          assert k>0
          selected_idx = np.argpartition(-rank, k)[:k]
          if selected_idx.numel() > 0:
            sampling_mask[p, selected_idx] = True

        if (sampling_mask.sum(dim=1) > 0).sum():
          extracted_feature[data['gt']][0].append(sampling_mask)
          extracted_feature[data['gt']][1].append(patch_token)
          
      which_gt_no_ref = [k for k,v in extracted_feature.items() if not len(v[0])]
      if len(which_gt_no_ref):
        raise RuntimeError('check ref for label(s)! when using provided dataset, youshould not receive this error, please check your datah and path. for other dataset, add the miss class to idx_list can skip this check')

    return extracted_feature
        
  def detect(self, data_list, clip_fea, clip_attn_squeeze1, n_jobs=-1, clip_fea_nms=None, query_labels=None):
    n_jobs = min(32, max(len(data_list), n_jobs))
    
    if query_labels is None:
      query_labels = [str(d['pred'].item()) for d in data_list]
    queries_masks = self.sapmling_feature_mask(data_list, clip_feas=clip_fea, clip_attn_squeeze1=clip_attn_squeeze1, training=False) 
    out_list = []
    
    pool = Pool(n_jobs)
    length = len(data_list) // n_jobs
    split = len(data_list) % n_jobs
    mp_res = [pool.apply_async(detect_single, (self.data, self.coord, queries_masks[length * i:length * (i + 1)], clip_fea[length * i:length * (i + 1)], query_labels[length * i:length * (i + 1)])) if i<n_jobs-split else pool.apply_async(detect_single, (self.data, self.coord, queries_masks[length * i+i-n_jobs+split:length * (i + 1)+i-n_jobs+split+1], clip_fea[length * i+i-n_jobs+split:length * (i + 1)+i-n_jobs+split+1], query_labels[length * i+i-n_jobs+split:length * (i + 1)+i-n_jobs+split+1])) for i in range(n_jobs)]
    pool.close()
    pool.join()
    
    for p in mp_res:
      out = p.get()
      out_list.extend(out)
        
    return out_list