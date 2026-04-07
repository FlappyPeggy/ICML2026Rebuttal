import os

import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import gc

from model.clip_wrapper import CLIPWrapper
from model.clip_wrapper import forward_clip
from model.clip_cam import CLIPCAM
from model.utils import apply_visual_prompts, adj_fuse, tensor_closing

from model.post_process import PostProcess, to_competitive_cam, otsu, clean_multi_counter, fill_closed_mask

from utils.utils import EPS

      
class CLIPSeg(nn.Module):

  def __init__(
      self,
      mask_threshold=-1,
      mode='gradcam_refine',#'cam_sam_matching_spatial_refine',
      guid=None,
      zoom_level=2,
      otsu_bin=False,
      refine_text=False,
      device=None
  ):
    super(CLIPSeg, self).__init__()
    # CLIP parameters
    torch.cuda.set_device(device)
    self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(f"cuda:{device}")
    self.semantic_templates = ['a clean origami {}.',
                     'a photo of a {}.',
                     'This is a photo of a {}',
                     'There is a {} in the scene',
                     'There is the {} in the scene',
                     'a photo of a {} in the scene',
                     'a photo of a small {}.',
                     'a photo of a medium {}.',
                     'a photo of a large {}.',
                     'This is a photo of a small {}.',
                     'This is a photo of a medium {}.',
                     'This is a photo of a large {}.',
                     'There is a small {} in the scene.',
                     'There is a medium {} in the scene.',
                     'There is a large {} in the scene.'
                     ]
    self.bg_cls = ['ground', 'land', 'grass', 'tree', 'building',
             'wall', 'sky', 'lake', 'water', 'river', 'sea',
             'railway', 'railroad', 'helmet', 'cloud', 'house',
             'mountain', 'ocean', 'road', 'rock', 'street',
             'valley', 'bridge', 'room', 'blur']
    self.mask_threshold = mask_threshold
    self.refine_text = refine_text
    self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
    self.clip_model = CLIPWrapper(self.clip_model)
    self.clip_model.eval()
    self.post_process = PostProcess(device=self.device)
    self.mask_generator = CLIPCAM(
        self.clip_model,
        device=self.device,
        bg_cls=self.bg_cls,
    )
    self.mode = mode
    self.otsu_bin = otsu_bin
    self.zoom_level = zoom_level
    if guid is None:
      self.guid = {
            'mode': ['image_prompt', 'cls_token', 'pe_token', ], # , 'final_attention', 'every_attention',
            'mask': None}
    else:
      self.guid = guid
    # self.sam_pipeline = load_sam('sam_vit_h.pth', 'vit_h', mode=self.mode) if 'sam' in self.mode else None

  def get_confidence(self, cam_map, binary_cam_map):
    confidence_map = torch.sum(cam_map * binary_cam_map[None], dim=[2, 3])
    confidence_map = confidence_map / torch.sum(binary_cam_map, dim=[1, 2])
    confidence_score = confidence_map.squeeze()
    return confidence_score
  
  def auto_mask_threshold(self, masks, step=0.004):
    best_th = 0.
    best_score = torch.tensor([0.])#.to(masks.device)
    for th in np.arange(0, 1, step):
      bin_mask = masks>th
      score = (2*bin_mask.any(0).sum() - bin_mask.sum()).cpu()
      if score > best_score: 
        best_score, best_th = score, th
    return best_th
    

  def apply_visual_prompts(self, image, mask, mask_threshold, alpha=1.):
    # if torch.sum(mask).item() <= 1:
      # return image
    image_array = np.array(image)
    img_h, img_w = image_array.shape[0:2]
    mask = F.interpolate(mask[None][None], size=(img_h, img_w), mode="nearest").squeeze().detach().cpu().numpy()
    mask = (mask > mask_threshold).astype(np.uint8)
    prompted_image = apply_visual_prompts(
        image_array, mask, alpha
    )
    return prompted_image
  
  # def aug_sam(self, mask, mask_list, crf_image=None, iteration=1):
  #   for _ in range(iteration):
  #     # intersection = torch.einsum("mij,nij->mn", mask, mask)
  #     # area = torch.einsum("mij->m", mask)
  #     # unique_idx = adj_fuse((intersection / (area.unsqueeze(1) + area.unsqueeze(0) - intersection + EPS))>0.95)
  #     # mask_, mask_list_ = [], []
  #     # for subset in unique_idx:
  #     #   mask_.append(mask[subset[0]] if len(subset)==1 else mask[[subset]].bool().any(0))
  #     #   mask_list_.append({'segmentation': mask_[-1].bool().cpu().numpy()})
  #     # mask, mask_list = torch.stack(mask_, dim=0).to(mask.dtype), mask_list_
  #
  #     intersection = torch.einsum("mij,nij->mn", mask, mask)
  #     area = torch.einsum("mij->m", mask)
  #     iom_target = torch.einsum("mn,n->mn", intersection, 1 / area)
  #     iom_pred = torch.einsum("mn,m->mn", intersection, 1 / area)
  #     # iou = intersection / (2 * area - intersection + EPS)
  #     pairs = torch.tril(
  #       (torch.max(iom_target, iom_pred)>0.95)
  #       , diagonal=-1)
  #
  #     pairs = list(zip(*torch.nonzero(pairs, as_tuple=True)))
  #
  #     if len(pairs)>0:
  #       pairs = [p if area[p[0]]>area[p[1]] else p[::-1] for p in pairs]
  #       pairs.sort(key=lambda pair: area[pair[0]].item(), reverse=False)
  #       large = torch.tensor([p[0] for p in pairs], dtype=torch.long)
  #       small = torch.tensor([p[1] for p in pairs], dtype=torch.long)
  #       new_mask = F.relu(mask[large] - mask[small])
  #       new_mask = new_mask[new_mask.sum((1,2))>8]
  #       intersection = torch.einsum("mij,nij->mn", new_mask, mask)
  #       union = torch.einsum("mij->m", new_mask).unsqueeze(1) + \
  #         torch.einsum("nij->n", mask).unsqueeze(0) - \
  #         intersection
  #       mask_keep = torch.amax(intersection / (union + EPS), dim=1)<=0.95
  #       new_mask = new_mask[mask_keep]#.view(mask_keep.sum(), *mask.size()[1:])
  #       if mask_keep.sum()>0:
  #         new_mask = tensor_closing(tensor_closing(new_mask.unsqueeze(1), dilate_iters=2, erode_iters=2, reverse=True),dilate_iters=2, erode_iters=2).squeeze(1)
  #         new_mask = clean_multi_counter(new_mask)
  #
  #         mask_keep = torch.ones(len(new_mask), dtype=torch.bool, device=new_mask.device)
  #         intersection = torch.einsum("mij,nij->mn", new_mask, new_mask)
  #         union = torch.einsum("mij->m", new_mask).unsqueeze(1) + \
  #           torch.einsum("nij->n", new_mask).unsqueeze(0) - \
  #           intersection
  #         iou_matrix = (intersection / (union + EPS))>0.95
  #         for i in range(len(new_mask)):
  #           if mask_keep[i]:
  #             mask_keep[iou_matrix[i]] = False
  #             mask_keep[i] = True
  #         new_mask = new_mask[mask_keep]
  #
  #         if crf_image: new_mask = self.post_process(crf_image, new_mask)
  #         mask = torch.cat([mask, new_mask], dim=0)
  #         mask_list += [{'segmentation': m.bool().cpu().numpy()} for m in new_mask]
  #     if len(mask)>1000:
  #       break
  #   return mask, mask_list

  def get_mask_confidence(self, prompted_images, prompt_text, bg_text=None, guid=None, pre_prompt_ensem=False):
    prompted_tensor = torch.stack(
        [self.preprocess(img) for img in prompted_images], dim=0
    )
    prompted_tensor = prompted_tensor.to(self.device)
    h, w = prompted_tensor.shape[-2:]
    text_prediction, v_feature = forward_clip(
        self.clip_model, prompted_tensor, prompt_text, h, w, guid=guid, bg_text=bg_text, pre_prompt_ensem=pre_prompt_ensem, return_feature=True
    )
    del prompted_tensor
    return text_prediction, v_feature

  def _forward_cam(self, ori_img, cam_text, semantic_prompt_text, bg_prompt_text=None, cls_idx=None, fg_mask=None):
    mask_proposals, _ = self.mask_generator(ori_img, cam_text, fg_mask=fg_mask)
    mask_threshold = self.auto_mask_threshold(mask_proposals[cls_idx]) if self.mask_threshold in [-1, None] else self.mask_threshold
    prompted_imgs = [
        self.apply_visual_prompts(ori_img, cam_map, mask_threshold)
        for cam_map in mask_proposals
    ]
    mask_scores, prompted_cam_feature = self.get_mask_confidence(prompted_imgs, semantic_prompt_text, bg_prompt_text, guid=None, pre_prompt_ensem=True)
    mask_scores = torch.diagonal(mask_scores).numpy()
      
    ori_mask_id = np.where(mask_scores>0.25)[0]
    if not len(ori_mask_id):
      ori_mask_id = [np.argmax(mask_scores)]
    texts = [cam_text[i] for i in ori_mask_id]
    remove_idxs = [idx for idx in range(len(mask_proposals)) if idx not in ori_mask_id]
    mask_scores[remove_idxs] = 0

    return texts, mask_scores, mask_proposals, prompted_cam_feature

  def _get_save_path(self, text):
    folder_name = "_".join([t.replace(" ", "_") for t in text])
    return [os.path.join(os.path.join('./outputs', folder_name), t.replace(" ", "_")) for t in text]

  def forward(self, ori_img, text_dict, cls_key=None, text_feature=None, text_feature_part=None):
    full_image_resized = ori_img.resize((224, 224))
    text = [k+"'s "+v_ for k, v in text_dict.items() for v_ in v]
    cam_text = text.copy()
    mask_tensor = None
    available = True
    if cls_key is None:
      semantic_prompt_text = [[template.format(t) for t in text_dict.keys()] for template in self.semantic_templates]
      prompted_tensor = self.preprocess(full_image_resized)
      h, w = prompted_tensor.shape[-2:]
      cls_idx = self.clip_model(prompted_tensor.unsqueeze(0).to(self.device),
                        clip.tokenize([x for sub in semantic_prompt_text for x in sub]).to(self.device) if text_feature is None else None, 
                        h, w, repeat_last=False, num_cls_to_ensem=len(semantic_prompt_text[0]),
                        text_features=text_feature).detach().cpu()[0].argmax()
      cls_key = list(text_dict.keys())[cls_idx]
    else:
      cls_idx = list(text_dict.keys()).index(cls_key)
    
    part_idx_within_cls = [i for i, v in enumerate(text) if v.startswith(cls_key)]
    semantic_prompt_text = [[template.format(t) for t in text] for template in self.semantic_templates]
    semantic_prompt_text_within_cls = [[template.format(t) for t in [cls_key+"'s "+v for v in text_dict[cls_key]]] for template in self.semantic_templates]
    
    # zsl-cam-seg
    cam_cls = self.mask_generator(full_image_resized, [k+"'s " for k in list(text_dict.keys())])#["a clean origami {}.".format(t) for t in list(text_dict.keys())])
    _fg_mask_proposals, fg_token_proposals = cam_cls[0][cls_idx], cam_cls[1][cls_idx].cpu().numpy()
    _fg_mask_proposals = fill_closed_mask(_fg_mask_proposals.unsqueeze(0)).unsqueeze(0)
    _fg_mask_proposals = F.interpolate(_fg_mask_proposals.detach().cpu(), size=ori_img.size[::-1], mode="nearest").squeeze()
    fg_mask_proposals = _fg_mask_proposals.numpy()
    ys, xs = np.where(fg_mask_proposals > 0.4)
    if len(xs) == 0 or len(ys) == 0:
      ys, xs = np.where(fg_mask_proposals)
      if len(xs) == 0 or len(ys) == 0:
        ys, xs = np.array([0, fg_mask_proposals.shape[0]-1]), np.array([0, fg_mask_proposals.shape[1]-1])
        # return None, None, None, None
        # raise "OOD no CAM"
      available = False
    crop_boxA = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    crop_len = max(xs.max()-xs.min(), ys.max()-ys.min())
    minx, miny = max(min((xs.max()+xs.min()-crop_len)/2, fg_mask_proposals.shape[-1]-crop_len), 0), max(min((ys.max()+ys.min()-crop_len)/2, fg_mask_proposals.shape[-2]-crop_len), 0)
    crop_boxB = (np.round(minx, 0).astype(int), np.round(miny, 0).astype(int), min(np.round(minx+crop_len + 1, 0).astype(int), fg_mask_proposals.shape[-1]), min(np.round(miny+crop_len + 1, 0).astype(int), fg_mask_proposals.shape[-2]))
    if self.zoom_level==0:
      seg_box, score_box = crop_boxB, crop_boxB
    elif self.zoom_level==1:
      seg_box, score_box = crop_boxA, crop_boxB
    elif self.zoom_level==2:
      seg_box, score_box = crop_boxA, crop_boxA
    else:
      raise f'no implementation for zoom_level:{self.zoom_level}'
    masked_image_zoomed = Image.fromarray((np.array(ori_img)).astype(np.uint8)).crop(seg_box).resize((224, 224))
    fg_mask_proposals = Image.fromarray(fg_mask_proposals).crop(seg_box).resize((224, 224))
    
    # run cam-seg
    num_positive_last = len(cam_text)
    cnt = 0
    while True:
      cam_predicted_class_idx = [text.index(t) for t in cam_text]
      if len(cam_predicted_class_idx) == len(text):
        cam_text, final_all_scores, cam_mask_proposals, prompted_cam_feature = self._forward_cam(masked_image_zoomed, cam_text, semantic_prompt_text, cls_idx=[i for i, v in enumerate(cam_text) if v.startswith(cls_key)], fg_mask=fg_token_proposals)
      else:
        cam_text, final_all_scores[cam_predicted_class_idx], cam_mask_proposals[cam_predicted_class_idx], prompted_cam_feature[cam_predicted_class_idx] = self._forward_cam(masked_image_zoomed, cam_text, semantic_prompt_text, cls_idx=[i for i, v in enumerate(cam_text) if v.startswith(cls_key)], fg_mask=fg_token_proposals)
      semantic_prompt_text = [[template.format(t) for t in cam_text] for template in self.semantic_templates]
      num_positive = len(cam_text)
      if num_positive == num_positive_last or (not self.refine_text and cnt>0):
        break
      num_positive_last = num_positive
      cnt+=1
      
    cam_predicted_class_idx = [text.index(t) for t in cam_text]
    cam_predicted_class_idx_within_cls = [text.index(t) for t in cam_text if cls_key in t]
    cam_mask_proposals[cam_predicted_class_idx] = to_competitive_cam(cam_mask_proposals[cam_predicted_class_idx], fg_cam=torch.from_numpy(np.array(fg_mask_proposals))[None].to(cam_mask_proposals.device))
    final_predicted_masks = torch.zeros_like(cam_mask_proposals)
    final_predicted_masks[cam_predicted_class_idx] = self.post_process(masked_image_zoomed, cam_mask_proposals[cam_predicted_class_idx])
      
    # run sam-refine
    # if 'sam' in self.mode and 'prompt' in self.mode and len(cam_predicted_class_idx_within_cls):
    #   prompts = {'box':CAM2SAMBox(cam_mask_proposals[cam_predicted_class_idx_within_cls]),
    #             'click':CAM2SAMClick(cam_mask_proposals[cam_predicted_class_idx_within_cls]),
    #             'mask':[None]*len(cam_predicted_class_idx_within_cls)}
    #               # F.interpolate(cam_mask_proposals[cam_predicted_class_idx_within_cls][:, None].float(), size=(256,256), mode="nearest").cpu().numpy()}
    #   mask_tensor, mask_list = generate_masks_from_sam(
    #       np.array(masked_image_zoomed),
    #       save_path="./",
    #       pipeline=self.sam_pipeline,
    #       prompt=prompts
    #   )
    #   final_predicted_masks[cam_predicted_class_idx_within_cls] = mask_tensor.to(self.device)
    #
    # if 'sam' in self.mode and 'match' in self.mode:
    #   mask_tensor, mask_list = generate_masks_from_sam(
    #       np.array(masked_image_zoomed),
    #       save_path="./",
    #       pipeline=self.sam_pipeline,
    #       prompt=None
    #   )
    #   mask_tensor, mask_list = self.aug_sam(mask_tensor.to(self.device), mask_list, iteration=2)# sam合并和扩增
    #   prompted_imgs = [
    #       self.apply_visual_prompts(masked_image_zoomed, cam_map, 0.5) # ori_img
    #       for cam_map in mask_tensor
    #   ]
    #   # mask_scores, prompted_sam_feature = self.get_mask_confidence(prompted_imgs, semantic_prompt_text, pre_prompt_ensem=True)
    #   mask_scores, prompted_sam_feature = zip(*[
    #     self.get_mask_confidence(prompted_imgs[i : i + 1000], semantic_prompt_text_within_cls, pre_prompt_ensem=True)
    #     for i in range(0, len(prompted_imgs), 1000)
    #   ])
    #   mask_scores, prompted_sam_feature = torch.cat(mask_scores, dim=0), torch.cat(prompted_sam_feature, dim=0)
    #   # mask_scores, prompted_sam_feature = self.get_mask_confidence(prompted_imgs, semantic_prompt_text_within_cls, pre_prompt_ensem=True)#
    #   # print(mask_scores.topk(4, dim=0, largest=True, sorted=True))
    #
    #   if 'spatial' in self.mode:
    #     feature_similiarity = None
    #   else:
    #     feature_similiarity = prompted_sam_feature @ prompted_cam_feature[part_idx_within_cls].t()
    #
    #   # cam_mask_proposals[part_idx_within_cls] = to_competitive_cam(cam_mask_proposals[part_idx_within_cls], fg_cam=torch.from_numpy(np.array(fg_mask_proposals))[None].to(cam_mask_proposals.device))
    #   mapping, iou = map_sam_to_cam(mask_tensor,
    #                                 cam_mask_proposals[part_idx_within_cls],
    #                                 mask_scores,
    #                                 torch.from_numpy(final_all_scores==0).to(mask_tensor.device)[part_idx_within_cls],
    #                                 feature_similiarity,
    #                                 fg_cam=torch.from_numpy(np.array(fg_mask_proposals))[None].to(mask_tensor.device))
    #   final_predicted_masks[part_idx_within_cls] = merge_sam_to_cam(mask_tensor, otsu(cam_mask_proposals[part_idx_within_cls]), mapping, iou)
    if self.otsu_bin:
      final_predicted_masks[part_idx_within_cls] = otsu(cam_mask_proposals[part_idx_within_cls])
      
    final_predicted_masks_orisize = torch.zeros((final_predicted_masks.size(0)+1, *ori_img.size[::-1]))
    final_predicted_masks_orisize[:-1, seg_box[1]:seg_box[3], seg_box[0]:seg_box[2]] = F.interpolate(final_predicted_masks[:, None].float(), size=(seg_box[3]-seg_box[1], seg_box[2]-seg_box[0]), mode="nearest").bool().squeeze()
    final_predicted_masks_orisize[-1] = _fg_mask_proposals > 0.4
    
    if self.zoom_level == 1:#
      final_predicted_masks[part_idx_within_cls] = F.interpolate(final_predicted_masks_orisize[part_idx_within_cls, score_box[1]:score_box[3], score_box[0]:score_box[2]].unqueeze(1), size=(224, 224), mode="nearest").squeeze()
      masked_image_zoomed = Image.fromarray((np.array(ori_img)).astype(np.uint8)).crop(score_box).resize((224, 224))
      
    if self.guid['mode'] is not None and len(self.guid['mode'])>0:
      self.guid['mask'] = F.adaptive_avg_pool2d(torch.cat([final_predicted_masks[part_idx_within_cls],torch.zeros_like(final_predicted_masks[0])[None]], dim=0).unsqueeze(1), output_size=(14, 14)).squeeze(1).view(len(part_idx_within_cls)+1, -1).half()
      self.guid['mask'] /= self.guid['mask'].sum(-1, keepdim=True).clamp(min=EPS)
    else:
      self.guid = None
    
    if self.guid is not None and 'image_prompt' in self.guid['mode']:
      prompted_imgs = [
        self.apply_visual_prompts(masked_image_zoomed, cam_map, 0.5, alpha=1.)
        for cam_map in final_predicted_masks[part_idx_within_cls]
      ]
    else:
      prompted_imgs = [masked_image_zoomed.copy() for _ in range(len(part_idx_within_cls))]
    prompted_imgs.append(self.apply_visual_prompts(masked_image_zoomed, torch.zeros_like(final_predicted_masks[0]), 0.5, alpha=1.))
    score_mat, final_prompt_feature_within_class = self.get_mask_confidence(prompted_imgs, semantic_prompt_text_within_cls, pre_prompt_ensem=True, guid=self.guid)
    final_all_scores[part_idx_within_cls] = torch.diagonal(score_mat).numpy()
    
    del final_predicted_masks, cam_mask_proposals
    
    return available, final_predicted_masks_orisize[part_idx_within_cls+[-1]], final_all_scores, final_prompt_feature_within_class.cpu(), cls_key
