import torch
import torch.nn.functional as F
import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf import utils
from model.utils import erode, tensor_closing

class DenseCRF(object):
  def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
    self.iter_max = iter_max
    self.pos_w = pos_w
    self.pos_xy_std = pos_xy_std
    self.bi_w = bi_w
    self.bi_xy_std = bi_xy_std
    self.bi_rgb_std = bi_rgb_std

  def __call__(self, image, probmap):
    c, h, w = probmap.shape

    u = utils.unary_from_softmax(probmap)
    u = np.ascontiguousarray(u)

    image = np.ascontiguousarray(image)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
    d.addPairwiseBilateral(
        sxy=self.bi_xy_std,
        srgb=self.bi_rgb_std,
        rgbim=image,
        compat=self.bi_w,
    )

    q = d.inference(self.iter_max)
    q = np.array(q).reshape((c, h, w))
    
    q = tensor_closing(tensor_closing(torch.from_numpy(q).unsqueeze(1), erode_iters=3, dilate_iters=3, reverse=True)).squeeze(1).numpy()

    return q


class PostProcess:
  def __init__(self, device):
    self.device = device
    self.postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

  def apply_crf(self, image, cams):
    bg_score = 1 - np.max(cams, axis=0, keepdims=True)
    prob = np.concatenate((bg_score, cams), axis=0)

    image = image.astype(np.uint8).transpose(1, 2, 0)
    prob = self.postprocessor(image, prob)

    label = np.argmax(prob, axis=0)

    label_tensor = torch.from_numpy(label).long()
    refined_mask = F.one_hot(label_tensor, num_classes=len(prob)).to(device=self.device)
    refined_mask = refined_mask.permute(2, 0, 1)
    refined_mask = refined_mask[1:].float()
    return refined_mask

  def __call__(self, image, cams):
    mean_bgr = (104.008, 116.669, 122.675)
    image = np.array(image).astype(np.float32)

    image = image[:, :, ::-1]
    image -= mean_bgr
    image = image.transpose(2, 0, 1)

    if isinstance(cams, torch.Tensor):
      cams = cams.cpu().detach().numpy()

    return self.apply_crf(image, cams)


def IoM(pred, target, min_pred_threshold=0.2, mode='max', single_way=False):
  intersection = torch.einsum("mij,nij->mn", pred.to(target.device), target)
  area_pred = torch.einsum("mij->m", pred)
  area_target = torch.einsum("nij->n", target)
  iom_pred = torch.einsum("mn,m->mn", intersection, 1 / area_pred)
  if single_way:
    return iom_pred
  iom_target = torch.einsum("mn,n->mn", intersection, 1 / area_target)
  iom_target[iom_pred < min_pred_threshold] = 0
  if mode=='max':
    iom = torch.max(iom_target, iom_pred)
  elif mode=='min':
    iom = torch.min(iom_target, iom_pred)
  return iom

def IoU(pred, target):
  pred, target = pred.float(), target.float()
  intersection = torch.einsum("mij,nij->mn", pred.to(target.device), target)
  area_pred = torch.einsum("mij->m", pred)
  area_target = torch.einsum("nij->n", target)
  union = area_pred.unsqueeze(1) + area_target.unsqueeze(0) - intersection
  iou = intersection / (union + 1e-8)
  return iou


def to_competitive_cam(cams, fg_cam=None, tau=0.2):
  if fg_cam is not None:
    cams = torch.cat([1-fg_cam, cams], dim=0)
  N, H, W = cams.shape
  cams_flat = cams.view(N, -1)                      
  density = F.softmax(cams_flat / (cams_flat.mean(dim=1, keepdim=True) + 1e-8) / tau, dim=0) 
  if fg_cam is not None:
    density = density[1:]
    cams_flat = cams_flat[1:]
  density *= cams_flat.sum(dim=0, keepdim=True)
  return ((density - density.amin(1, keepdim=True)) / (density.amax(1, keepdim=True) - density.amin(1, keepdim=True) + 1e-8)).view(-1, H, W)


def otsu(masks: torch.Tensor, num_bins: int = 256, conf_only: bool = False):
    binary_masks_th = torch.zeros(masks.size(0), device=masks.device)
    
    thres = torch.linspace(0, 1, steps=num_bins+1, device=masks.device)
    thres = (thres[:-1] + thres[1:]) / 2
    
    for i, mask in enumerate(masks):
        hist = torch.histc(mask, bins=num_bins, min=0.0, max=1.0)
        total = hist.sum()
        if total == 0:
            continue
        
        sum_total = (hist * thres).sum()
        sumB = torch.tensor(0.0, device=masks.device)
        wB = torch.tensor(0.0, device=masks.device)
        
        max_var = torch.tensor(0.0, device=masks.device)
        t = torch.tensor(0.0, device=masks.device)
        
        for j in range(num_bins):
            wB += hist[j]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += hist[j] * thres[j]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                t = thres[j]
        binary_masks_th[i] = t
    
    if conf_only:
      fold_mask = masks>= binary_masks_th[:, None, None].expand_as(masks)
      binary_masks_th = (masks*fold_mask).sum((1,2)) / fold_mask.sum((1,2))
    return (masks >= binary_masks_th[:, None, None]).float()
    

def fill_closed_mask(softmask: torch.Tensor,
                            tol: float = 1e-6) -> torch.Tensor:
    with torch.no_grad():
        J = 1.0 - softmask              

        I = torch.full_like(J, float("-inf"))
        I[:, 0, :] = J[:, 0, :]
        I[:, -1, :] = J[:, -1, :]
        I[:, :, 0] = J[:, :, 0]
        I[:, :, -1] = J[:, :, -1]

        tmp = torch.empty_like(J)

        for i in range((softmask.size(-1)-1)//2):
            tmp.copy_(F.max_pool2d(I.unsqueeze(1),
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
                      .squeeze(1))
            tmp = torch.min(tmp, J)
            I[:, i+1, :] = tmp[:, i+1, :]
            I[:, -2-i, :] = tmp[:, -2-i, :]
            I[:, :, i+1] = tmp[:, :, i+1]
            I[:, :, -i-2] = tmp[:, :, -i-2]
            
    return 1.0 - I
  
  
def remove_multi_counter(
    masks: torch.Tensor,
    th: torch.Tensor = 0.4,
) -> torch.Tensor:
  C, H, W = masks.shape

  seed = F.one_hot(masks.view(C, -1).argmax(dim=1), num_classes=H * W).to(device=masks.device).float().view(*masks.shape).unsqueeze(1)
  masks_ = masks.unsqueeze(1)
  kernel = torch.ones((1, 1, 3, 3), device=masks.device).float()
  
  for _ in range(H-1):
    dilated = (F.conv2d(seed, kernel, padding=1) > 0).float()   # bool mask
    seed_ = dilated * masks_                
    if torch.equal(seed_, seed):
      break
    seed = seed_
  r = (masks_.sum((1,2,3)) - seed.sum((1,2,3)))/seed.sum((1,2,3))
  
  return (r<th) | (r>((1-th)/th))
  
def clean_multi_counter(
    masks: torch.Tensor,
) -> torch.Tensor:
  C, H, W = masks.shape
  masks_ = masks.unsqueeze(1)
  seed = masks_.clone()
  
  for _ in range(H-1):
    eroded = erode(seed, 3, 1)
    end = eroded.sum((1,2,3))==0
    eroded[end] = seed[end]
    if torch.equal(eroded, seed):
      break
    seed = eroded
  
  kernel = torch.ones((1, 1, 3, 3), device=masks.device).float()
  
  for _ in range(H-1):
    dilated = (F.conv2d(seed, kernel, padding=1) > 0).float()   # bool mask
    dilated *= masks_           
    if torch.equal(dilated, seed):
      break
    seed = dilated
  
  return seed.squeeze(1)