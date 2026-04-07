import warnings

import clip
import cv2
import numpy as np
import torch
import gc

from model.utils import img_ms_and_flip
from model.utils import reshape_transform
from model.utils import scoremap2bbox
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F

warnings.filterwarnings("ignore")

_EPSILON = 1e-15

def scale_cam_image(cam, target_size=None):
  """Normalize and rescale cam image."""
  result = []
  for img in cam:
    img = img - np.min(img)
    img = img / (_EPSILON + np.max(img))
    if target_size is not None:
      img = cv2.resize(img, target_size)
    result.append(img)
  result = np.float32(result)

  return result


class ActivationsAndGradients:
  """Class for extracting activations and registering gradients from targetted intermediate layers."""

  def __init__(self, model, target_layers, reshape_transform, stride=16):
    self.model = model
    self.gradients = []
    self.activations = []
    self.reshape_transform = reshape_transform
    self.handles = []
    self.stride = stride
    self.target_layers = target_layers
    for target_layer in target_layers:
      self.handles.append(
          target_layer.register_forward_hook(self.save_activation)
      )
      self.handles.append(
          target_layer.register_forward_hook(self.save_gradient)
      )

  def save_activation(self, module, input, output):
    """Saves activations from targetted layer."""
    activation = output

    if self.reshape_transform is not None:
      activation = self.reshape_transform(activation, self.height, self.width)
    self.activations.append(activation.cpu().detach())

  def save_gradient(self, module, input, output):
    if not hasattr(output, "requires_grad") or not output.requires_grad:
      return

    def _store_grad(grad):
      if self.reshape_transform is not None:
        grad = self.reshape_transform(grad, self.height, self.width)
      self.gradients = [grad.cpu().detach()] + self.gradients

    output.register_hook(_store_grad)

  def __call__(self, x, h, w):
    self.height = h // self.stride
    self.width = w // self.stride
    self.gradients = []
    self.activations = []
    if isinstance(x, tuple) or isinstance(x, list):
      return self.model.forward_last_layer(x[0], x[1])
    else:
      return self.model(x)

  def release(self):
    for handle in self.handles:
      handle.remove()
      
  def reinit(self):
    for handle in self.handles:
      handle.remove()
      
    self.handles.clear()
    self.activations.clear()
    self.gradients.clear()
    
    self.handles = []
    self.gradients = []
    self.activations = []
    for target_layer in self.target_layers:
      self.handles.append(
          target_layer.register_forward_hook(self.save_activation)
      )
      self.handles.append(
          target_layer.register_forward_hook(self.save_gradient)
      )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class CAM:
  """CAM module."""

  def __init__(
      self,
      model,
      target_layers,
      device,
      reshape_transform=None,
      compute_input_gradient=False,
      stride=16,
  ):
    self.model = model.eval()
    self.target_layers = target_layers
    self.model = model.to(device)
    self.reshape_transform = reshape_transform
    self.compute_input_gradient = compute_input_gradient
    self.activations_and_grads = ActivationsAndGradients(
        self.model, target_layers, reshape_transform, stride=stride
    )

  def get_cam(self, activations, grads, fg_mask=None):
    grads_ = grads if fg_mask is None else grads*fg_mask[None, None]
    activations_ = activations if fg_mask is None else activations*fg_mask[None, None]
    weights = np.mean(grads_, axis=(2, 3))
    weighted_activations = weights[:, :, None, None] * activations_
    cam = weighted_activations.sum(axis=1)
    return cam

  def forward(
      self,
      input_tensor,
      targets,
      target_size,
      fg_mask=None
  ):
    """CAM forward pass."""
    if self.compute_input_gradient:
      input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

    w, h = self.get_target_width_height(input_tensor)
    outputs = self.activations_and_grads(input_tensor, h, w)

    self.model.zero_grad()
    if isinstance(input_tensor, (tuple, list)):
      loss = sum(
          [target(output[0]) for target, output in zip(targets, outputs)]
      )
    else:
      loss = sum([target(output) for target, output in zip(targets, outputs)])
    loss.backward(retain_graph=True)
    cam_per_layer = self.compute_cam_per_layer(target_size, fg_mask=fg_mask)
    if isinstance(input_tensor, (tuple, list)):
      return (
          self.aggregate_multi_layers(cam_per_layer),
          outputs[0],
          outputs[1],
      )
    else:
      return self.aggregate_multi_layers(cam_per_layer), outputs

  def get_target_width_height(self, input_tensor):
    width = None
    height = None
    if isinstance(input_tensor, (tuple, list)):
      width, height = input_tensor[-1], input_tensor[-2]
    return width, height

  def compute_cam_per_layer(self, target_size, fg_mask=None):
    activations_list = [
        a.cpu().data.numpy() for a in self.activations_and_grads.activations
    ]
    grads_list = [
        g.cpu().data.numpy() for g in self.activations_and_grads.gradients
    ]

    cam_per_target_layer = []
    for i in range(len(self.target_layers)):
      layer_activations = None
      layer_grads = None
      if i < len(activations_list):
        layer_activations = activations_list[i]
      if i < len(grads_list):
        layer_grads = grads_list[i]
      cam = self.get_cam(layer_activations, layer_grads, fg_mask=fg_mask)
      cam = np.maximum(cam, 0).astype(np.float32)
      scaled = scale_cam_image(cam, target_size)
      cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer

  def aggregate_multi_layers(self, cam_per_target_layer):
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    return scale_cam_image(result)

  def __call__(
      self,
      input_tensor,
      targets=None,
      target_size=None,
      fg_mask=None
  ):
    return self.forward(input_tensor, targets, target_size, fg_mask=fg_mask)

  def __del__(self):
    self.activations_and_grads.release()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self.activations_and_grads.release()
    if isinstance(exc_value, IndexError):
      # Handle IndexError here...
      print(
          f"An exception occurred in CAM with block: {exc_type}. "
          f"Message: {exc_value}"
      )
      return True



################################################################################
#                               CLIP CAM BELLOW


class ClipOutputTarget:

  def __init__(self, category):
    self.category = category

  def __call__(self, model_output):
    if len(model_output.shape) == 1:
      return model_output[self.category]
    return model_output[:, self.category]


def zeroshot_classifier(classnames, templates, model, device):
  """Zeroshot classifier."""
  with torch.no_grad():
    zeroshot_weights = []
    for classname in classnames:
      if templates is None:
        texts = [classname]
      else:
        texts = [template.format(classname) for template in templates]
      texts = clip.tokenize(texts).to(device)  # tokenize
      class_embeddings = model.encode_text(texts)  # embed with text encoder
      class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
      class_embedding = class_embeddings.mean(dim=0)
      class_embedding /= class_embedding.norm()
      zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
  return zeroshot_weights.t()


class CLIPCAM:
  def __init__(
      self,
      clip_model,
      device,
      threshold=0.4,
      bg_cls=None,
  ):
    self.device = device
    self.clip_model = clip_model.to(device)
    self.threshold = threshold
    self.stride = self.clip_model.visual.patch_size

    self.bg_cls = bg_cls
    self.bg_text_features = None
    if self.bg_cls is not None:
      self.bg_text_features = zeroshot_classifier(
          self.bg_cls,
          ("a clean origami {}.",),
          self.clip_model,
          self.device,
      ).to(self.device)
    self.target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
    self.cam = CAM(
        model=self.clip_model,
        target_layers=self.target_layers,
        reshape_transform=reshape_transform,
        device=device,
        stride=self.stride,
    )

  def set_bg_cls(self, bg_cls):
    if not bg_cls:
      self.bg_cls = None
      self.bg_text_features = None
    else:
      self.bg_cls = bg_cls
      self.bg_text_features = zeroshot_classifier(
          self.bg_cls,
          ("a clean origami {}.",),
          self.clip_model,
          self.device,
      ).to(self.device)

  def __call__(self, ori_img, text_features, scale=1.0, fg_mask=None):
    ori_width = ori_img.size[0]
    ori_height = ori_img.size[1]

    ms_imgs = img_ms_and_flip(ori_img, ori_height, ori_width, scales=[scale])
    image = ms_imgs[0]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]
    image = image.to(self.device)
    image_features, attn_weight_list = self.clip_model.encode_image(image, h, w)

    highres_cam_to_save = []
    refined_cam_to_save = []
    token_cam_to_save = []

    bg_features_temp = self.bg_text_features.to(self.device)
    text_features_temp = torch.cat(
          [text_features, bg_features_temp], dim=0
      )
    input_tensor = [
        image_features,
        text_features_temp.to(self.device),
        h,
        w,
    ]

    for idx in range(len(text_features)):
      targets = [ClipOutputTarget(idx)]

      # torch.cuda.empty_cache()
      grayscale_cam, _, attn_weight_last = self.cam(
          input_tensor=input_tensor, targets=targets, target_size=None, fg_mask=fg_mask
      )  # (ori_width, ori_height))

      grayscale_cam = grayscale_cam[0, :]
      if grayscale_cam.max() == 0:
        input_tensor_fg = (
            image_features,
            text_features,
            h,
            w,
        )
        grayscale_cam, _, attn_weight_last = self.cam(
            input_tensor=input_tensor_fg,
            targets=targets,
            target_size=None,
        )
        grayscale_cam = grayscale_cam[0, :]

      grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
      highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

      if idx == 0:
        attn_weight_list.append(attn_weight_last)
        attn_weight = [
            aw[:, 1:, 1:] for aw in attn_weight_list
        ]  # (b, hxw, hxw)
        attn_weight = torch.stack(attn_weight, dim=0)[-8:]
        attn_weight = torch.mean(attn_weight, dim=0)
        attn_weight = attn_weight[0].cpu().detach()
      attn_weight = attn_weight.float()

      box, cnt = scoremap2bbox(
          scoremap=grayscale_cam,
          threshold=self.threshold,
          multi_contour_eval=True,
      )
      aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
      for i_ in range(cnt):
        x0_, y0_, x1_, y1_ = box[i_]
        aff_mask[y0_:y1_, x0_:x1_] = 1

      aff_mask = aff_mask.view(
          1, grayscale_cam.shape[0] * grayscale_cam.shape[1]
      )
      aff_mat = attn_weight

      trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
      trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

      for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
      trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

      for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

      trans_mat = trans_mat * aff_mask

      cam_to_refine = torch.FloatTensor(grayscale_cam)
      cam_to_refine = cam_to_refine.view(-1, 1)

      # (n,n) * (n,1)->(n,1)
      cam_token = torch.matmul(trans_mat, cam_to_refine)
      cam_refined = cam_token.reshape(h // self.stride, w // self.stride)
      cam_refined = cam_refined.cpu().numpy().astype(np.float32)
      cam_refined_highres = scale_cam_image(
          [cam_refined], (ori_width, ori_height)
      )[0]
      refined_cam_to_save.append(torch.tensor(cam_refined_highres))
      token_cam_to_save.append(cam_token.squeeze())

    cam_masks = torch.stack(refined_cam_to_save, dim=0)
    cam_tokens = torch.stack(token_cam_to_save, dim=0)
    self.cam.activations_and_grads.reinit()
    self.cam.model.zero_grad()
    self.clip_model.clear_all()
    del image
    if bg_features_temp is not None:
      del bg_features_temp
    return cam_masks.to(self.device), (cam_tokens/(cam_tokens.sum(1,keepdim=True)+1e-8)).view((-1, h // self.stride, w // self.stride)).to(self.device)#, fg_features_temp.to(self.device)

  
  def forward(self, ori_img, text_features, scale=1.0, fg_mask=None):
      ori_img = _ensure_image_tensor(ori_img, self.device)

      if ori_img.dim() != 3:
          raise ValueError(f"ori_img should be (3,H,W), got {tuple(ori_img.shape)}")

      _, ori_height, ori_width = ori_img.shape
      if scale != 1.0:
          proc_h = max(1, int(round(ori_height * scale)))
          proc_w = max(1, int(round(ori_width * scale)))
          image = F.interpolate(
              ori_img.unsqueeze(0),
              size=(proc_h, proc_w),
              mode="bilinear",
              align_corners=False
          )
      else:
          image = ori_img.unsqueeze(0)

      h, w = image.shape[-2], image.shape[-1]
      image = image.to(self.device)

      image_features, attn_weight_list = self.clip_model.encode_image(image, h, w)

      bg_features_temp = getattr(self, "bg_text_features", None)
      if bg_features_temp is not None:
          bg_features_temp = bg_features_temp.to(self.device)
          text_features_temp = torch.cat([text_features, bg_features_temp], dim=0)
      else:
          text_features_temp = text_features

      input_tensor = [
          image_features,
          text_features_temp,
          h,
          w,
      ]
      num_cls = len(text_features)
      targets = [ClipOutputTarget(i) for i in range(num_cls)]
      cams = []
      for idx in range(num_cls//8):
          targets = [ClipOutputTarget(idx)]
          cam_i, _, _ = self.cam(
              input_tensor=input_tensor,
              targets=targets,
              target_size=None,
              fg_mask=fg_mask,
          )
      [cams.append(torch.from_numpy(cam_i[0])) for idx in range(num_cls)]
      cams = torch.stack(cams, dim=0)

      if isinstance(cams, (list, tuple)):
          cams = torch.stack([torch.as_tensor(c, device=self.device) for c in cams], dim=0)
      else:
          cams = cams.to(self.device)

      if cams.dim() == 2:
          cams = cams.unsqueeze(0)
      elif cams.dim() == 4 and cams.shape[1] == 1:
          cams = cams[:, 0]

      cams = cams.float()
      
      with torch.no_grad():
          C, cam_h, cam_w = cams.shape
          aff_mask = (cams > self.threshold).float().view(C, -1)
          attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
          attn_weight = torch.stack(attn_weight, dim=0)[-8:]
          attn_weight = torch.mean(attn_weight, dim=0)
          attn_weight = attn_weight[0]  # (N,N)

          trans_mat = attn_weight
          trans_mat = trans_mat / (torch.sum(trans_mat, dim=0, keepdim=True) + _EPSILON)
          trans_mat = trans_mat / (torch.sum(trans_mat, dim=1, keepdim=True) + _EPSILON)

          for _ in range(2):
              trans_mat = trans_mat / (torch.sum(trans_mat, dim=0, keepdim=True) + _EPSILON)
              trans_mat = trans_mat / (torch.sum(trans_mat, dim=1, keepdim=True) + _EPSILON)

          trans_mat = (trans_mat + trans_mat.transpose(0, 1)) / 2.0

          trans_mat = torch.matmul(trans_mat, trans_mat)

          trans_mat = trans_mat.unsqueeze(0) * aff_mask.unsqueeze(1)   # (C,N,N)

          cam_flat = cams.view(C, -1, 1)                               # (C,N,1)
          cam_token = torch.matmul(trans_mat, cam_flat).squeeze(-1)    # (C,N)

          cam_refined = cam_token.view(C, cam_h, cam_w)

          cam_refined_highres = F.interpolate(
              cam_refined.unsqueeze(1),
              size=(ori_height, ori_width),
              mode="bilinear",
              align_corners=False
          ).squeeze(1)

          if hasattr(self.cam, "activations_and_grads") and hasattr(self.cam.activations_and_grads, "reinit"):
              self.cam.activations_and_grads.reinit()
          if hasattr(self.cam, "model") and hasattr(self.cam.model, "zero_grad"):
              self.cam.model.zero_grad(set_to_none=True)
          if hasattr(self.clip_model, "clear_all"):
              self.clip_model.clear_all()

          cam_tokens_norm = cam_token / (cam_token.sum(1, keepdim=True) + _EPSILON)
          cam_tokens_norm = cam_tokens_norm.view(C, cam_h, cam_w)

          return cam_refined_highres.to(self.device), cam_tokens_norm.to(self.device)
        
        

def _ensure_image_tensor(img, device):
    """
    PIL / Tensor -> (3,H,W) float tensor on device, range [0,1]
    """
    if torch.is_tensor(img):
        x = img
        if x.dim() == 4:
            x = x[0]
        x = x.to(device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return x

    x = pil_to_tensor(img).to(device).float() / 255.0
    return x