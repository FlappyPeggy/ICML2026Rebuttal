import os

import cv2
import numpy as np
from PIL import Image
import colorsys
import random

import matplotlib
import logging
matplotlib._log.setLevel(logging.ERROR)
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import torch.nn.functional as F

try:
  from torchvision.transforms import InterpolationMode

  BICUBIC = InterpolationMode.BICUBIC
except ImportError:
  BICUBIC = Image.BICUBIC

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _transform_resize(h, w):
    return Compose([
      Resize((h, w), interpolation=BICUBIC),
      _convert_image_to_rgb,
      ToTensor(),
      Normalize(
          (0.48145466, 0.4578275, 0.40821073),
          (0.26862954, 0.26130258, 0.27577711),
      ),
    ])


def img_ms_and_flip(image, ori_height, ori_width, scales=1.0, patch_size=16):
    if isinstance(scales, float):
        scales = [scales]

    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(
            int(np.ceil(scale * int(ori_height) / patch_size) * patch_size),
            int(np.ceil(scale * int(ori_width) / patch_size) * patch_size),
        )
        image = preprocess(image)
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(
      tensor.size(0), height, width, tensor.size(2)
    )

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    while True:
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        if thr_gray_heatmap.max() > 0 or threshold <= 0:
          break
        threshold -= 0.1
    contours = cv2.findContours(
      image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[_CONTOUR_INDEX]

    # if len(contours) == 0:
    if not contours:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def mask2chw(arr):
  rows, cols = np.where(arr == 1)
  center_y = int(np.mean(rows))
  center_x = int(np.mean(cols))
  height = rows.max() - rows.min() + 1
  width = cols.max() - cols.min() + 1
  return (center_y, center_x), height, width


def apply_visual_prompts(
    image_array,
    mask,
    alpha=1.,
    blur_strength=(15,15),
):  
    prompted_image = image_array.copy()
    blurred = cv2.GaussianBlur(prompted_image.copy(), blur_strength, 0)
    mask = np.where(mask, alpha, 1-alpha)[:, :, None]
    sharp_region = (prompted_image * mask).astype(np.uint8)
    blurred_region = (blurred * (1 - mask)).astype(np.uint8)
    prompted_image = cv2.add(sharp_region, blurred_region)
    prompted_image = Image.fromarray(prompted_image.astype(np.uint8))
    return prompted_image

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: 
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[ry] < self.rank[rx]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def adj_fuse(adj: torch.Tensor) -> list[list[int]]:
    N = adj.size(0)
    uf = UnionFind(N)

    idx_i, idx_j = torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=True)
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        uf.union(i, j)

    groups = {}
    for i in range(N):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


def dilate(x: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    pad = kernel_size // 2
    out = x.float()
    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size, stride=1, padding=pad)
    return (out > 0.5).to(x.dtype)

def erode(x: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    pad = kernel_size // 2
    out = (1 - x.float())
    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size, stride=1, padding=pad)
    out = 1 - out
    return (out > 0.5).to(x.dtype)

def tensor_closing(x: torch.Tensor,
            kernel_size: int = 3,
            dilate_iters: int = 1,
            erode_iters: int = 1,
            reverse: bool = False) -> torch.Tensor:
    return dilate(erode(x, kernel_size, erode_iters), kernel_size, dilate_iters) if reverse else erode(dilate(x, kernel_size, dilate_iters), kernel_size, erode_iters)

def normalize(x, dim=None, eps=1e-15):
  if dim is None:
    return (x - x.min()) / (x.max() - x.min())
  # Normalize to [0, 1].
  numerator = x - x.min(axis=dim, keepdims=True)[0]
  denominator = (
      x.max(axis=dim, keepdims=True)[0]
      - x.min(axis=dim, keepdims=True)[0]
      + eps
  )
  return numerator / denominator


  