import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import EPS, SEMANTIC_TEMPLATE
from torch.cuda.amp import GradScaler, autocast

_CONTEXT_LENGTH = 77

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
  embed = F.upsample(
      embed,
      size=new_size,
      mode='bilinear',
  )
  embed = embed.view(d, -1).contiguous()
  embed = embed.permute(1, 0)
  embed = torch.cat([first, embed], 0)
  embed = nn.parameter.Parameter(embed.half())
  return embed


def forward_clip_single(model, image, text, h, w, guid=None, softmax=True, return_feature=False):
  num_cls_to_ensem = None
  if isinstance(text[0], list):
    num_cls_to_ensem = len(text[0])
    text = [x for sub in text for x in sub]
  text_tokens = clip.tokenize(text).to(image.device)
  text_prediction = model(image, text_tokens, h, w, guid=guid, repeat_last=True, softmax=softmax, num_cls_to_ensem=num_cls_to_ensem, return_feature=return_feature)
  del text_tokens
  if return_feature:
    return [text_prediction[0].detach().cpu(), text_prediction[1]]
  return text_prediction.detach().cpu()


def forward_clip(model, image, text, h, w, guid=None, bg_text=None, pre_prompt_ensem=False, return_feature=False):
  if pre_prompt_ensem:
    text_prediction = forward_clip_single(model, image, text, h, w, softmax=False, guid=guid, return_feature=return_feature)
  else:
    text_prediction = torch.stack(
        [forward_clip_single(model, image, t, h, w, guid=guid) for t in text], dim=0
      ).sum(0)
    if bg_text is None:
      text_prediction = F.softmax(text_prediction.float(), dim=-1)
    else:
      bg_prediction = torch.stack(
        [forward_clip_single(model, image, t, h, w, guid=guid) for t in bg_text], dim=0
      ).sum(0)
      text_prediction_ = F.softmax(torch.cat([text_prediction, bg_prediction], dim=-1), dim=-1)
      text_prediction = torch.cat([text_prediction_[:, :len(text_prediction)], text_prediction_[:, len(text_prediction):].sum(1,keepdim=True)], dim=-1) 
  
  return [text_prediction[0].float(), text_prediction[1]] if isinstance(text_prediction, list) else text_prediction.float()


class CustomBlock(nn.Module):
  """A customized attention block."""

  def __init__(self, block, loramode=None):
    super().__init__()
    for k, v in vars(block).items():
      setattr(self, k, v)
    

  def attention(self, x):
    self.attn_mask = None if self.attn_mask is None else self.attn_mask.to(dtype=x.dtype, device=x.device)
    self.attn = self.attn.to(dtype=x.dtype, device=x.device)
    return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
    
  def forward(self, x, guid=None, compute_v=False):
    x_ln = self.ln_1(x)
    
    if compute_v:
      v = F.linear(x_ln.permute(1, 0, 2), self.attn.in_proj_weight, self.attn.in_proj_bias)
      N, L, C = v.shape
      v = v.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
      v = F.linear(v, self.attn.out_proj.weight, self.attn.out_proj.bias)
      v = v.tensor_split(3, dim=0)[-1].permute(1, 0, 2) + x
      v = v + self.mlp(self.ln_2(v))
    
    
    attn_output, attn_weight = self.attention(x_ln)
    if guid is not None and 'every_attention' in guid['mode']:
      attn_output = attn_output.permute(1, 0, 2)  
      weighted_attn = torch.einsum('nl,nld->nd', guid['mask'], attn_output[:, 1:]) 
      weighted_attn = (weighted_attn - weighted_attn.mean(-1, True)) \
         * (attn_output[:, 0].std(-1, keepdim=True).clamp(min=EPS) / weighted_attn.std(-1, keepdim=True).clamp(min=EPS)) \
         + attn_output[:, 0].mean(-1, True)
      attn_output[:, 0] = attn_output[:, 0] * 0.5 + weighted_attn * 0.5
      attn_output = attn_output.permute(1, 0, 2)
      
    x = x + attn_output
    x = x + self.mlp(self.ln_2(x))
    if compute_v:
      return x, v
    return x, attn_weight


class CustomTransformer(nn.Module):

  def __init__(self, transformer):
    super().__init__()
    for k, v in vars(transformer).items():
      setattr(self, k, v)

    self.resblocks = nn.Sequential(
        *[CustomBlock(block) for idx, block in enumerate(self.resblocks)]
    )

  def forward(self, x, guid=None, return_feature_list=False, training=False):
    attn_weights, fea_list = [], []
    if training:
      layers = self.layers if x.shape[0] == _CONTEXT_LENGTH else self.layers - 1
      for i in range(layers):
        x, attn_weight = self.resblocks[i](x, guid)
      return x, None
    else:
      with torch.no_grad():
        layers = self.layers if x.shape[0] == _CONTEXT_LENGTH else self.layers - 1
        for i in range(layers):
          x, attn_weight = self.resblocks[i](x, guid)
          attn_weights.append(attn_weight)
          if return_feature_list and i in [3, 6, 9]:
            fea_list.append(x[0].detach().cpu()[:, None])
      if return_feature_list:
        return x, attn_weights, fea_list
      else:
        return x, attn_weights


class CustomVisionTransformer(nn.Module):

  def __init__(self, model, loramode):
    super().__init__()
    for k, v in vars(model).items():
      setattr(self, k, v)
    self.patch_size = self.conv1.kernel_size[0]
    self.transformer = CustomTransformer(self.transformer)

  def forward(self, x, h=224, w=224, guid=None, return_feature_list=False, grad=False):
    self.positional_embedding_new = upsample_position_embedding(
        self.positional_embedding, (h // self.patch_size, w // self.patch_size)
    )
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    zeros = torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    cls_token = self.class_embedding.to(x.dtype) + zeros
    pe_token = self.positional_embedding_new.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
    
    # init cls token
    if guid is not None and 'cls_token' in guid['mode']:
      weighted_cls = torch.einsum('nl,nld->nd', guid['mask'], x).unsqueeze(1)
      weighted_cls = (weighted_cls - weighted_cls.mean(-1, True)) \
         * (cls_token.std(-1, keepdim=True).clamp(min=EPS) / weighted_cls.std(-1, keepdim=True).clamp(min=EPS)) \
         + cls_token.mean(-1, True)
      cls_token = (cls_token + weighted_cls) /2
    if guid is not None and 'pe_token' in guid['mode']:
      weighted_pe = torch.einsum('nl,nld->nd', guid['mask'], pe_token[:, 1:])
      weighted_pe = (weighted_pe - weighted_pe.mean(-1, True)) \
         * (pe_token[:, 0].std(-1, keepdim=True).clamp(min=EPS) / weighted_pe.std(-1, keepdim=True).clamp(min=EPS)) \
         + pe_token[:, 0].mean(-1, True)
      pe_token[:, 0] = (pe_token[:, 0] + weighted_pe) /2
      
    x = torch.cat([cls_token, x], dim=1)
    x = x + pe_token
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)
    if return_feature_list:
      x, attn_weight, fea_list = self.transformer(x, guid, True)
      del cls_token, pe_token
      return x, attn_weight, fea_list
    else:
      x, attn_weight = self.transformer(x, guid, training=grad)
      del cls_token, pe_token
      return x, attn_weight


class CLIPWrapper(nn.Module):
  def __init__(self, clip_model, text=None, loramode=None):#, reset_scale=False):
    super().__init__()
    for k, v in vars(clip_model).items():
      setattr(self, k, v)
    self.device = self.visual.conv1.weight.device
    self.visual = CustomVisionTransformer(self.visual, loramode)
    self.transformer = CustomTransformer(self.transformer)
    
    if text is not None:
      self.text_features = self.get_text_features(text, SEMANTIC_TEMPLATE)
    # if reset_scale:
    #   self.logit_scale = torch.nn.Parameter(torch.zeros(1, device=self.logit_scale.device, dtype=self.dtype))

  @property
  def dtype(self):
    return self.visual.conv1.weight.dtype

  def encode_image(self, image, h, w):
    return self.visual(image.type(self.dtype), h, w)
  

  def get_normed_fea(self, x, cls_only=True, return_feature=False, grad=False, return_attn=False, return_v=False, ori=False):
    assert not grad or not return_feature
    if return_feature:
      cube, _, feature_list = self.visual(x.to(self.device), 224, 224, guid=None, return_feature_list=True)
    else:
      cube, _ = self.visual(x.to(self.device), 224, 224, guid=None, return_feature_list=False, grad=grad)
    
    cube, attn = self.visual.transformer.resblocks[self.visual.transformer.layers - 1](cube, compute_v=return_v)
    if return_v:
      v = self.visual.ln_post(attn.permute(1, 0, 2))[:, 1:]
      v = F.normalize(v @ self.visual.proj, p=2, dim=-1)
    cube = cube.permute(1, 0, 2)
    if return_feature: feature_list.append(cube[:, :1].detach().cpu())
    
    cube = self.visual.ln_post(cube)
    if cls_only:
      cube = cube[:, 0]
    
    if return_attn:
      return F.normalize(cube @ self.visual.proj, p=2, dim=-1), torch.cat(feature_list, dim=1), cube.cpu(), attn.mean(1)[:, 1:].cpu()
    if return_v:
      return F.normalize(cube @ self.visual.proj, p=2, dim=-1), torch.cat(feature_list, dim=1), v
    if return_feature:
      return F.normalize(cube @ self.visual.proj, p=2, dim=-1), torch.cat(feature_list, dim=1)
    if ori:
      return cube
    return F.normalize(cube @ self.visual.proj, p=2, dim=-1) # B N C
      
  
  def encode_text(self, text, return_eot=True):
    x = self.token_embedding(text).type(
        self.dtype
    )

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x, _ = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)
    
    if not return_eot:
      return x
    return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
  
  @torch.no_grad()
  def get_text_features(self, text, TEMPLATE):
    prompted_text = [[template.format(t) for t in text] for template in TEMPLATE]
    text_features = self.encode_text(clip.tokenize([x for sub in prompted_text for x in sub]).to(self.device))
    text_features = F.normalize(text_features.view(-1, len(prompted_text[0]), text_features.size(-1)).mean(0), dim=1, p=2)
    return text_features

  def pool_visual(self, x, use_cls_token=False, guid=None):
    if guid is not None and 'final_attention' in guid['mode']:
      weighted_out = torch.einsum('nl,nld->nd', guid['mask'], x[:, 1:])  # [N, E]
      weighted_out = (weighted_out - weighted_out.mean(-1, True)) \
         * (x[:, 0].std(-1, keepdim=True).clamp(min=EPS) / weighted_out.std(-1, keepdim=True).clamp(min=EPS)) \
         + x[:, 0].mean(-1, True)
      return x[:, 0]* 0.5+ weighted_out * 0.5 if use_cls_token else weighted_out
    else:
      return x[:, 0] if use_cls_token else torch.mean(x[:, 1:, :], dim=1)
      
  @torch.enable_grad()
  def grad_stability(self, x, text_features, repeat_last, num_cls_to_ensem):
    grad_stability = []
    for input in x.clone().permute(1, 0, 2).unsqueeze(2):
      input = (input*(torch.randn((input.size(0), 32, input.size(-1)), device=input.device, dtype=input.dtype)/(input.size(-1)**0.5))).requires_grad_(True)
      sim = self.forward_last_layer(
          input, text_features, use_cls_token=True, repeat_last=repeat_last, softmax=False, num_cls_to_ensem=num_cls_to_ensem, 
      )[0]
      # maxsim = torch.logsumexp(sim, dim=1).sum()
      maxsim = torch.amax(sim, dim=1).sum()
      grad_stability.append(torch.norm(torch.autograd.grad(outputs=maxsim, inputs=input)[0], p=2, dim=-1).mean())
    return torch.stack(grad_stability)

  def forward_last_layer(
      self, image_features, text_features, use_cls_token=False, repeat_last=True, softmax=True, num_cls_to_ensem=None, return_feature=False, guid=None
  ):
    if repeat_last:
      if image_features.shape[1]>1000:
        x, attention_weight = [], []
        for chunk in range(int(np.ceil(image_features.shape[1]/1000))):
          x_, attention_weight_ = self.visual.transformer.resblocks[
            self.visual.transformer.layers - 1
          ](image_features[:, chunk*1000:min(image_features.shape[1], chunk*1000+1000)])
          x.append(x_.cpu());attention_weight.append(attention_weight_.cpu())
        x = torch.cat(x, dim=1).to(image_features.device)
        attention_weight = torch.cat(attention_weight, dim=0).to(image_features.device)
      else:
        x, attention_weight = self.visual.transformer.resblocks[
          self.visual.transformer.layers - 1
        ](image_features)
        
    else:
      x = image_features
      attention_weight = None
    x = x.permute(1, 0, 2)  # LND -> NLD
    
    x = self.visual.ln_post(x)
    x = self.pool_visual(x, use_cls_token=use_cls_token, guid=guid)

    if self.visual.proj is not None:
      x = x @ self.visual.proj

    image_features_no_norm = x
    image_features = image_features_no_norm / image_features_no_norm.norm(dim=-1, keepdim=True)
    # text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # if num_cls_to_ensem is not None:
    #   text_features = text_features.view(-1, num_cls_to_ensem, text_features.size(-1)).mean(0)
    #   text_features = text_features / text_features.norm(dim=1, keepdim=True)
    #   attention_weight = None
    
    # cosine similarity as logits
    logits_per_image = image_features @ text_features.t()
    if len(logits_per_image.shape)==3:
      logits_per_image = logits_per_image.amax(1)
    if softmax:
      logit_scale = self.logit_scale.exp()
      logits_per_image = logit_scale * logits_per_image
      logits_per_image = F.softmax(logits_per_image.float(), dim=-1)
      
    if return_feature:
      logits_per_image = [logits_per_image, image_features_no_norm]
    return logits_per_image, attention_weight

  @torch.no_grad()
  def forward(self, image, text, h=224, w=224, softmax=True, repeat_last=False, guid=None, num_cls_to_ensem=None, return_feature=False, text_features=None):
    if text_features is None:
      if len(text)>10000 and num_cls_to_ensem is not None:
        text_features = torch.cat([self.encode_text(chunk)[None].cpu() for chunk in text.view(-1, num_cls_to_ensem, text.size(-1))], dim=0)
      else:
        text_features = self.encode_text(text)
      if num_cls_to_ensem is not None:
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(-1, num_cls_to_ensem, text_features.size(-1)).mean(0)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.to(image.device)
    
    if len(image)>1000:
      feature_map = torch.cat([self.visual(image[chunk*1000:min(len(image), chunk*1000+1000)].type(self.dtype), h, w, guid=guid)[0].cpu() for chunk in range(int(np.ceil(len(image)/1000)))], dim=1).to(image.device)
    else:
      feature_map, _ = self.visual(image.type(self.dtype), h, w, guid=guid)

    logits_per_image, _ = self.forward_last_layer(
        feature_map, text_features, use_cls_token=True, repeat_last=repeat_last, softmax=softmax, num_cls_to_ensem=num_cls_to_ensem, return_feature=return_feature, guid=guid
      )
    return logits_per_image

  def clear_all(self):
    if hasattr(self.visual, 'positional_embedding_new'):
      setattr(self.visual, 'positional_embedding_new', None)
    
    # for resblocks in [self.visual.transformer.resblocks, self.transformer.resblocks]:
    #   for block in resblocks:
    #     if hasattr(block, 'attn_mask'):
    #       setattr(block, 'attn_mask', None)
    #     if hasattr(block, 'attn'):
    #       setattr(block, 'attn', None)