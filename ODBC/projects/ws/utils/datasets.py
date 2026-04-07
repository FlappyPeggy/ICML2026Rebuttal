import random
from torchvision import transforms
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import os.path
import torch.nn.functional as F


PREFIX = '../../data/'


class MultiCrop(object):
    def __init__(self, n_crop=256):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


# def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
#     assert 0.0 <= v <= 0.2
#     if v <= 0.:
#         return img

#     v = v * img.size[0]
#     return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


class SVHN(data.Dataset):
    def __init__(self, root,
                 transform=None):
        self.root = root
        self.transform = transform
        self.filename = "selected_test_32x32.mat"
        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        self.data = loaded_mat['X']
        self.targets = loaded_mat['y']
        self.targets = (self.targets % 10).squeeze()  # convert to zero-based indexing
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        return img, self.transform(img), target.astype(np.int64)

    def __len__(self):
        return len(self.data)

class FromLoader(Dataset):
    def __init__(self, fnames, gt, transform, coreset_idx=None):
        self.filenames = fnames
        self.transform = transform
        self.gt = gt
        self.coreset_idx = coreset_idx

    def __len__(self):
        return len(self.filenames) if self.coreset_idx is None else len(self.coreset_idx)

    def __getitem__(self, idx):
        idx_ = idx if self.coreset_idx is None else self.coreset_idx[idx]
        im = Image.open(self.filenames[idx_]).convert('RGB')
        return {'pil': im,
                'tensor': self.transform(im),
                'gt': self.gt[idx_]}


class Loader(Dataset):
    def __init__(self, fnames, transform, subset=-1, ALL_LABEL=None, mapper=None):
        self.filenames = fnames
        self.transform = transform
        if ALL_LABEL is None:
            try:
                self.gt = np.array([int(path.split('/')[-2]) for path in self.filenames])
            except:
                self.gt = list(set([path.split('/')[-2] for path in self.filenames]))
                self.gt.sort()
                self.gt = np.array([self.gt.index(path.split('/')[-2]) for path in self.filenames])
        else:
            try:
                # should in this case
                self.gt = np.array([ALL_LABEL.index(path.split('/')[-2][4:]) for path in self.filenames])
            except:
                try:
                    self.gt = np.array([ALL_LABEL.index(mapper[path.split('/')[-2]]) for path in self.filenames])
                except:
                    raise "laod fail, ALL_LABEL mismatch"
            
        if subset>0:
            all_gts = np.unique(self.gt)
            self.filenames = [self.filenames[idx] for gt in all_gts for idx in np.where(self.gt==gt)[0][:subset]]
            self.gt = np.array([self.gt[idx] for gt in all_gts for idx in np.where(self.gt==gt)[0][:subset]])
                
        self.gt = torch.from_numpy(self.gt)
            
    def get(self):
        return self.filenames, self.gt

    def get_class_num(self):
        return len(np.unique(np.array(self.gt)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        im = Image.open(self.filenames[idx]).convert('RGB')
        return {'pil': im,
                'tensor': self.transform(im),
                'gt': self.gt[idx]}
        

def to_same_type(all_data, tosize=224):
    smask = []
    for item in all_data:
        _, h, w = item.shape
        if h==tosize and w ==tosize:
            smask.append(item.half())
        else:
            if h < w:
                new_h, new_w = tosize, int(tosize * w / h)
                # i, j = 0, max(0, (tosize - new_w) // 2)
                i, j = 0, max(0, (new_w - tosize) // 2)
            else:
                new_h, new_w = int(tosize * h / w), tosize
                # i, j = max(0, (tosize - new_h) // 2), 0
                i, j = (new_h - tosize ) // 2, 0
            smask.append(F.interpolate(item.unsqueeze(0).half(), size=(new_h, new_w), mode="nearest").squeeze(0)[:, i:i+tosize, j:j+tosize])
        
    return smask

pil_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])
        
def to_same_size(pil_im):
    return pil_transform(pil_im)

def collate(batch):
    images = [item['pil'] for item in batch]
    tensor = torch.stack([item['tensor'] for item in batch], dim=0).half()
    labels = torch.stack([item['gt'] for item in batch], dim=0)
    return images, tensor, labels

def collate_dict(batch):
    batch_ = {'fea':torch.stack([item['fea'] for item in batch], dim=0),
              'pred':torch.stack([item['pred'] for item in batch], dim=0),
              'smask':[item['smask'] for item in batch],
              'sfea':[item['sfea'] for item in batch],
              'sood':torch.stack([item['sood'] for item in batch], dim=0)} 
    return None, batch_, torch.stack([item['gt'] for item in batch], dim=0)


class FeatureLoader(Dataset):
    def __init__(self, root, debug=False):
        self.data = {'fea':[],
                     'pred':[],
                     'smask':[],
                     'sfea':[],
                     'sood':[]}
        self.gt = []
        self.lengths = 0
        
        # fnames = [path for path in glob.glob(root+'*.h5')]
        # fnames = [root + 'data_00.npz'] if debug else [path for path in glob.glob(root + 'data_*.npz')]
        fnames = [path for path in glob.glob(root + 'data_*.npz')]
        fnames.sort()
        for path in fnames:
            # with h5py.File(path, 'r') as f:
            #     self.lengths += f['data'].shape[0]
            #     self.data.append(torch.from_numpy(f['data']).half())
            #     self.gt.append(torch.from_numpy(f['gt']).long())
            f = np.load(path)
            self.lengths += f['data'].shape[0]
            self.data['fea'].append(torch.from_numpy(f['data']).half())
            self.data['pred'].append(torch.from_numpy(f['pred']).long())
            self.gt.append(torch.from_numpy(f['label']).long())
        self.data['fea'] = torch.cat(self.data['fea'], dim=0)
        self.data['pred'] = torch.cat(self.data['pred'], dim=0)
        self.gt = torch.cat(self.gt, dim=0)
        # assert len(self.gt) == len(glob.glob(root.replace('h5', '')+'*/*.*'))
        
        # fnames = [root + 'seg_mask_00.npz'] if debug else [path for path in glob.glob(root + 'seg_mask_*.npz')]
        fnames = [path for path in glob.glob(root + 'seg_mask_*.npz')]
        fnames.sort()
        [self.data['smask'].extend(load_all_masks_with_offsets(path)) for path in fnames]
        self.data['smask'] = to_same_type(self.data['smask'])
        
        # fnames = [root + 'seg_fea_00.npz'] if debug else [path for path in glob.glob(root + 'seg_fea_*.npz')]
        fnames = [path for path in glob.glob(root + 'seg_fea_*.npz')]
        fnames.sort()
        [self.data['sfea'].extend(load_all_fea_with_offsets(path)) for path in fnames]
        
        # fnames = [root + 'seg_available_00.npz'] if debug else [path for path in glob.glob(root + 'seg_available_*.npz')]
        fnames = [path for path in glob.glob(root + 'seg_available_*.npz')]
        fnames.sort()
        self.data['sood'] = torch.cat([torch.from_numpy(np.load(path)['data'].astype(bool)) for path in fnames], dim=0)

    def __len__(self):
        return self.lengths
    
    def get_class_num(self):
        return len([c for c in np.unique(np.array(self.gt)).tolist() if c>-1])

    def __getitem__(self, idx):
        return {'gt':self.gt[idx],
                'fea':self.data['fea'][idx],
                'pred':self.data['pred'][idx],
                'smask':self.data['smask'][idx],
                'sfea':self.data['sfea'][idx],
                'sood':self.data['sood'][idx]} 


def load_data(which_dataset='cifar10', data_source='feature', save_by_class=False, hard_id=False, batch_size='16', test_batch_size="1", valid_split=0.2, n_job=20, subset=-1, debug=False, multi_crop=False, datasetname_prefix='', coreset_idx=None, datasetname_prefix_=None):
    preprocess_prefix = '-BICUBIC'
    from utils.utils_1k import MEAN, STD, ALL_LABEL, NUM_CLASSES
    if datasetname_prefix_ is None: datasetname_prefix_=datasetname_prefix
    tosize = 32 if 'cifar' in which_dataset else 224
    if data_source!='feature':
        img_transform = [
            transforms.Resize(tosize),
            transforms.CenterCrop(tosize),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['clip'], std=STD['clip'])
        ]
        transform = transforms.Compose(img_transform.copy())
        if multi_crop:
            img_transform = MultiCrop(multi_crop)
        else:
            img_transform = transforms.Compose(img_transform)

    test_name = '/objectnet' if hard_id else '/test'
    train_name = '/train'
    val_name = '/val'
    if which_dataset == 'cifar10':
        dataset_name = 'CIFAR10-r'
    elif which_dataset == 'cifar100':
        dataset_name = 'CIFAR100-r'
    elif which_dataset == 'imagenet100':
        dataset_name = 'ImageNet100'
    elif which_dataset == 'imagenet54':
        dataset_name = 'ImageNet54-r'
    elif which_dataset == 'imagenet104':
        dataset_name = 'ImageNet104-r'
        train_name = '/traincore200'
    elif which_dataset == 'imagenet1000':
        dataset_name = 'ImageNet1000'
        train_name = '/train'
        test_name = '/objectnet' if hard_id else '/test'
    elif which_dataset == 'imagenet896':
        dataset_name = 'ImageNet896-r'
    elif 'cub' in which_dataset:
        dataset_name = 'CUB200-2011/idsplit'
    else:
        raise 'specify which_dataset from: [cifar10/cifar100/imagenet100/domainnetl/domainnets/domainnet]'
        
    if data_source =='processed':
        val_set, dataset, dataset_, test_set = [], [], [], []
        for cls_name in os.listdir(PREFIX + dataset_name + train_name) if save_by_class else ['*']:
            train_idx = glob.glob(PREFIX + dataset_name + train_name + "/"+cls_name+"/*.*")
            if valid_split:
                train_idx, valid_idx = [], []
                for i in [cls_name] if save_by_class else os.listdir(PREFIX + dataset_name + train_name + "/"):
                    indices = glob.glob(PREFIX + dataset_name + train_name + "/" + i + "/*.*")
                    split = int(np.floor(valid_split * len(indices)))
                    # if which_dataset == 'imagenet104':
                    #     train_idx += indices
                    # else:
                    train_idx += indices[split:]
                    valid_idx += indices[:split]

            val_set.append(Loader(valid_idx, transform, subset=subset, ALL_LABEL=ALL_LABEL))
            dataset.append(Loader(train_idx, img_transform, subset=subset, ALL_LABEL=ALL_LABEL))
            f, g = dataset[-1].get()
            dataset_.append(FromLoader(f, g, transform, coreset_idx=coreset_idx))
            test_set.append(Loader(glob.glob(PREFIX + dataset_name + test_name + "/"+cls_name+"/*.*"), transform, subset=subset, ALL_LABEL=ALL_LABEL))
            # test_set.append(Loader(glob.glob(PREFIX + 'clipsegdemo' + test_name + "/"+cls_name+"/*.*"), transform, subset=subset, ALL_LABEL=ALL_LABEL)) # for test
        if not save_by_class:
            val_set, dataset, dataset_, test_set = val_set[0], dataset[0], dataset_[0], test_set[0]
                
    if data_source =='feature':
        if 'imagenet' in which_dataset:
            dataset_ =FeatureLoader(PREFIX + dataset_name + train_name + datasetname_prefix+'_noaugh5/', debug=debug)
            dataset = dataset_
            test_set =FeatureLoader(PREFIX + dataset_name + test_name + datasetname_prefix_+'h5/', debug=debug)
            out_datasets = [FeatureLoader(os.path.join(PREFIX, 'OOD_dataset/SUNh5/'+which_dataset+datasetname_prefix_+'/'), debug=debug),
                            FeatureLoader(os.path.join(PREFIX, f'OOD_dataset/Placesh5/'+which_dataset+datasetname_prefix_+'/'), debug=debug),
                            FeatureLoader(os.path.join(PREFIX, 'OOD_dataset/iNaturalisth5/'+which_dataset+datasetname_prefix_+'/'), debug=debug),
                            FeatureLoader(os.path.join(PREFIX, 'OOD_dataset/dtd/imagesh5/'+which_dataset+datasetname_prefix_+'/'), debug=debug)]
        collate_fn = collate_dict
    else:
        collate_fn = collate
        if 'cifar' in which_dataset:
            out_datasets = [SVHN(root=os.path.join(PREFIX, 'OOD_dataset/SVHN'), transform=transform),
                            Loader(glob.glob(os.path.join(PREFIX, f'OOD_dataset/Places/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, 'OOD_dataset/LSUN/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, 'OOD_dataset/iSUN/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, 'OOD_dataset/dtd/images/*/*.*')), transform, subset=subset)]

        elif 'imagenet' in which_dataset:
            out_datasets = [Loader(glob.glob(os.path.join(PREFIX, f'OOD_dataset/SUN/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, f'OOD_dataset/Places/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, f'OOD_dataset/iNaturalist/*/*.*')), transform, subset=subset),
                            Loader(glob.glob(os.path.join(PREFIX, f'OOD_dataset/dtd/images/*/*.*')), transform, subset=subset)]
    
    if 'imagenet' in which_dataset:
        out_datasets = [torch.utils.data.Subset(out_dataset, np.random.choice(len(out_dataset), 5000, replace=False))
                        if len(out_dataset) > 5000 else out_dataset for out_dataset in out_datasets]
        if data_source == 'feature':
            [np.random.choice(datalength, 5000, replace=False) for datalength in [10000, 328500, 10000, 5640]]
    class_nums = None if save_by_class else dataset.get_class_num()
    
    if debug:
        # val_set = torch.utils.data.Subset(val_set, np.random.choice(len(val_set), batch_size*2, replace=False))
        dataset_ = torch.utils.data.Subset(dataset_, np.random.choice(len(dataset_), batch_size*2, replace=False))
        test_set = torch.utils.data.Subset(test_set, np.random.choice(len(test_set), test_batch_size*2, replace=False))
        out_datasets = [torch.utils.data.Subset(out_dataset, np.random.choice(len(out_dataset), test_batch_size*2, replace=False)) for out_dataset in out_datasets]
        class_nums = NUM_CLASSES[which_dataset] 
    
    test_set_ = [torch.utils.data.DataLoader(out_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0 if which_dataset=='imagenet1000' else n_job, collate_fn=collate_fn) for out_dataset in out_datasets]
    
    if save_by_class:
        val_dataloader = None
        # [(torch.utils.data.DataLoader(
        #     s,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=n_job,
        #     collate_fn = collate_fn
        # ) if valid_split else None) for s in val_set]

        train_dataloader_ = [torch.utils.data.DataLoader(
            s,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=n_job,
            collate_fn = collate_fn
        ) for s in dataset_]
        test_dataloader = [torch.utils.data.DataLoader(
            s,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=n_job,
            collate_fn=collate_fn
        ) for s in test_set]
    else:
        val_dataloader = None
        train_dataloader_ = torch.utils.data.DataLoader(
            dataset_,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,#n_job,
            collate_fn = collate_fn
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,# if which_dataset=='imagenet1000' else n_job,
            collate_fn=collate_fn
        )

    if 'imagenet' in which_dataset:
        path = {'train': PREFIX + dataset_name + train_name + datasetname_prefix + 'h5/',
                'trainnoaug': PREFIX + dataset_name + train_name + datasetname_prefix + '_noaugh5/',
                'test': PREFIX + dataset_name + test_name + datasetname_prefix_ + 'h5/',
                # 'test': PREFIX + 'clipsegdemo' + test_name + 'h5/',
                'val': PREFIX + dataset_name + val_name + datasetname_prefix_ + 'h5/',
                'ood': [
                    os.path.join(PREFIX, 'OOD_dataset/SUNh5/'+which_dataset+datasetname_prefix_),
                    os.path.join(PREFIX, f'OOD_dataset/Placesh5/'+which_dataset+datasetname_prefix_),
                    os.path.join(PREFIX, 'OOD_dataset/iNaturalisth5/'+which_dataset+datasetname_prefix_),
                    os.path.join(PREFIX, 'OOD_dataset/dtd/imagesh5/'+which_dataset+datasetname_prefix_)]
                }

    return (None if save_by_class else len(dataset), class_nums, path), val_dataloader, train_dataloader_, test_dataloader, test_set_






def load_counterfact(data_source='feature', test_batch_size="1", ):
    from utils.utils_1k import MEAN, STD, ALL_LABEL
    if data_source!='feature':
        img_transform = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['clip'], std=STD['clip'])
        ]
        img_transform = transforms.Compose(img_transform.copy())

    test_name = '/test'
    ood_name = '/ood'
    train_name = '/train'
    dataset_name = 'counterfact'
        
    if data_source =='processed':
        train_idx = glob.glob(PREFIX + dataset_name + train_name + "/*/*.*")
        dataset = Loader(train_idx, img_transform, ALL_LABEL=ALL_LABEL)
        f, g = dataset.get()
        dataset_ = FromLoader(f, g, img_transform)
        test_set=Loader(glob.glob(PREFIX + dataset_name + test_name + "/*/*.*"), img_transform, ALL_LABEL=ALL_LABEL)
        collate_fn = collate
        out_datasets = [Loader(glob.glob(PREFIX + dataset_name + ood_name + "/*/*.*"), img_transform, ALL_LABEL=ALL_LABEL)]
        
    elif data_source =='feature':
        dataset_ =FeatureLoader(PREFIX + dataset_name + train_name + '_noaugh5/')
        test_set =FeatureLoader(PREFIX + dataset_name + test_name + 'h5/')
        out_datasets = [FeatureLoader(PREFIX + dataset_name + ood_name + 'h5/')]
        collate_fn = collate_dict
    
    test_set_ = [torch.utils.data.DataLoader(out_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn) for out_dataset in out_datasets]
    
    train_dataloader_ = torch.utils.data.DataLoader(
        dataset_,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,#n_job,
        collate_fn = collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,# if which_dataset=='imagenet1000' else n_job,
        collate_fn=collate_fn
    )
    path = {'trainnoaug': PREFIX + dataset_name + train_name  + '_noaugh5/',
            'test': PREFIX + dataset_name + test_name  + 'h5/',
            'ood': [PREFIX + dataset_name + ood_name  + 'h5/']
                }

    return path, train_dataloader_, test_dataloader, test_set_
    
    
def save_npz(path, mark, fea, gt, pred, availableseg=None, maskseg=None, feaseg=None):
    fea, gt, pred = np.array(fea), np.array(gt), np.array(pred)
    if not os.path.exists(os.path.join(path, f"data_{mark}.npz")):
        np.savez(os.path.join(path, f"data_{mark}.npz"), data=fea, label=gt, pred=pred)
    if maskseg is not None and not os.path.exists(os.path.join(path, f"seg_mask_{mark}.npz")):
        np.savez_compressed(
            os.path.join(path, f"seg_mask_{mark}.npz"),
            packed=np.packbits(np.concatenate([m.astype(np.uint8).ravel() for m in maskseg]), bitorder='big'),
            shape= np.array([m.shape for m in maskseg], dtype=np.int64),
            offsets=np.cumsum([0] + [m.size for m in maskseg] , dtype=np.int64))
    if feaseg is not None and not os.path.exists(os.path.join(path, f"seg_fea_{mark}.npz")):
        np.savez_compressed(
            os.path.join(path, f"seg_fea_{mark}.npz"),
            data=np.vstack(feaseg),
            offsets=np.cumsum([0] + [arr.shape[0] for arr in feaseg]))
    if availableseg is not None and not os.path.exists(os.path.join(path, f"seg_available_{mark}.npz")):
        np.savez_compressed(
            os.path.join(path, f"seg_available_{mark}.npz"),
            data=np.array(availableseg))


def load_all_masks_with_offsets(fn: str) -> list[torch.tensor]:
    data = np.load(fn)
    packed = data['packed']      
    shapes = data['shape']      #(Ndim)
    offsets = data['offsets']    # 

    all_bits = np.unpackbits(packed, bitorder='big')

    result = []
    for i in range(len(shapes)):
        start, end = offsets[i], offsets[i+1]
        flat = all_bits[start:end]
        arr = torch.from_numpy(flat.reshape(shapes[i]).astype(bool))
        result.append(arr)

    return result

def load_all_fea_with_offsets(fn: str) -> list[torch.tensor]:
    data = np.load(fn)
    feas = data['data']
    offsets = data['offsets'] 

    return [torch.from_numpy(feas[offsets[i]:offsets[i+1]]).half() for i in range(len(offsets)-1)]