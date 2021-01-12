import os
import random
from contextlib import redirect_stdout
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.utils import data
from pycocotools.coco import COCO
import numpy as np
from glob import glob
import pandas as pd


class CocoDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1., 1., 1.]
        self.training = training
        self.image = False

        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        self.ids = sorted(self.ids)
        if 'categories' in self.coco.dataset:
            self.categories_inv = { k: i for i, k in enumerate(self.coco.getCatIds()) }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        if self.coco:
            image = self.coco.loadImgs(id)[0]['file_name']
        im = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im.size)
        if ratio * max(im.size) > self.max_size:
            ratio = self.max_size / max(im.size)
        im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))

        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().view(*im.size[::-1], len(im.mode))

        data = data.permute(2, 0, 1)
        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        """
        # debug
        if id == 139:
            print('id: {}, shape: {}'.format(id, data.shape))
            np.save(os.path.join('/workspace/retinanet/debug', '{}.npy'.format(id)), data.cpu().numpy())
        """

        return data, id, ratio 

    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios

class VisDroneDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1., 1., 1.]
        self.training = training
        self.image = False

        self.images = sorted(os.listdir(self.path))
        self.ids = range(len(self.images))

        self.annotations = pd.read_csv(annotations, header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'])
        self.annotations['Width'] = self.annotations['Width'] + self.annotations['X']
        self.annotations['Height'] = self.annotations['Height'] + self.annotations['Y']

        visdrone_gt_mapping = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:0}
        self.annotations['ClassId'] = self.annotations['ClassId'].map(visdrone_gt_mapping)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        image = self.images[id]
        im0 = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im0.size)
        if ratio * max(im0.size) > self.max_size:
            ratio = self.max_size / max(im0.size)
        im = im0.resize((int(ratio * d) for d in im0.size), Image.BILINEAR)

        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))

        # imnp = cv2.imread('{}/{}'.format(self.path, image))
        imnp = np.asarray(im)

        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().view(*im.size[::-1], len(im.mode))

        data = data.permute(2, 0, 1)
        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        """
        # debug
        if id == 139:
            print('id: {}, shape: {}'.format(id, data.shape))
            np.save(os.path.join('/workspace/retinanet/debug', '{}.npy'.format(id)), data.cpu().numpy())
        """

        return data, id, ratio, imnp

    def _get_target(self, id):
        'Get annotations for sample'
        # raise NotImplementedError

        frame = self.annotations[self.annotations['FrameId'] == id]
        bboxes_xyxy = torch.Tensor(frame[['X', 'Y', 'Width', 'Height']].to_numpy())
        gtclasses = torch.Tensor(frame['ClassId'].to_numpy()).int()
        identities = torch.Tensor(frame['Id'].to_numpy())
        return bboxes_xyxy, identities, gtclasses

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios, imgs = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios, imgs

class DataIterator():
    'Data loader for data parallel'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False):
        self.resize = resize
        self.max_size = max_size

        print('Data loader for data parallel')
        self.dataset = VisDroneDataset(path, resize=resize, max_size=max_size,
            stride=stride, annotations=annotations, training=training)
        self.ids = self.dataset.ids
        # self.coco = self.dataset.coco
    
        self.sampler = data.distributed.DistributedSampler(self.dataset) if world > 1 else None
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size // world,
            sampler=self.sampler, collate_fn=self.dataset.collate_fn, num_workers=2, pin_memory=True)
        
    def __repr__(self):
        return '\n'.join([
            '    loader: pytorch',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return len(self.dataloader)
        
    def __iter__(self):
        for output in self.dataloader:
            if self.dataset.training:
                data, target = output
            else:
                data, ids, ratio, im = output

            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            if self.dataset.training:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                yield data, target
            else:
                if torch.cuda.is_available():
                    ids = ids.cuda(non_blocking=True)
                    ratio = ratio.cuda(non_blocking=True)
                yield data, ids, ratio, im
  
