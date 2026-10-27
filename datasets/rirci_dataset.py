"""Dataset reader for the RIRCI visible watermark removal experiments.

The reader expects the layout that produced the numbers in the paper. Every
sample is described by five images that share the same file stem:

    <dataset_dir>/
        train/  (or valid/, test/)
            watermarked/<stem>.png  J = A * W + (1 - A) * I
            gt/<stem>.png           I   (watermark-free image)
            mask/<stem>.png         M   (0/255 watermark region)
            cleancomp/<stem>.png    C_b = (1 - A) * I
            alpha/<stem>.png        A   (watermark opacity, 0-255)

All five folders are required, because the training loss is split by watermark
opacity.
"""

import os
import os.path as osp

import torch
from PIL import Image
from torchvision import transforms


class RIRCIDataset(torch.utils.data.Dataset):
    """Reader for the HWVOC / PW style visible watermark splits."""

    # Split argument -> directory name under <dataset_dir>.
    _SPLITS = {'train': 'train', 'val': 'valid', 'test': 'test'}
    _SUBDIRS = ('watermarked', 'gt', 'mask', 'cleancomp', 'alpha')
    _IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, is_train, args, ids_list=None, mask_list=None):
        if is_train not in self._SPLITS:
            raise ValueError(
                "Unknown split '{}', expected one of {}".format(
                    is_train, sorted(self._SPLITS)))

        self.is_train = is_train
        args.is_train = is_train == 'train'
        self.args = args
        self.root = osp.join(args.dataset_dir, self._SPLITS[is_train])
        if not osp.isdir(self.root):
            raise FileNotFoundError(
                "Split directory not found: {}".format(self.root))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.paths = self._index(self.root)
        self.ids = sorted(self.paths['gt'])

    @classmethod
    def _index_folder(cls, folder):
        files = {}
        for fname in sorted(os.listdir(folder)):
            stem, ext = osp.splitext(fname)
            if ext.lower() in cls._IMAGE_EXT:
                files[stem] = osp.join(folder, fname)
        if not files:
            raise FileNotFoundError(
                "No images found in {} (supported extensions: {}).".format(
                    folder, ', '.join(cls._IMAGE_EXT)))
        return files

    def _index(self, root):
        paths = {}
        for name in self._SUBDIRS:
            folder = osp.join(root, name)
            if not osp.isdir(folder):
                raise FileNotFoundError(
                    "Missing '{}' folder under {}.".format(name, root))
            paths[name] = self._index_folder(folder)

        reference = set(paths['gt'])
        for name in self._SUBDIRS:
            missing = reference - set(paths[name])
            if missing:
                raise ValueError(
                    "The '{}' folder is not aligned with 'gt': missing {}".format(
                        name, sorted(missing)[:5]))
        return paths

    @staticmethod
    def _load(path, mode):
        with Image.open(path) as img:
            return img.convert(mode)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        sample = {
            'image': self._load(self.paths['watermarked'][stem], 'RGB'),
            'target': self._load(self.paths['gt'][stem], 'RGB'),
            'comp': self._load(self.paths['cleancomp'][stem], 'RGB'),
            'mask': self._load(self.paths['mask'][stem], 'L'),
            'alpha': self._load(self.paths['alpha'][stem], 'L'),
        }
        out = {key: self.transform(value) for key, value in sample.items()}
        out['img_path'] = self.paths['watermarked'][stem]
        return out
