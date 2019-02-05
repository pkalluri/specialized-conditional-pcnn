"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import os

import numpy as np
import torch
import torchvision
import torch.utils as utils
from torchvision.datasets.mnist import read_label_file, read_image_file



def get_onehot_fcn(n_classes):
    def onehot_fcn(x):
        y = np.zeros((n_classes), dtype='float32')
        y[x] = 1
        return y
    return onehot_fcn


def augment(rotate=5):
    return torchvision.transforms.Compose([torchvision.transforms.RandomRotation(rotate),
                               torchvision.transforms.ToTensor()])

class SortedDataset(utils.data.Dataset):
    def __init__(self, unsorted_dataset, n_classes, onehot_fcn, batch_size, num_workers, pin_memory):
        self.n_classes = n_classes
        self.onehot_fcn = onehot_fcn
        self.batch_size = batch_size
        # map x to relevant ys
        all_y, all_x = next(iter(utils.data.DataLoader(unsorted_dataset, batch_size=len(unsorted_dataset))))
        self.x2ys = {}
        n_batches = 0
        for x in range(self.n_classes):
            ys = all_y[all_x == x]
            n_batches += int(len(ys)/batch_size)
            dataset = utils.data.TensorDataset(ys)
            self.x2ys[x] = utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.n_batches = n_batches
    def __len__(self):
        return self.n_batches
    def __iter__(self):
        # reset iterators
        self.x2ys_iterable = {}
        for x, ys in self.x2ys.items():
            self.x2ys_iterable[x] = iter(ys)
        # start iterating
        try:
            for idx in range(self.__len__()):
                x = idx % self.n_classes
                Y = next(self.x2ys_iterable[x])[0]
                yield Y, torch.tensor([self.onehot_fcn(x)]).expand(Y.shape[0],self.n_classes)
        except StopIteration: # TODO: instead, randomly pick x, delete x from iterator dict if next is empty
            pass

def get_sorted_data(dataset_name, batch_size, n_workers=8):
    assert dataset_name.lower() in ['mnist', 'emnist', 'fashionmnist']
    dataset_args = {'root':os.path.join('data', dataset_name.lower()), # TODO put data dir inside project dir
                    'download':True,
                    'transform':torchvision.transforms.ToTensor()}
    if dataset_name.lower()== 'mnist':
        dataset_init = torchvision.datasets.MNIST
        n_classes = 10
    elif dataset_name.lower()== 'emnist':
        dataset_init = torchvision.datasets.EMNIST
        n_classes = 37
        dataset_args.update({'split':'letters'})
    else:
        dataset_init = torchvision.datasets.FashionMNIST
        n_classes = 10
    onehot_fcn = get_onehot_fcn(n_classes)

    unsorted_val_data = dataset_init(train=False, **dataset_args)
    sorted_val_data = SortedDataset(unsorted_val_data, n_classes=n_classes, onehot_fcn=onehot_fcn,
                                    batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    dataset_args['transform'] = augment()
    unsorted_train_data = dataset_init(train=True, **dataset_args)
    sorted_train_data = SortedDataset(unsorted_train_data, n_classes=n_classes, onehot_fcn=onehot_fcn,
                                      batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    return sorted_train_data, sorted_val_data, onehot_fcn, n_classes

def get_loaders(dataset, batch_size, n_workers=8):
    assert dataset.lower() in ['mnist','emnist','fashionmnist']

    loader_args = {'batch_size':batch_size,
                   'num_workers':n_workers,
                   'pin_memory':True}
    datapath = os.path.join(os.getenv('HOME'), 'data', dataset.lower())
    dataset_args = {'root':datapath,
                    'download':True,
                    'transform':torchvision.transforms.ToTensor()}

    if dataset.lower()=='mnist':
        dataset_init = torchvision.datasets.MNIST
        n_classes = 10
    elif dataset.lower()=='emnist':
        dataset_init = torchvision.datasets.EMNIST
        n_classes = 37
        dataset_args.update({'split':'letters'})
    else:
        dataset_init = torchvision.datasets.FashionMNIST
        n_classes = 10
    onehot_fcn = get_onehot_fcn(n_classes)
    dataset_args.update({'target_transform':onehot_fcn})

    val_loader = torch.utils.data.DataLoader(
        dataset_init(train=False, **dataset_args), shuffle=False, **loader_args)

    dataset_args['transform'] = augment()
    train_loader = torch.utils.data.DataLoader(
        dataset_init(train=True, **dataset_args), shuffle=True, **loader_args)

    return train_loader, val_loader, onehot_fcn, n_classes


# # Note: Can't build master ver of pytorch/torchvision, so copied this here
# class EMNIST(torchvision.datasets.MNIST):
#     """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.
#     Args:
#         root (string): Root directory of dataset where ``processed/training.pt``
#             and  ``processed/test.pt`` exist.
#         split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
#             ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
#             which one to use.
#         train (bool, optional): If True, creates dataset from ``training.pt``,
#             otherwise from ``test.pt``.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """
#     url = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
#     splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
#
#     def __init__(self, root, split, **kwargs):
#         if split not in self.splits:
#             raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
#                 split, ', '.join(self.splits),
#             ))
#         self.split = split
#         self.training_file = self._training_file(split)
#         self.test_file = self._test_file(split)
#         super(EMNIST, self).__init__(root, **kwargs)
#
#     def _training_file(self, split):
#         return 'training_{}.pt'.format(split)
#
#     def _test_file(self, split):
#         return 'test_{}.pt'.format(split)
#
#     def download(self):
#         """Download the EMNIST data if it doesn't exist in processed_folder already."""
#         import errno
#         from six.moves import urllib
#         import gzip
#         import shutil
#         import zipfile
#
#         if self._check_exists():
#             return
#
#         # download files
#         try:
#             os.makedirs(os.path.join(self.root, self.raw_folder))
#             os.makedirs(os.path.join(self.root, self.processed_folder))
#         except OSError as e:
#             if e.errno == errno.EEXIST:
#                 pass
#             else:
#                 raise
#
#         print('Downloading ' + self.url)
#         data = urllib.request.urlopen(self.url)
#         filename = self.url.rpartition('/')[2]
#         raw_folder = os.path.join(self.root, self.raw_folder)
#         file_path = os.path.join(raw_folder, filename)
#         with open(file_path, 'wb') as f:
#             f.write(data.read())
#
#         print('Extracting zip archive')
#         with zipfile.ZipFile(file_path) as zip_f:
#             zip_f.extractall(raw_folder)
#         os.unlink(file_path)
#         gzip_folder = os.path.join(raw_folder, 'gzip')
#         for gzip_file in os.listdir(gzip_folder):
#             if gzip_file.endswith('.gz'):
#                 print('Extracting ' + gzip_file)
#                 with open(os.path.join(raw_folder, gzip_file.replace('.gz', '')), 'wb') as out_f, \
#                         gzip.GzipFile(os.path.join(gzip_folder, gzip_file)) as zip_f:
#                     out_f.write(zip_f.read())
#         shutil.rmtree(gzip_folder)
#
#         # process and save as torch files
#         for split in self.splits:
#             print('Processing ' + split)
#             training_set = (
#                 read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
#                 read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
#             )
#             test_set = (
#                 read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
#                 read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
#             )
#             with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
#                 torch.save(training_set, f)
#             with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
#                 torch.save(test_set, f)
#
#         print('Done!')