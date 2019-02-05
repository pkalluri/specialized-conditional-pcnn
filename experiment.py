"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import json
import os

import torch

import data
import losses
import model
import train
from vis import generate_between_classes
import datetime

def run(dataset='mnist', n_samples=50000, n_bins=4,
        n_features=200, batch_size=64, n_layers=6,
        loss='standard', optimizer='adam', learnrate=1e-4, dropout=0.9, max_epochs=35, cuda=True, resume=False,
        exp_dir='out', note=''):

    # Data
    if dataset=='mnistog': train_data, val_data, onehot_fcn, n_classes = data.get_loaders('mnist', batch_size)
    if dataset=='mnist': train_data, val_data, onehot_fcn, n_classes = data.get_sorted_data('mnist', batch_size)

    if not resume:
        # Make dir
        exp_name = datetime.datetime.now().strftime("%m_%d_%y-%H_%M_%S")
        exp_name += '_{}_{}samples_{}_{}'.format(dataset, n_samples, loss, note)
        print("Out directory: " + exp_name)
        exp_dir = os.path.join(os.path.expanduser(exp_dir), exp_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        # Store experiment params in params.json
        params = {'data':dataset, 'n_samples': n_samples, 'loss':loss, 'batch_size':batch_size, 'n_features':n_features,
                  'n_layers':n_layers, 'n_bins':n_bins, 'optimizer': optimizer,
                  'learnrate':learnrate, 'dropout':dropout, 'cuda':cuda, 'note':note}
        print("Params: "+str(params.items()))
        with open(os.path.join(exp_dir,'params.json'),'w') as f:
            json.dump(params,f)
        net = model.PixelCNN(1, n_classes, n_features, n_layers, n_bins, dropout)
    else:
        # if resuming, need to have params, stats and checkpoint files
        if not (os.path.isfile(os.path.join(exp_dir,'params.json'))
                and os.path.isfile(os.path.join(exp_dir,'stats.json'))
                and os.path.isfile(os.path.join(exp_dir,'last_checkpoint'))):
            raise Exception('Missing param, stats or checkpoint file on resume')
        net = torch.load(os.path.join(exp_dir, 'last_checkpoint'))

    # Define loss fcn, incl. label formatting from input
    def input2label(x):
        return torch.squeeze(torch.round((n_bins-1)*x).type(torch.LongTensor),1)
    loss_fcns = {'official': losses.official_loss_function,
                 'standard': losses.standard_loss_function,
                 'sum': losses.sum_loss_function,
                 'min': losses.min_loss_function,
                 'debug': torch.nn.NLLLoss()}
    loss_fcn = loss_fcns[loss]

    # Train
    train.fit(train_data, val_data, n_samples, net, exp_dir, input2label, loss_fcn,
              onehot_fcn, n_classes, optimizer, learnrate=learnrate, cuda=cuda,
              max_epochs=max_epochs, resume=resume)

    # # Generate some between-class examples
    # generate_between_classes(net, [28, 28], [1, 7],
    #                          os.path.join(exp_dir,'1-7.jpeg'), n_classes, cuda)
    # generate_between_classes(net, [28, 28], [3, 8],
    #                          os.path.join(exp_dir,'3-8.jpeg'), n_classes, cuda)
    # generate_between_classes(net, [28, 28], [4, 9],
    #                          os.path.join(exp_dir,'4-9.jpeg'), n_classes, cuda)
    # generate_between_classes(net, [28, 28], [5, 6],
    #                          os.path.join(exp_dir,'5-6.jpeg'), n_classes, cuda)

debug = True

if debug:
    run(dataset='mnistog', n_samples=64, n_bins=4, n_features=200, batch_size=64,
        n_layers=6, loss='official', optimizer='adam', learnrate=1e-4, dropout=0.9, max_epochs=3, cuda=True, resume=False,
        exp_dir='out', note='debug')
else:
    run(dataset='mnist', n_samples=50000, n_bins=4, n_features=200, batch_size=64,
        n_layers=6, loss='standard', optimizer='adam', learnrate=1e-4, dropout=0.9, max_epochs=1000000, cuda=True, resume=False,
        exp_dir='out', note='less_data_to_completion')