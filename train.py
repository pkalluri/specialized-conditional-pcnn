"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import imageio
import json
import os
from PIL import Image
import time

import matplotlib;

from data import get_onehot_fcn

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
from skimage.transform import resize
import torch
from torchvision import utils
from torch.autograd import Variable

import losses
from vis import plot_stats, clearline, generate, tile_images

def generate_images(model,img_size,n_classes,onehot_fcn,cuda=True):
    y = np.array(list(range(min(n_classes,10)))*5)  # gpu mem limit
    y = np.concatenate([onehot_fcn(x)[np.newaxis,:] for x in y])
    return generate(model, img_size, y, cuda=cuda)


def plot_loss(train_loss, val_loss):
    fig = plt.figure(num=1, figsize=(4, 4), dpi=70, facecolor='w',
                     edgecolor='k')
    plt.plot(range(1,len(train_loss)+1), train_loss, 'r', label='training')
    plt.plot(range(1,len(val_loss)+1), val_loss, 'b', label='validation')
    plt.title('After %i epochs'%len(train_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.rcParams.update({'font.size':10})
    fig.tight_layout(pad=1)
    fig.canvas.draw()

    # now convert the plot to a numpy array
    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot = plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    plt.close(fig)
    return plot


def fit(train_data, val_data, n_samples, model, exp_path, label_preprocess, loss_fcn,
        onehot_fcn, n_classes=10, optimizer='adam', learnrate=1e-4, cuda=True,
        patience=10, max_epochs=200, resume=False, bgd=False):

    if cuda:
        model = model.cuda()
        loss_fcn = loss_fcn.cuda()

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = os.path.join(exp_path,'stats.json')

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(model.parameters(),lr=learnrate,momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    # load a single example from the iterator to get the image size
    x,y = next(iter(train_data))
    # utils.save_image(x,
    #                  os.path.join(os.getcwd(), 'sample.png'),
    #                  nrow=1, padding=2)
    img_size = list(x.numpy().shape[-2:])

    if not resume:
        stats = {'loss':{'train':[],'val':[]},
                 'mean_output':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
        generated = []
        plots = []
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])-1
        generated = list(np.load(os.path.join(exp_path,'generated.npy')))
        plots = list(np.load(os.path.join(exp_path,'generated_plots.npy')))
        print('Resuming from epoch %i'%start_epoch)

    def save_img(x,filename):
        Image.fromarray((255*x).astype('uint8')).save(filename)

    def epoch(dataloader,training,bgd):
        bar = ProgressBar()
        epoch_losses = []
        mean_outs = []
        sample_idx = 0
        accumulated_loss_dict = {}
        for x,y in bar(dataloader):
            label = label_preprocess(x)
            if cuda:
                x,y = x.cuda(),y.cuda()
                label = label.cuda()
            x,y = Variable(x),Variable(y)
            label = Variable(label)
            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            output = model(x,y)

            one_hot_digits = [np.argmax(one_hot).item() for one_hot in y]
            n_classes = y.shape[1]
            for i in range(n_classes):
                selector = torch.LongTensor([j for j in range(len(one_hot_digits)) if one_hot_digits[j] == i])
                # print(i, selector.nelement())
                if selector.nelement() == 0:
                    continue

                class_labels = torch.index_select(label, 0, selector)
                class_outputs = torch.index_select(output, 0, selector)
                class_batch_loss = loss_fcn(class_outputs, class_labels)
                # print(i, class_labels.shape, class_outputs.shape, class_batch_loss.shape)

                if bgd:
                    class_batch_loss = class_batch_loss.unsqueeze(0)
                    if i not in accumulated_loss_dict:
                        accumulated_loss_dict[i] = class_batch_loss
                    else:
                        accumulated_loss_dict[i] = torch.cat([accumulated_loss_dict[i], class_batch_loss])

                if training and not bgd:
                    class_batch_loss.backward()
                    optimizer.step()

                # track mean output
                class_outputs = class_outputs.data.cpu().numpy()
                mean_outs.append(np.mean(np.argmax(class_outputs,axis=1))/class_outputs.shape[1])
                epoch_losses.append(class_batch_loss.data.cpu().numpy())

            sample_idx += x.shape[0]
            if sample_idx >= n_samples:
                break

        if bgd:
            epoch_losses = None
            for k,v in accumulated_loss_dict.items():
                if isinstance(loss_fcn, losses.OfficialLossModule) or isinstance(loss_fcn, losses.StandardLossModule):
                    class_epoch_loss = torch.sum(v)
                elif isinstance(loss_fcn, losses.SumLossModule):
                    class_epoch_loss = torch.logsumexp(v)
                elif isinstance(loss_fcn, losses.MinLossModule):
                    class_epoch_loss = torch.min(v)
                else:
                    print("Unknown loss function [{}] for batch gradient descent, exiting.".format(type(loss_fcn)))
                    exit()

                print("Epoch class [{}] loss: {}".format(k, class_epoch_loss.item()))
                class_epoch_loss = class_epoch_loss.unsqueeze(0)
                if epoch_losses is None:
                    epoch_losses = class_epoch_loss
                else:
                    epoch_losses = torch.cat([epoch_losses, class_epoch_loss])

            epoch_loss = torch.sum(epoch_losses)
            if training:
                epoch_loss.backward()
                optimizer.step()

            return epoch_loss.item(), np.mean(mean_outs)
        else:
            # clearline()
            return float(np.mean(epoch_losses)), np.mean(mean_outs)

    for e in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        loss,mean_out = epoch(train_data, training=True, bgd=bgd)
        time_per_example = (time.time()-t0)/len(train_data)
        stats['loss']['train'].append(loss)
        stats['mean_output']['train'].append(mean_out)
        print(('Epoch %3i:    Training loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(e,loss,mean_out,time_per_example*1000))

        # Validation
        t0 = time.time()
        loss,mean_out = epoch(val_data, training=False, bgd=bgd)
        time_per_example = (time.time()-t0)/len(val_data)
        stats['loss']['val'].append(loss)
        stats['mean_output']['val'].append(mean_out)
        print(('            Validation loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(loss,mean_out,time_per_example*1000))

        # Generate images and update gif
        new_frame = tile_images(generate_images(model, img_size, n_classes,
                                                onehot_fcn, cuda=cuda))
        generated.append(new_frame)

        # Update gif with loss plot
        plot_frame = plot_loss(stats['loss']['train'],stats['loss']['val'])
        if new_frame.ndim==2:
            new_frame = np.repeat(new_frame[:,:,np.newaxis],3,axis=2)
        nw = int(new_frame.shape[1]*plot_frame.shape[0]/new_frame.shape[0])
        new_frame = resize(new_frame,[plot_frame.shape[0],nw], anti_aliasing=True, order=0, preserve_range=True, mode='constant')
        plots.append(np.concatenate((plot_frame.astype('uint8'),
                                     new_frame.astype('uint8')),
                                    axis=1))

        # Save gif arrays so it can resume training if interrupted
        np.save(os.path.join(exp_path,'generated.npy'),generated)
        np.save(os.path.join(exp_path,'generated_plots.npy'),plots)

        # Save stats and update training curves
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        torch.save(model,os.path.join(exp_path,'last_checkpoint'))
        if loss<best_val:
            best_val = loss
            stall = 0
            torch.save(model,os.path.join(exp_path,'best_checkpoint'))
            imageio.imsave(os.path.join(exp_path, 'best_generated.jpeg'),
                           generated[-1].astype('uint8'))
            imageio.imsave(os.path.join(exp_path, 'best_generated_plots.jpeg'),
                           plots[-1].astype('uint8'))
            imageio.mimsave(os.path.join(exp_path, 'generated.gif'),
                            np.array(generated), format='gif', loop=0, fps=2)
            imageio.mimsave(os.path.join(exp_path, 'generated_plot.gif'),
                            np.array(plots), format='gif', loop=0, fps=2)
        else:
            stall += 1
        if stall>=patience:
            break






