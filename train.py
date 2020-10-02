import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import argparse
from network import DenseLayer, FeedForwardNet
import trainer
from loss import *
from helpers import EarlyStopping
import os
from glob import glob


if __name__ == "__main__":

    # Example run: CUDA_VISIBLE_DEVICES=0 python train.py -e 50 -r 1e-3 -b 32 -l trl -n ff -d mnist -la 0 -p 5
    
    parser = argparse.ArgumentParser(description='Layerwise Training')

    parser.add_argument('-e', '--epoch', type=int, required=True, help='Epoch')
    parser.add_argument('-r', '--rate', type=float, required=True, help='Learning Rate')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch Size')
    parser.add_argument('-l', '--loss', type=str, required=True, help='1-Original Triplet Center, 2-Modified')
    parser.add_argument('-n', '--network', type=str, required=True, help='Network Model')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Choose dataset')
    parser.add_argument('-la', '--layer', type=int, required=True, help='Choose which layer to train.')
    parser.add_argument('-p', '--patience', type=int, required=True, help='Patience for early stopping (epochs)')
    parser.add_argument('-f', '--full', type=int, required=True, help='Use learned centers for init.')
    
    args = parser.parse_args()

    assert args.dataset in ['mnist', 'cifar10']
    assert args.network in ['ff', 'resnet', 'densenet']
    assert args.loss in ['trl', 'tcl', 'ce']

    BATCH_SIZE = args.batch
    EPOCH = args.epoch
    LAYER = args.layer
    LR = args.rate
    PATH = 'ep_{}_lr_{}_b_{}_l_{}_n_{}_d_{}_layer_{}_full_{}/'.format(EPOCH, LR, BATCH_SIZE, args.loss, args.network, args.dataset, args.layer, args.full)
    
    os.makedirs('models/' + PATH)
    
    if args.dataset == 'mnist':
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

        train_set = torchvision.datasets.MNIST('./', train=True, transform=transform, download=True)
        train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])
        test_set = torchvision.datasets.MNIST('./', train=False, transform=transform, download=True)

    elif args.dataset == 'cifar10':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ])

        train_set = torchvision.datasets.CIFAR10('./', train=True, transform=transform, download=True)
        train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])
        test_set = torchvision.datasets.CIFAR10('./', train=False, transform=transform, download=True) 


    CLASS_NUM = len(train_set.dataset.classes)
    ACTIVATION = 'sigmoid' if args.loss == 'trl' else 'tanh'

    if args.network == 'ff':
        
        network = FeedForwardNet(CLASS_NUM, ACTIVATION, args.dataset)
        
        if args.full:
            model = nn.Sequential(*(network.network()))
        else:
            model = network.network()[LAYER]
           
        DIM = network.dims[LAYER]
 
    elif args.network == 'resnet':
        pass

    elif args.network == 'densenet':
        pass
    
    
    net_length = len(network.network())
    
    assert LAYER < net_length, "ERROR: Choose a layer number smaller than {} to train".format(net_length)
    
    current_loss = args.loss
    last_layer_train = False
    
    if LAYER == net_length-1:
        args.loss = 'ce'
    

    if args.full:
        
        if LAYER == net_length-1:
            layers = []
            last_layer_train = True

            for i in range(LAYER):
                layers.append(network.network()[i])

                load_path = glob('models/*l_{}_n_{}_d_{}_layer_{}_full_0/*'.format(current_loss, args.network,
                                                                                   args.dataset, i))[0] 

                print("LOAD: " + load_path)
                load_dict = torch.load(load_path)
                layers[-1].load_state_dict(load_dict['state_dict'])
            
            layers.append(network.network()[-1])
            
            del model
            model = nn.Sequential(*layers)    
        
        LAYER = 0
        
   
    if args.loss == 'trl':
        centers = torch.from_numpy(np.eye(DIM)).type(torch.cuda.FloatTensor)
        loss_fn = expLoss

    elif args.loss == 'tcl':
        centers = torch.from_numpy(-np.ones((DIM,DIM)) + 2*np.eye(DIM)).type(torch.cuda.FloatTensor)
        loss_fn = tripletCenterLoss

    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
        centers = torch.from_numpy(np.eye(DIM)).type(torch.cuda.FloatTensor) # Dummy useless

    trainLoader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valLoader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    testLoader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

    # TRAINING 

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=args.patience, path='models/' + PATH)
    t_losses = []
    v_losses = []
    previous_model = None
    
    
    if LAYER > 0 and not last_layer_train:
        
        previous_layers = []

        for i in range(LAYER):
            previous_layers.append(network.network()[i])
            
            load_path = glob('models/*l_{}_n_{}_d_{}_layer_{}_full_{}/*'.format(current_loss,
                                                                                args.network,
                                                                                args.dataset, i,
                                                                                args.full))[0] 
            
            print("LOAD: " + load_path)
            load_dict = torch.load(load_path)
            previous_layers[-1].load_state_dict(load_dict['state_dict'])

        previous_model = nn.Sequential(*previous_layers)
        
    

    cfg = {
        'm': np.sqrt(DIM) if args.loss == 'tcl' else 0,
        'epoch': EPOCH,
        'class_num': CLASS_NUM,
        'layer': LAYER,
        'previousModel': previous_model,
        'loss': args.loss
    }

    loader = {
        'train': trainLoader,
        'val': valLoader
    }

    for ep in range(EPOCH):

        train_loss, val_loss = trainer.trainer(model, loader, optimizer, loss_fn, cfg, centers, ep)

        t_losses.append(train_loss)
        v_losses.append(val_loss)

        state = {
            'epoch': EPOCH, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': t_losses,
            'val_loss': v_losses,
            'dim': DIM,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'centers': centers
        }

        early_stopping(val_loss, model, ep, state)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    if not early_stopping.early_stop:
        torch.save(state, 'models/' + PATH + 'final.pth.tar'.format(EPOCH))

