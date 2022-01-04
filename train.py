import os

import numpy as np
import torch
import torch.nn as nn

import dataset
import model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, ckpt_dir='./checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_dir = ckpt_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optim, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optim, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')
            
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        torch.save({'net': model.state_dict(),
                'optim': optim.state_dict()},
                "%s/model_epoch%d.pth" % (self.ckpt_dir, epoch))

        self.val_loss_min = val_loss

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(args):
    if args.mode=="train":
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        
        ckpt_dir = args.ckpt_dir
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        dataset_full = dataset.Dataset(data_dir=args.data_dir, type="train")

        batch_size = args.batch_size

        num_data = len(dataset_full)
        num_data_train = num_data // 100 * (100 - args.split_rate)
        num_batch_train = np.ceil(num_data_train / batch_size)

        dataset_train, dataset_val = torch.utils.data.random_split(dataset_full, [num_data_train, num_data-num_data_train])
        loader_train = torch.utils.data.DataLoader(dataset_train,
                                batch_size=batch_size,
                                shuffle=True, num_workers=0)
        loader_val = torch.utils.data.DataLoader(dataset_val,
                                batch_size=batch_size,
                                shuffle=True, num_workers=0)

        net = model.ResNetRegressor(out_channels=2).to(device)
        criterion = nn.MSELoss(reduction="mean").to(device)
        optim = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))

        init_epoch = 0
        early_stop = EarlyStopping(ckpt_dir=ckpt_dir, trace_func=print)
        for epoch in range(init_epoch + 1, 50):
            net.train()
            loss_train = []
            val_data = next(iter(loader_val))
            val_input = val_data["image"].to(device)
            val_target = val_data["label"].to(device)

            for batch, data in enumerate(loader_train, 1):
                image = data["image"].to(device)
                label = data["label"].to(device)

                output = net(image)

                set_requires_grad(net, True)
                optim.zero_grad()

                loss = criterion(output, label)
                loss.backward()
                optim.step()

                loss_train += [float(loss.item())]

            print("TRAIN: EPOCH %04d / %04d | "
                "LOSS %.8f | \n"%
                (epoch, 50, np.mean(loss_train)))
                
            # forward netP
            with torch.no_grad():
                net.eval()
                val_output = net(val_input)
                
                # Early stop when validation loss does not reduce
                val_loss = criterion(val_output, val_target)
                early_stop(val_loss=val_loss, model=net, optim=optim, epoch=epoch)
                
            if early_stop.early_stop:
                break