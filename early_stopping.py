import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.epoch = 1

    def __call__(self, val_loss, state_dict, epoch):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, state_dict)
        elif score > self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            os.remove(os.path.join(self.save_path, f'early_stopping_model_{str(self.epoch).zfill(4)}.tar'))
            self.epoch = epoch
            self.save_checkpoint(val_loss, state_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, state_dict):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(state_dict, os.path.join(self.save_path, f'early_stopping_model_{str(self.epoch).zfill(4)}.tar'))# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

