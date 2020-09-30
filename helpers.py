import torch
import os


def distance(m1, m2):
    return (m1-m2)**2

def centerUpdate(centers, c_points, numClass):
    for i in range(numClass):
        centers[i] = c_points[i].mean(dim=0)

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, path='checkpoint.pt', trace_func=print):
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
        self.verbose = True
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1e32
        self.path = path
        self.trace_func = trace_func
        self.save_ep = None

    def __call__(self, val_loss, model, epoch, state):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, state)

        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, state)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, state):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if self.save_ep:
            os.remove(self.path + 'ep_%d.pth.tar' % self.save_ep)

        torch.save(state, self.path + 'ep_{}.pth.tar'.format(epoch+1))

        self.save_ep = epoch+1
        self.val_loss_min = val_loss