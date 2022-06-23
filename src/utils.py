import json
import os, sys
import pathlib
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

from dataset import *
from validation import validation_multi


def seed_everything(seed=3407):
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out


def train_test_split(num, val_ratio):
    indices = list(range(num))
    split = int(np.floor(val_ratio * num))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def get_time_suffix():
    ''' get current time, return string in format: year_month_day, e.g., 2021_11_11 '''
    time_struct = time.localtime(time.time())
    year, month, day = time_struct.tm_year, time_struct.tm_mon, time_struct.tm_mday
    time_suffix = str(year) + '_' + str(month) + '_' + str(day)
    return time_suffix


def set_model_saved_path(args):
    time_suffix = get_time_suffix()
    if args.train_dataset == 'Endo18_train':
        save_model_path = 'saved_model/' + args.method + '/' + args.train_dataset + '/' + args.val_dataset
    elif args.train_dataset == 'Blend':
        save_model_path = 'saved_model/' + args.method + '/' + args.train_dataset + '/' + args.val_dataset + '/' + args.blend_mode
    if args.augmix != 'None':
        save_model_path = save_model_path + '/augmix_' + args.augmix + '_L' + str(args.augmix_level)
    if args.save_model == 'True':
        if not os.path.isdir(save_model_path):
            print('==> Model will be saved to:', save_model_path)
            pathlib.Path(save_model_path).mkdir(parents=True, exist_ok=True)
        else:
            print("==> WARNING: The same model path exists!")
            sys.exit()
    else:
        print('==> Model will not be saved.')
    
    
    return save_model_path


def train(args, model, criterion, train_loader, valid_loader, validation, optimizer, model_path, test_loader=None):

    valid_losses = []
    save_model_path = model_path
    best_dice, best_epoch_dice = 0.0, 0
    start_epoch = 0

    print('==> Training started.')

    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        model.train()

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        for _, (inputs, targets) in enumerate(train_loader):

            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()

            tq.update(batch_size)
        tq.close()
        print('Train loss:', loss)

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']
        valid_dice = valid_metrics['dice']
        args.experiment.log_metrics(valid_metrics, step=epoch)


        checkpoint = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_iou": valid_iou,
            "valid_dice": valid_dice,
            "best_dice": best_dice,
            "best_epoch_dice": best_epoch_dice,
            "valid_loss": valid_loss
        }

        if valid_dice > best_dice:
            print('=================== New best model of dice! ========================')
            best_dice = valid_dice
            best_epoch_dice = epoch
            if args.save_model == 'True':
                torch.save(checkpoint, os.path.join(save_model_path, "best_model_dice.pt"))
        
        print('Best epoch for dice unitl now:', best_epoch_dice, ' best dice:', best_dice)
     
    print('==> Training finished.')
    print('Best epoch dice:', best_epoch_dice,
          ' best dice:', best_dice)
    args.experiment.log_others({'best dice': best_dice, 'best dice epoch': best_epoch_dice})


# Modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
