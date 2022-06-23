# import comet_ml at the top of your file
from comet_ml import Experiment
import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from pprint import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from albumentations import (CenterCrop, Compose, HorizontalFlip,  # FDA,
                            Normalize, PadIfNeeded, RandomCrop, Resize,
                            VerticalFlip)
from torch import nn
from torch.optim import Adam, SGD

from utils import *
from loss import CELoss, LossBinary
from models import model_list, AlbuNet, LinkNet34, UNet, UNet11, UNet16
from validation import validation_multi, validation_binary

warnings.filterwarnings("ignore")

# assign GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

def main():
    # The results are reproduciable, deterministic training
    seed_everything(3407)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch_size', type=int, default=64)
    arg('--n_epochs', type=int, default=100, help='the number of total epochs')
    arg('--lr', type=float, default=1e-3)
    arg('--workers', type=int, default=8)
    arg('--num_classes', type=int, default=1) # 1 (nuclei) + 1 (background)
    arg('--model', type=str, default='UNet', choices=model_list.keys())
    arg('--method', type=str, default='Baseline', choices=['Baseline', 'FDA'])
    arg('--save_model', type=str, default='True', help='save model or not')

    arg('--max_samples', type=int, default=5000)

    arg('--train_dataset', type=str, default='Blend', choices=["Endo18_train", "Blend", "None"], help='training dataset')
    arg('--val_dataset', type=str, default='Endo18_test', choices=["Endo18_test", "Blend", "None"], help='validation dataset')
    arg('--blend_mode', type=str, default='None', 
        choices=["alpha", "gaussian", "laplacian", 
        "paste", "paste_2base_1instru", 
        "paste_2base_1instru_bgold", 
        'paste_2base50_1instru_bgold', 'paste_2base50_multi_instru_bgold', 
        'paste_3base50_1instru_bgold', 'paste_3base50_multi_instru_bgold', 
        'paste_2base50_multi_instru_2bg', 'paste_3base50_multi_instru_2bg',
        "paste_multi", "paste_multi_color", "paste_multi_pos_color", 
        "multi_agl", "multi_gp", "None"])
    # arg('--test_dataset', type=str, default='Blend', choices=["Endo18_test", "Blend", "None"], help='test dataset')

    arg('--augmix', type=str, default='None',
        choices=["None", "I", "II", "III", "IV"])
    arg('--augmix_level', type=int, default=0, choices=[0, 1, 2, 3])

    
    arg('--comet_api_key', type=str, default='', help='comet api key')
    arg('--experiment', help='Comet experiment instance')


    args = parser.parse_args()

    # Create an experiment with your api key
    disable_comet = (args.save_model != 'True')
    print("==> Save experiment to Comet:", (not disable_comet))
    args.experiment = Experiment(api_key=args.comet_api_key, project_name="xxx", workspace="xxx", disabled=disable_comet)
    
    if args.train_dataset == 'Blend':
        args.experiment.set_name(f'{args.method}-train_dataset_{args.train_dataset}-mode_{args.blend_mode}-val_dataset_{args.val_dataset}-augmix_{args.augmix}-L{args.augmix_level}')
    elif args.train_dataset == 'Endo18_train':
        args.experiment.set_name(f'{args.method}-train_dataset_{args.train_dataset}-val_dataset_{args.val_dataset}-augmix_{args.augmix}-L{args.augmix_level}')


    print('=====================================')
    print('model            :', args.model)
    print('method           :', args.method)
    print('train_dataset    :', args.train_dataset)
    print('val_dataset      :', args.val_dataset)
    print('blend_mode       :', args.blend_mode)
    print('augmix           :', args.augmix)
    print('augmix_level     :', args.augmix_level)
    print('=====================================')
    
    # initialize model
    if args.model == 'UNet':
        model = UNet(num_classes=args.num_classes)
    else:
        model_name = model_list[args.model]
        model = model_name(num_classes=args.num_classes, pretrained=True)

    model = model.cuda()  # put model weights into GPU

    # assign GPU device
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        print('The total available GPU_number:', num_gpu)
        if num_gpu > 1:  # has more than 1 gpu
            device_ids = np.arange(num_gpu).tolist()
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    # loss function
    loss = LossBinary()
    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # validation function
    valid = validation_binary
    
    endo18_root_path = Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/')  
    endo18_test_path = endo18_root_path / 'test' # Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/')
    
    train_image_paths, val_image_paths = [], []

    if args.train_dataset == 'Blend':
        if args.blend_mode == 'paste_multi':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_multi_instru/')
        elif args.blend_mode == 'paste_2base_1instru':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2base_1instru/')
        elif args.blend_mode == 'paste_2base_1instru_bgold':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2base_1instru_bgold_resize/')
        elif args.blend_mode == 'paste_2base50_1instru_bgold':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2base50_1instru_bgold/') 
        elif args.blend_mode == 'paste_2base50_multi_instru_bgold':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2base50_multi_instru_bgold/')  
        elif args.blend_mode == 'paste_3base50_1instru_bgold':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_3base50_1instru_bgold/') 
        elif args.blend_mode == 'paste_3base50_multi_instru_bgold':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_3base50_multi_instru_bgold/')    
        # 2 bg
        elif args.blend_mode == 'paste_2base50_multi_instru_2bg':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2bg/blended_2base50_multi_instru_2bg/')   
        elif args.blend_mode == 'paste_3base50_multi_instru_2bg':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_2bg/blended_3base50_multi_instru_2bg/')          
        elif args.blend_mode == 'paste_multi_color':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_multi_instru_color/')
        elif args.blend_mode == 'paste_multi_pos_color':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended_multi_instru_colorize_pos/')
        elif args.blend_mode == 'paste' or args.blend_mode == 'multi_gp':
            paste_root_path = Path('/mnt/data-ssd/wa/overlay/FgBgAug/blended/')
        else:
            paste_root_path = Path('')
        paste_image_path = paste_root_path / 'images'
        paste_image_list = list(paste_image_path.glob('*'))

        blend_data_root_path = Path('/mnt/data-ssd/wa/semi-synthetic/endo_blended/')
        alpha_image_path = blend_data_root_path / 'alpha'
        gaussian_image_path = blend_data_root_path / 'gaussian'
        laplacian_image_path = blend_data_root_path / 'laplacian'

        if 'paste' in args.blend_mode:
            image_list = paste_image_list
        else:   
            all_list, mask_list, image_list = [], [], []    
            if args.blend_mode == 'multi_agl':
                all_list += list(alpha_image_path.glob('*'))
                all_list += list(gaussian_image_path.glob('*'))
                all_list += list(laplacian_image_path.glob('*'))
                mask_list += list(alpha_image_path.glob('*seg*'))
                mask_list += list(gaussian_image_path.glob('*seg*'))
                mask_list += list(laplacian_image_path.glob('*seg*'))
                image_list = list(set(all_list)-set(mask_list))
            elif args.blend_mode == 'multi_gp':
                all_list = list(gaussian_image_path.glob('*'))
                mask_list = list(gaussian_image_path.glob('*seg*'))
                image_list = list(set(all_list)-set(mask_list))
                image_list += paste_image_list
            elif args.blend_mode == 'alpha':
                all_list = list(alpha_image_path.glob('*'))
                mask_list = list(alpha_image_path.glob('*seg*'))
                image_list = list(set(all_list)-set(mask_list))
            elif args.blend_mode == 'gaussian':
                all_list = list(gaussian_image_path.glob('*'))
                mask_list = list(gaussian_image_path.glob('*seg*'))
                image_list = list(set(all_list)-set(mask_list))
            elif args.blend_mode == 'laplacian':
                all_list = list(laplacian_image_path.glob('*'))
                mask_list = list(laplacian_image_path.glob('*seg*'))
                image_list = list(set(all_list)-set(mask_list))
        
        print('Original total number of dataset images:', len(image_list))
        np.random.shuffle(image_list)
        # check max number of training samples
        if len(image_list) > args.max_samples:
            image_list = image_list[:args.max_samples]

        # image_list.sort()
        train_idx, val_idx = train_test_split(len(image_list), 0.2) # 20% for validation

        for idx in train_idx:
            train_image_paths.append(image_list[idx])  

        if args.val_dataset == 'Blend':
            for idx in val_idx:
                val_image_paths.append(image_list[idx]) # 982
        elif args.val_dataset == 'Endo18_test':
            val_image_paths = list((endo18_test_path / 'images').glob('*')) # 999
    elif args.train_dataset == 'Endo18_train':
        endo18_isinet_path = endo18_root_path / 'ISINet_Train_Val'
        endo18_train_path = endo18_isinet_path / 'train'
        endo18_val_path = endo18_isinet_path / 'val'
        train_image_paths = list((endo18_train_path / 'images').glob('*')) + list((endo18_val_path / 'images').glob('*')) # 2235
        val_image_paths = list((endo18_test_path / 'images').glob('*')) # 999
    test_images_paths = list((endo18_test_path / 'images').glob('*')) # 999

    # remove 2 images without annotations
    bug_images = [Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/images/seq_3_frame249.png'),
                  Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/images/seq_4_frame249.png')]
    for bug_image in bug_images:
        if bug_image in val_image_paths:
            val_image_paths.remove(bug_image)
        if bug_image in test_images_paths:
            test_images_paths.remove(bug_image)

    print('Num train = {}, Num_val = {}, Num test = {}'.format(len(train_image_paths), len(val_image_paths), len(test_images_paths)))
    # sys.exit(0)

    args.experiment.log_parameters(vars(args))
    # args.experiment.log_others(vars(args))
    # sys.exit(0)

    # set model saved path
    model_path = set_model_saved_path(args)

    if args.method == 'Baseline':
        if args.augmix == 'None':
            train_loader = make_loader(args, train_image_paths, shuffle=True, transform=train_transform(p=1), mode='train')
        else:
            train_loader = make_loader(args, train_image_paths, shuffle=True, transform=train_transform_augmix(p=1), mode='train')
        valid_loader = make_loader(args, val_image_paths, transform=val_transform(p=1), mode='val')
        test_loader = make_loader(args, test_images_paths, transform=test_transform(p=1), mode='eva')

        with args.experiment.train():
            train(
                args=args,
                model=model,
                criterion=loss,
                train_loader=train_loader,
                valid_loader=valid_loader,
                validation=valid,
                optimizer=optimizer,
                model_path=model_path,
                test_loader=test_loader
            )

        if args.save_model == 'True':
            with args.experiment.test():
                print('==> Test begin...')
                checkpoint = torch.load(os.path.join(model_path, 'best_model_dice.pt'))
                model.load_state_dict(checkpoint['net'])  # load the model's parameters
                # print(checkpoint['optimizer']['param_groups'])

                test_metrics = validation_binary(args, model=model, criterion=loss, valid_loader=test_loader)
                args.experiment.log_metrics(test_metrics)
                
                print('==> Test finished.')
                print('Test result:', test_metrics)

  
if __name__ == '__main__':
    main()
