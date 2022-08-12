from __future__ import print_function
import datetime
import os
import time
import math
#from tqdm.auto import tqdm

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F

import utils
from datasets_pil import get_datasets, get_data_loaders
from model import build_model

# 慎用，最好手动对超出尺寸的图片进行裁剪
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, cur_epoch, val_dataloader, classes, args):
    epoch_start = time.time()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    batch_num = len(data_loader)
    print('Train data num: {}'.format(epoch_data_len))
    print('Train batch num: {}'.format(batch_num))

    for batch_idx, (image, target) in enumerate(data_loader):
        batch_start = time.time()
        image, target = image.cuda(), target.cuda()
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)

        loss_ = loss.item() * image.size(0) # this batch loss
        correct_ = torch.sum(preds == target.data) # this batch correct number

        running_loss += loss_
        running_corrects += correct_

        batch_end = time.time()
        if batch_idx % args.print_freq == 0 and batch_idx != 0:
            lr = get_lr(optimizer)
            print('[TRAIN] Epoch: {}/{}, Batch: {}/{}, lr:{}, BatchAcc: {:.4f}, BatchAvgLoss: {:.4f}, BatchTime: {:.4f}'.format(
                cur_epoch, args.epochs, batch_idx, batch_num, lr, 
                correct_.double()/image.size(0), loss_/image.size(0), batch_end-batch_start))

        # if this result is the best, save it
        # show the best model in validation
        if (batch_idx+1) % args.eval_freq == 0 and batch_idx != 0:
            val_acc = evaluate(model, criterion, val_dataloader, cur_epoch, batch_idx, args)
            model.train()
            # the first or best will save
            if len(g_val_accs) == 0 or val_acc > g_val_accs.get(max(g_val_accs, key=g_val_accs.get), 0.0):
                print('*** GET BETTER RESULT READY SAVE ***')
                if args.save_path:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'epoch': cur_epoch,
                        'batch_id': batch_idx,
                        'classes': classes},
                        os.path.join(args.save_path, '{}@epoch{}_{}_{}.pth'.format(args.model, cur_epoch, batch_idx, lr)))
                    print('*** SAVE.DONE. VAL_BEST_INDEX: {}_{}, VAL_BEST_ACC: {} ***'.format(cur_epoch, batch_idx, val_acc))
            g_val_accs[str(cur_epoch)+'_'+str(batch_idx)] = val_acc
            k = max(g_val_accs, key=g_val_accs.get)
            print('val_best_index: [ {} ], val_best_acc: [ {} ]'.format(k, g_val_accs[k]))

    lr = get_lr(optimizer)
    epoch_loss = running_loss / epoch_data_len
    epoch_acc = running_corrects.double() / epoch_data_len
    epoch_end = time.time()
    print()
    print('[Train@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}, lr: {}'.format(cur_epoch,
          args.epochs, epoch_acc, epoch_loss, epoch_end-epoch_start, lr))
    print()
    print()


def evaluate(model, criterion, data_loader, epoch, step, args):
    epoch_start = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    epoch_data_len = len(data_loader.dataset)
    print('Val data num: {}'.format(epoch_data_len))

    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(data_loader):
            batch_start = time.time()
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            loss_ = loss.item() * image.size(0) # this batch loss
            correct_ = torch.sum(preds == target.data) # this batch correct number

            running_loss += loss_
            running_corrects += correct_

            batch_end = time.time()
            if batch_idx % args.print_freq == 0:
                print('[VAL] Epoch: {}/{}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(step,
                      epoch, args.epochs, batch_idx, math.ceil(epoch_data_len/args.batch_size), correct_.double()/image.size(0),
                      loss_/image.size(0), batch_end-batch_start))

        epoch_loss = running_loss / epoch_data_len
        epoch_acc = running_corrects.double() / epoch_data_len
        epoch_end = time.time()
        print('[Val@] Epoch: {}/{}, EpochAcc: {:.4f}, EpochLoss: {:.4f}, EpochTime: {:.4f}'.format(epoch,
              args.epochs, epoch_acc, epoch_loss, epoch_end-epoch_start))
        print()
    return epoch_acc

def load_ckpt(checkpoint_fpath, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    return model, optimizer, lr_scheduler, checkpoint['epoch']+1


def main(args):
    print("Loading data")
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    dataset, dataset_test = get_datasets(traindir, valdir, args.resize, args.crop, args.pretrain)

    print("Creating data loaders")
    data_loader, val_dataloader = get_data_loaders(dataset, dataset_test, args.batch_size, args.num_workers)

    # show all classes
    classes = data_loader.dataset.classes
    #print(classes)

    model = build_model(args.model, pretrained=True, fine_tune=True, weights=args.weight, num_classes=len(classes))

    # support muti gpu
    model = nn.DataParallel(model, device_ids=args.device)
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    criterion = nn.CrossEntropyLoss().cuda(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)

    start_epoch = 0

    if args.resume:
        assert(args.checkpoint), "You need to give a checkpoint model!"
        print("Resuming from checkpoint")
        model, optimizer, lr_scheduler, start_epoch = load_ckpt(args.checkpoint, model, optimizer, lr_scheduler)

    #if args.test_only:
    #    evaluate(model, criterion, val_dataloader)
    #    return

    print("Start training")
    start_time = time.time()
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, epoch, val_dataloader, classes, args)
        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Finetune Training')

    parser.add_argument('--data-dir', default='/ssd/nsfw', help='dataset')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--weight', default='IMAGENET1K_V2', help='the weight of pretrained model')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('--pretrain', default=True, help='use pretrained weights or train from scratch')
    parser.add_argument('-b', '--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--step', default='10,20,25', type=str,
                        help='steps for MultiStepLR, the last num should less then the num of epochs')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate,0.0001 for vit, 0.01 for resnet and efficientnet')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--eval-freq', default=50, type=int, help='validation frequency of batchs')
    parser.add_argument('--save_path', default='./exps', help='path where to save')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--checkpoint', help='the resume checkpoint, need --resume to be True')
    parser.add_argument('--resize', default=256, type=int, help='size of resize')
    parser.add_argument('--crop', default=224, type=int, help='size of crop')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    args = parser.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    g_val_accs = {}

    print(args)
    main(args)
