import os
import time
import sys
import traceback

import cv2
import torch
import torch.nn as nn
import timm

import numpy as np

from models.build_model import build_model
from data.datasets import build_test_loader
from collections import OrderedDict


def main(args):
    
    print("Creating data loaders")
    test_dataloader = build_test_loader(args.test_path, args.input_size, args.batch_size, args.num_workers)
    
    classes = torch.load(args.checkpoint)['classes']
    #print(classes)

    if args.hub == 'tv':
        model = build_model(args.net, pretrained=False, fine_tune=False, num_classes=len(classes))
    elif args.hub == 'timm':
        model = timm.create_model(args.net, pretrained=False, num_classes=len(classes))
    else:
        raise NameError('Model hub only support tv or timm')
    # support multi gpu
    model = nn.DataParallel(model)#, device_ids=args.device)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model.cuda()

    model.eval()

    epoch_data_len = len(test_dataloader.dataset)
    batch_num = len(test_dataloader)

    bg_time = time.time()
    with torch.no_grad():
        for images, img_path in test_dataloader:
            try:
                input = images.cuda()
                output = model(input)   #[batch_size, len(classes)]

                indexes = output.detach().cpu().numpy().argmax(1)
                for i, index in enumerate(indexes):
                    print('{}\t{}\t{}'.format(img_path[i], classes[index], index))
                    sys.stdout.flush()

            except:
                #print(image)
                traceback.print_exc()

    total_time = time.time() - bg_time
    print('Total used time:{}, Avg used time:{}'.format(total_time, {total_time/epoch_data_len*batch_num}))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Batch Inference')

    parser.add_argument('--test-path', default='./data/beauty', help='dataset')
    parser.add_argument('--hub', default='tv', help='model hub, from torchvision(tv) or timm')
    parser.add_argument('--net', default='resnet50', help='model name')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--num-workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--checkpoint', default='./checkpoints/model_2_600.pth', help='checkpoint')
    parser.add_argument('--input-size', default=224, type=int, help='size of input')

    args = parser.parse_args()

    print(args)
    main(args)
