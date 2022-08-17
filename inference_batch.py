import os
import time
import sys
import traceback

import cv2
import torch
import torch.nn as nn
import numpy as np

from utils.build_model import build_model
from datasets import TestDataset, get_transforms
from collections import OrderedDict

def build_loaders(data_paths, args):
    transforms = get_transforms(args.resize, args.resize, mode='test', pretrained=True)
    dataset = TestDataset(
        data_paths,
        transforms=transforms,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return dataloader

def main(args):
    
    print("Creating data loaders")
    test_dataloader = build_loaders(args.test_path, args)
    
    classes = torch.load(args.checkpoint)['classes']
    #print(classes)

    model = build_model(args.net, pretrained=False, fine_tune=False, num_classes=len(classes))
    # support muti gpu
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
    parser.add_argument('--net', default='resnet50', help='model name')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--checkpoint', default='./checkpoints/model_2_600.pth', help='checkpoint')
    parser.add_argument('--resize', default=224, type=int, help='size of resize')

    args = parser.parse_args()

    print(args)
    main(args)
