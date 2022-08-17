import os
import time
import sys
import traceback

import cv2
import torch
import numpy as np

from utils.build_model import build_model
from collections import OrderedDict

def transforms_cv2(image, resize=(224, 224)):
    rs_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(rs_image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image).permute(2, 0, 1).float()
    image.sub_(127.5).mul_(0.00784313725)
    image_tensor = image.unsqueeze(0)

    return image_tensor

def main(args):
    # only use single gpu or cpu
    device = torch.device('cuda:0') if args.device=='cuda' else torch.device(args.device)
    classes = torch.load(args.checkpoint, map_location=torch.device(device))['classes']
    model = build_model(args.net, pretrained=False, fine_tune=False, num_classes=len(classes))
    print('Loading trained model weightes...')
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in 
        torch.load(args.checkpoint, map_location=torch.device(device))['model_state_dict'].items()})

    model = model.to(device)

    model.eval()

    cnt = 0
    bg_time = time.time()
    for image_name in os.listdir(args.test_path):
        try:
            image_ = args.test_path + '/' + image_name
            image = cv2.imread(image_)
            image_tensor = transforms_cv2(image, resize=(args.resize, args.resize))
            input = image_tensor.to(device)
            output = model(input)

            index = output.detach().cpu().numpy().argmax()
            #print(output.data.cpu().numpy())
            print('{}\tpredict: {}\t{}'.format(image_, classes[index], index))
            sys.stdout.flush()

            cnt += 1

        except:
            #print(image)
            traceback.print_exc()

    total_time = time.time() - bg_time
    print(f'Total used time:{total_time}, Avg used time:{total_time}/{cnt}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('--test-path', default='./data/beauty', help='dataset')
    parser.add_argument('--net', default='resnet50', help='model name')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='device, cpu or cuda')
    parser.add_argument('--checkpoint', default='./checkpoints/model_2_600.pth', help='checkpoint')
    parser.add_argument('--resize', default=224, type=int, help='size of resize')

    args = parser.parse_args()

    print(args)
    main(args)
