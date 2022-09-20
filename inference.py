import os
import time
import sys
import traceback

import cv2
import torch
import timm
import numpy as np

from models.build_model import build_model
from collections import OrderedDict

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean[:, None, None]
    img *= denominator[:, None, None]
    return img

def transforms_cv2(image, resize=(224, 224)):
    image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0)

    return image

def main(args):
    # only use single gpu or cpu
    device = torch.device('cuda:0') if args.device=='cuda' else torch.device(args.device)
    classes = torch.load(args.checkpoint, map_location=torch.device(device))['classes']
    if args.hub == 'tv':
        model = build_model(args.net, pretrained=False, fine_tune=False, num_classes=len(classes))
    elif args.hub == 'timm':
        #print(timm.list_models(pretrained=True))
        model = timm.create_model(args.net, pretrained=False, num_classes=len(classes))
    else:
        raise NameError('Model hub only support tv or timm')
    print('Loading trained model weightes...')
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in 
        torch.load(args.checkpoint, map_location=torch.device(device))['model_state_dict'].items()})

    model = model.to(device)

    model.eval()

    cnt = 0
    total_used_time = 0.0
    for image_name in os.listdir(args.test_path):
        try:
            image_ = args.test_path + '/' + image_name
            image = cv2.imread(image_)
            cnt += 1
            if cnt >= 5: # warmup for gpu
                bg_time = time.time()
            image_tensor = transforms_cv2(image, resize=(args.resize, args.resize))
            input = image_tensor.to(device)
            output = model(input)
            if cnt >= 5:
                total_used_time += (time.time() - bg_time)

            index = output.detach().cpu().numpy().argmax()
            #print(output.data.cpu().numpy())
            print('{}\tpredict: {}\t{}'.format(image_, classes[index], index))
            sys.stdout.flush()

        except:
            #print(image)
            traceback.print_exc()

    print('Total test num: {}, Avg used time: {}'.format(cnt, total_used_time/(cnt-4)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('--test-path', default='./data/beauty', help='dataset')
    parser.add_argument('--hub', default='tv', help='model hub, from torchvision(tv) or timm')
    parser.add_argument('--net', default='resnet50', help='model name')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='device, cpu or cuda')
    parser.add_argument('--checkpoint', default='./checkpoints/model.pth', help='checkpoint')
    parser.add_argument('--resize', default=224, type=int, help='size of resize')

    args = parser.parse_args()

    print(args)
    main(args)
