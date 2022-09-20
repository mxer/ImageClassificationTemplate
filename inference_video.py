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
        model = timm.create_model(args.net, pretrained=False, num_classes=len(classes))
    else:
        raise NameError('Model hub only support tv or timm')
    print('Loading trained model weightes...')
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in 
        torch.load(args.checkpoint, map_location=torch.device(device))['model_state_dict'].items()})

    model = model.to(device)

    model.eval()

    capture = cv2.VideoCapture()
    capture.open(args.test_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps is: {}, total frame number is: {}".format(fps, frames))
    if capture.isOpened():
        frame_index = -1
        while True:
            ret, image = capture.read()
            if not ret: break
            else:
                #cv2.imwrite(args.test_path+"-"+str(frame_index+1)+".jpg", image)
                frame_index += 1
                image_tensor = transforms_cv2(image, resize=(args.resize, args.resize))
                input = image_tensor.to(device)
                output = model(input)

                index = output.detach().cpu().numpy().argmax()
                #print(output.data.cpu().numpy())
                print('The {} frame predict is: {}\t{}'.format(frame_index, classes[index], index))
    else:
        print("Video open failed!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classification Inference')

    parser.add_argument('--test-path', default='./data/beauty', help='dataset')
    parser.add_argument('--hub', default='tv', help='model hub, from torchvision(tv) or timm')
    parser.add_argument('--net', default='resnet50', help='model name')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='device, cpu or cuda')
    parser.add_argument('--checkpoint', default='./checkpoints/model_2_600.pth', help='checkpoint')
    parser.add_argument('--input-size', default=224, type=int, help='size of input')

    args = parser.parse_args()

    print(args)
    main(args)
