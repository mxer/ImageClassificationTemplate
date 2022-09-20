import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import timm

from models.build_model import build_model
from utils.plot_display import *
from data.datasets import build_loader

from sklearn.metrics import confusion_matrix, classification_report
from utils.metric import evaluate_accuracy


def test(model, data_loader, classes):
    n = len(data_loader.dataset)
    #print(n)
    batch_num = len(data_loader)
    print('Total Train batch num is: {}'.format(batch_num))
    y_pred = np.empty((n), dtype=int)
    y_true = np.empty((n), dtype=int)

    index = 0
    mis_imgs = []
    with torch.no_grad():
        for batch_idx, (image, target, img_path) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            output = model(image)
            
            scope = image.size(0) # batch num
            _, preds = torch.max(output, 1)
            mis_cls = torch.ne(preds, target)
            #print(mis_cls)
            mis_cls_index_list = torch.nonzero(mis_cls).squeeze(1).tolist()
            #print(mis_cls_index_list)
            if mis_cls_index_list:
                for mis_cls_index in mis_cls_index_list:
                    mis_imgs.append(img_path[mis_cls_index])
            else:
                print('Batch {} all correct!'.format(batch_idx))

            y_pred[index : index+scope] = preds.view(-1).cpu().numpy()
            y_true[index : index+scope] = target.detach().cpu().numpy()

            index += scope

    return y_pred, y_true, mis_imgs


def main(args):

    print("Creating data loaders")
    _, test_loader = build_loader(args.data_dir, args.input_size, args.batch_size, args.num_workers)

    # show all classes
    classes = test_loader.dataset.classes
    print(classes)

    if args.hub == 'tv':
        model = build_model(args.net, pretrained=False, fine_tune=False, num_classes=len(classes))
    elif args.hub == 'timm':
        #print(timm.list_models(pretrained=True))
        model = timm.create_model(args.net, pretrained=False, num_classes=len(classes))
    else:
        raise NameError('Model hub only support tv or timm')
    # support muti gpu
    model = nn.DataParallel(model, device_ids=args.device)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model.cuda()

    model.eval()

    bg_time = time.time()
    y_pred, y_true, mis_cls_images = test(model, test_loader, classes)

    total_time = time.time() - bg_time
    print(f'Total used time:{total_time}')
    print()

    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)
    print()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')
    os.makedirs('plt_images') if not os.path.exists('plt_images') else None
    plt.savefig('./plt_images/test_confusion_matrix.jpg')
    plt.show()
    print()

    report = classification_report(y_true, y_pred, labels=range(len(classes)), output_dict=True)
    print(report)
    print()

    print('All Miss Classified Images are:')
    for i, image in enumerate(mis_cls_images):
        print('{}\t{}'.format(i+1, image))

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Test')

    parser.add_argument('--data-dir', default='/data/beauty', help='dataset')
    parser.add_argument('--hub', default='tv', help='model hub, from torchvision(tv) or timm')
    parser.add_argument('--net', default='efficientnet_b6', help='model name')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-j', '--num-workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--checkpoint', default='exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01.pth', help='checkpoint')
    parser.add_argument('--input-size', default=224, type=int, help='size of input')

    args = parser.parse_args()

    print(args)
    main(args)
