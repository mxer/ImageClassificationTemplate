import torch
import torch.nn as nn

def build_model(net, pretrained=True, fine_tune=True, weights=None, num_classes=2):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        weights = weights
    else:
        print('[INFO]: Not loading pre-trained weights')
        weights = None

    model = torch.hub.load("pytorch/vision", net, weights=weights)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    if net == 'resnet50':
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif net == 'efficientnet_b6':
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    elif net == 'vit_l_16':
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes)

    return model
