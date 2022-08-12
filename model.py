import torch
import torch.nn as nn

def build_model(model, pretrained=True, fine_tune=True, weights=None, num_classes=2):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        weights = weights
    else:
        print('[INFO]: Not loading pre-trained weights')
        weights = None

    model = torch.hub.load("pytorch/vision", model, weights=weights)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    if model == 'resnet50':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model == 'efficientnetb6':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model == 'vit_l_16':
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model
