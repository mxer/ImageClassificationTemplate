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
    if net.startswith('resnet'):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif net.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    elif net.startswith('vit'):
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes)

    return model

def debarcle_layers(model, num_debarcle):
    '''Debarcle From the last [-1]layer to the [-num_debarcle] layers, 
    approximately(for there is Conv2d which has only weight parameter)'''
    param_name = [name for name,_ in model.named_parameters()] # All parameters name
    num_debarcle *= 2
    param_debarcle = param_name[-num_debarcle:]
    if param_debarcle[0].split('.')[-1] == 'bias':
        param_debarcle = param_name[-(num_debarcle + 1):]
    for name, param in model.named_parameters():
        param.requires_grad = True if name in param_debarcle else False