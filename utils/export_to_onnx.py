import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import timm
import numpy as np
import onnx
import onnxruntime
from onnxsim import simplify
import cv2
import argparse
from models.build_model import build_model
from models.rexnetv1 import ReXNetV1

class OnnxConvertor:
    """Convert a pytorch model to a onnx model.
    """
    def __init__(self):
        pass
    def convert(self, model, output_path, batch_inf=False):
        img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        img = img[:,:,::-1].astype(np.float32)
        #img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125 # 1/128  
        img = (img.transpose((2, 0, 1)) - 127.5) * 0.00784313725 # 1/127.5
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(device)
        # torch --> torchscript --> onnx
        torch.onnx.export(
            model,
            img,
            output_path,
            keep_initializers_as_inputs=False,
            verbose=True,
            opset_version=args.opset
        )

        onnx_model = onnx.load(output_path)

        # batch inference.
        if batch_inf:
            graph = onnx_model.graph
            graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            onnx.save(onnx_model, output_path)

        # simplify and optimize onnx model
        model_simp, check = simplify(onnx_model, perform_optimization=True)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_path)
        

    def classify_pytorch(self, model, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.transpose((2, 0, 1)) - 127.5) / 127.5
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
        with torch.no_grad():
            output = model(image).cpu().numpy()
        output = np.squeeze(output)
        #print(f'output of pytorch model:{output}')
        output = output/np.linalg.norm(output)
        return output

    def classify_onnx(self, output_path, image_path):
        #session =  onnxruntime.InferenceSession(output_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        #session =  onnxruntime.InferenceSession(output_path, providers=['CUDAExecutionProvider'])
        session =  onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'])
        #print(session.get_providers())
        model_inputs = session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        #input_name = session.get_inputs()[0].name
        #print(f'input_name:{input_name}')
        #outputs = session.get_outputs()
        #output_names = []
        #for o in outputs:
        #    output_names.append(o.name)
        model_outputs = session.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        #output_name = [output.name for output in session.get_outputs()]
        #print(output_names)
        image = cv2.imread(image_path)
        images = []
        images.append(image)
        blob = cv2.dnn.blobFromImages(images, 1.0/127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
        net_out = session.run(output_names, {input_names[0] : blob})[0]
        #print(f'net_out:{net_out}')
        #net_out = session.run(output_name, {input_name : blob})[0]
        #net_out = session.run(None, {input_name : blob})[0]
        feature = net_out[0]
        #print(f'feature:{feature}')
        feature = feature/np.linalg.norm(feature)
        return feature
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export model to onnx.')
    parser.add_argument('--hub', default='tv', choices=['tv', 'timm', 'local'], 
                        help='model hub, from torchvision(tv), timm or local')
    parser.add_argument('--net-name', type=str, default='efficientnet_b1_pruned', 
                      help = 'The Neural Network name, available when hub is tv or timm')
    parser.add_argument('--checkpoint', type = str, default = 'exps/efficientnet_b1_pruned/efficientnet_b1_pruned@epoch25_399_0.0001.pth',
                      help = 'The path of checkpoint model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], 
                      help='device, cpu or cuda')
    parser.add_argument('--output-path', type = str, default = 'exps/efficientnet_b1_pruned/efficientnet_b1_pruned@epoch25_399_0.0001.onnx',
                      help = 'The output path of onnx model')
    parser.add_argument('--opset', type=int, default=11, help='opset version.')
    parser.add_argument('--batch-inf', type=bool, default=False, help='Set if use for batch inference')
    args = parser.parse_args()

    image1 = 'test_images/ffda2bd6-181a-4ec6-9878-3d5a26c73a86_nsfw.jpg'
    image2 = 'test_images/fffbb437-f692-4dba-921e-a5e9b11ebe51_sfw.jpg'
    
    device = torch.device('cuda:0') if args.device=='cuda' else torch.device(args.device)
    classes = torch.load(args.checkpoint, map_location=torch.device(device))['classes']
    if args.hub == 'tv':
        model = build_model(args.net_name, pretrained=False, fine_tune=False, num_classes=len(classes))
    elif args.hub == 'timm':
        #print(timm.list_models(pretrained=True))
        model = timm.create_model(args.net_name, pretrained=False, num_classes=len(classes))
    elif args.hub == 'local':
        # The follow two linea need change to corresponding model name and output layer name
        model = ReXNetV1(width_mult=1.0)
        model.output[1] = nn.Conv2d(in_channels=model.output[1].in_channels, out_channels=len(classes), kernel_size=1, bias=True)
    else:
        raise NameError('Model hub only support tv, timm or local')
    print('Loading trained model weightes...')
    model.load_state_dict({
        k.replace('module.', ''): v for k, v in 
        torch.load(args.checkpoint, map_location=torch.device(device))['model_state_dict'].items()})

    model = model.to(device)

    model.eval()

    """ model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_path)['model_state_dict']
    new_pretrained_dict = {}
    for k in model_dict:
        new_pretrained_dict[k] = pretrained_dict['module.'+k] # DP model
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval() """

    convertor = OnnxConvertor()
    convertor.convert(model, args.output_path)
    output1_py = convertor.classify_pytorch(model, image1)
    output1_onnx = convertor.classify_onnx(args.output_path, image1)
    output2_py = convertor.classify_pytorch(model, image2)
    output2_onnx = convertor.classify_onnx(args.output_path, image2)
    
    # check result.
    print(output1_py)
    print(output1_onnx)
    print(output2_py)
    print(output2_onnx)
