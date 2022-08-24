
import torch
import numpy as np
import onnx
import onnxruntime
import cv2
import argparse
from build_model import build_model

class OnnxConvertor:
    """Convert a pytorch model to a onnx model.
    """
    def __init__(self):
        pass
    def convert(self, model, output_path, batch_inf=False):
        img = np.random.randint(0, 255, size=(224,224,3), dtype=np.uint8)
        img = img[:,:,::-1].astype(np.float32)
        #img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125 # 1/128  
        img = (img.transpose((2, 0, 1)) - 127.5) * 0.00784313725 # 1/127.5
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(device)
        torch.onnx.export(
            model,
            img,
            output_path,
            keep_initializers_as_inputs=False,
            verbose=True,
            opset_version=args.opset
        )
        # batch inference.
        if batch_inf:
            onnx_model = onnx.load(output_path)
            graph = onnx_model.graph
            graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            onnx.save(onnx_model, output_path)

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
        session = onnxruntime.InferenceSession(output_path, None)
        input_cfg = session.get_inputs()[0]
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = []
        images.append(image)
        blob = cv2.dnn.blobFromImages(images, 1.0/127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=False)
        net_out = session.run(output_names, {input_name : blob})[0]
        output = net_out[0]
        #print(f'output of onnx model:{output}')
        output = output/np.linalg.norm(output)
        return output
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export model to onnx.')
    parser.add_argument('--net_name', type=str, default='efficientnet_b6', 
                      help = 'The Neural Network name')
    parser.add_argument('--checkpoint', type = str, default = '../exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01.pth',
                      help = 'The path of checkpoint model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], 
                      help='device, cpu or cuda')
    parser.add_argument('--output_path', type = str, default = '../exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01.onnx',
                      help = 'The output path of onnx model')
    parser.add_argument('--opset', type=int, default=11, help='opset version.')
    parser.add_argument('--batch_inf', type=bool, default=False, help='Set if use for batch inference')
    args = parser.parse_args()

    image1 = 'test_images/ffda2bd6-181a-4ec6-9878-3d5a26c73a86_nsfw.jpg'
    image2 = 'test_images/fffbb437-f692-4dba-921e-a5e9b11ebe51_sfw.jpg'
    
    device = torch.device('cuda:0') if args.device=='cuda' else torch.device(args.device)
    classes = torch.load(args.checkpoint, map_location=torch.device(device))['classes']
    model = build_model(args.net_name, pretrained=False, fine_tune=False, num_classes=len(classes))
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