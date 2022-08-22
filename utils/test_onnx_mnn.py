import onnxruntime
import cv2
import MNN
import numpy as np
import argparse

def classify_onnx(onnx_path, image_path):
    session = onnxruntime.InferenceSession(onnx_path, None)
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

def classify_mnn(mnn_path, image_path):
    interpreter = MNN.Interpreter(mnn_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))
    interpreter.resizeSession(session)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image.transpose((2, 0, 1)) - 127.5) / 127.5
    image = image.astype(np.float32)
    image = image[..., ::-1]
    #print(image.shape)

    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    #print(output_tensor)
    output = output_tensor.getData()
    output = output/np.linalg.norm(output)

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compare onnx model amd mnn model.')
    parser.add_argument('--onnx_path', type = str, default = '../exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01.onnx',
                      help = 'The path of onnx model')
    parser.add_argument('--mnn_path', type = str, default = '../exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01_fp32.mnn',
                      help = 'The path of mnn model')
    args = parser.parse_args()

    image1 = 'test_images/ffda2bd6-181a-4ec6-9878-3d5a26c73a86_nsfw.jpg'
    image2 = 'test_images/fffbb437-f692-4dba-921e-a5e9b11ebe51_sfw.jpg'

    onnx_output1 = classify_onnx(args.onnx_path, image1)
    print(f'onnx output of image1:{onnx_output1}')
    mnn_output1 = classify_mnn(args.mnn_path, image1)
    print(f'mnn output of image1:{mnn_output1}')
    onnx_output2 = classify_onnx(args.onnx_path, image2)
    print(f'onnx output of image2:{onnx_output2}')
    mnn_output2 = classify_mnn(args.mnn_path, image2)
    print(f'mnn output of image2:{mnn_output2}')

