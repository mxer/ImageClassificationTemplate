import os
import re
import onnxruntime
import cv2
import MNN
import numpy as np
import argparse

def onnx_to_mnn(onnx_path, mnn_path, if_fp16=False):
    if not if_fp16:
        cmd = "python -m MNN.tools.mnnconvert -f ONNX --modelFile " + onnx_path + " --MNNModel " + mnn_path + " --bizCode MNN"
    else:
        cmd = "python -m MNN.tools.mnnconvert -f ONNX --modelFile " + onnx_path + " --MNNModel " + mnn_path + " --bizCode MNN" + " --fp16"

    res_str = os.popen(cmd).read()
    print(res_str)
    if "Converted Success" in res_str:
        return 0
    else:
        return -1

def classify_onnx(onnx_path, image_path):
    #session =  onnxruntime.InferenceSession(output_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        #session =  onnxruntime.InferenceSession(output_path, providers=['CUDAExecutionProvider'])
        session =  onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'])
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
    parser.add_argument('--onnx_path', type = str, default = '../exps/efficientnetb1v2/efficientnet_b1@epoch15_799_0.001.onnx',
                      help = 'The path of onnx model')
    parser.add_argument('--mnn_path', type = str, default = '../exps/efficientnetb1v2/efficientnet_b1@epoch15_799_0.001_fp32.mnn',
                      help = 'The path of mnn model')
    parser.add_argument('--if_fp16', type=bool, default=False, help='if use fp16')
    args = parser.parse_args()

    image1 = 'test_images/1.jpg'
    image2 = 'test_images/2.jpg'

    flag = onnx_to_mnn(args.onnx_path, args.mnn_path)
    assert flag == 0, "Model converted fail, please retry."

    onnx_output1 = classify_onnx(args.onnx_path, image1)
    print(f'onnx output of image1:{onnx_output1}')
    mnn_output1 = classify_mnn(args.mnn_path, image1)
    print(f'mnn output of image1:{mnn_output1}')
    onnx_output2 = classify_onnx(args.onnx_path, image2)
    print(f'onnx output of image2:{onnx_output2}')
    mnn_output2 = classify_mnn(args.mnn_path, image2)
    print(f'mnn output of image2:{mnn_output2}')

