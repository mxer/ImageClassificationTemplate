import os
import time
import sys
import traceback

import cv2
import numpy as np
import MNN


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
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    return image

def main(args):
    interpreter = MNN.Interpreter(args.mnn_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))
    interpreter.resizeSession(session)

    cnt = 0
    bg_time = time.time()
    for image_name in os.listdir(args.test_path):
        try:
            image_ = args.test_path + '/' + image_name
            image = cv2.imread(image_)
            image = transforms_cv2(image, resize=(args.resize, args.resize))
            image = image[..., ::-1]

            tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
            input_tensor.copyFrom(tmp_input)
            interpreter.runSession(session)
            output_tensor = interpreter.getSessionOutput(session)
            output = output_tensor.getData()
            #print(f'output:{output}')
            sys.stdout.flush()

            cnt += 1

        except:
            #print(image)
            traceback.print_exc()

    total_time = time.time() - bg_time
    print('Total used time:{}, Avg used time:{}'.format(total_time, total_time/cnt))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('--test-path', default='/ssd_1t/xum/nsfw/two_class/nsfw_binary/test/nsfw', help='dataset')
    parser.add_argument('--mnn-path', default='exps/efficientnetb6/efficientnet_b6@epoch7_3199_0.01_fp32.mnn', help='mnn model')
    parser.add_argument('--resize', default=224, type=int, help='size of resize')

    args = parser.parse_args()

    print(args)
    main(args)
