import os
import time
import sys
import traceback
import glob

import cv2
import numpy as np
import onnxruntime


def main(args):
    classes = ['nsfw', 'sfw']

    session =  onnxruntime.InferenceSession(args.model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = [output.name for output in session.get_outputs()]
    
    tp, fp, tn, fn = 0, 0, 0, 0
    total_num = 0
    total_used_time = 0.0
    for ipath in glob.glob("/data/test/*/*.jpg", recursive=True):
        image = cv2.imread(ipath)
        if image is not None:
            total_num += 1
            label = ipath.split('/')[-2]
            if total_num >= 5:
                bg_time = time.time()
            blob = cv2.dnn.blobFromImage(image, 1.0/127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
            out = session.run(output_name, {input_name: blob})[0][0]
            if total_num >= 5:
                total_used_time += (time.time() - bg_time)
            pred = classes[np.argmax(out)]
            if pred == label == "positive":
                tp += 1
            elif pred == label == "negative":
                tn += 1
            elif pred == "negative" and label == "positive":
                fn += 1
            else:
                fp += 1
    print('Total test number is: {}, average used time is: {}'.format(total_num, total_used_time/(total_num-4)))
    """
    FPR: False Positive Rate,  FPR = FP / N = FP / (FP+TN) = 1 - TNR
    FNR: False Negtive Rate, FNR = FN / P = FN / (FN+TP) = 1 - TPR
    TPR: True Positive Rate, TPR = TP / p = TP / (TP+FN) = 1 - FNR
    TNR: True Negtive Rate, TNR = TN / N = TN / (TN+FP) = 1 - FPR
    Accuracy = (TP+TN) / (TP+FP+FN+TN)
    Precision(PPV) = TP / (TP+FP)
    Recall(TPR) = TP / (TP+FN) = 1 - FNR
    F1 score = 2 * precision * recall / (precision + recall)
    要求FPR和FNR越低越好
    """
        
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2 * precision * recall / (precision + recall)

    print('FPR={}, FNR={}, accuracy={}, precision={}, recall={}, f1score={}'.format
          (fpr, fnr, accuracy, precision, recall, f1score))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ONNX GPU Inference')

    parser.add_argument('--model-path', default='./models/model.onnx', help='onnx model')
    parser.add_argument('--resize', default=224, type=int, help='size of resize')

    args = parser.parse_args()

    print(args)
    main(args)
