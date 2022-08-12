from __future__ import print_function
import os
import numpy as np
import random
import cv2
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import requests
from io import StringIO
from urllib import parse


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    corrupted = 0
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                if image_path.endswith('jpg'):
                    with open(image_path, 'rb') as f:
                        f.seek(-2, 2)
                        if f.read() == b'\xff\xd9':
                            yield image_path

                        else:
                            print('The Corrupted Images are:{}'.format(image_path))
                            corrupted += 1
                else:
                    yield image_path
    print('The num of corrupted images is:{}'.format(len(image_path)))

if __name__ == '__main__':
    import cv2

    imagePaths = list(list_images("./images"))

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        cv2.waitKey()

    cv2.destoryAllWindows()


def image_loader_url(url, tsfms):
    # image = Image.open(url).convert('RGB')
    header_info = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1581.2 Safari/537.36',
        'Host': parse.urlparse(url).hostname,
        'Origin': parse.urlparse(url).hostname,
        'Connection': 'keep-alive',
        # 'Referer': urlparse.urlparse(url).hostname,
        'Content-Type': 'application/x-www-form-urlencoded',
            }
    image = requests.get(url, timeout=3, headers=header_info).content
    image = Image.open(StringIO(image)).convert('RGB')
    image_tensor = tsfms(image)
    # fake batch dimension required to fit network's input dimensions
    return image_tensor

def randomFlip(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def randomBlur(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.filter(ImageFilter.BLUR)
    return image

def randomRotation(image, prob=0.5, angle=(1, 60)):
    rnd = np.random.random_sample()
    if rnd < prob:
        random_angle = np.random.randint(angle[0], angle[1])
        return image.rotate(random_angle)
    return image

def randomColor(image, prob=0.7, factor=(1, 90)):
    rnd = np.random.random_sample()
    if rnd < prob:
        # Factor 1.0 always returns a copy of the original image,
        # lower factors mean less color (brightness, contrast, etc), and higher values more
        random_factor = np.random.randint(2, 18) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(2, 18) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return image

def randomGaussian(image, prob=0.5, mean=0, sigma=10):
    rnd = np.random.random_sample()
    if rnd < prob:
        img_array = np.asarray(image)
        noisy_img = img_array + np.random.normal(mean, sigma, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255)

        return Image.fromarray(np.uint8(noisy_img))
    return image
