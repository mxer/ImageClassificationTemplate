import os
import cv2

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_filtering_images(basePath, contains=None):
    # return the set of files that are valid
    return list_filtering_files(basePath, validExts=image_types, contains=contains)


def list_filtering_files(basePath, validExts=None, contains=None):
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

def list_files(basePath, validExts=None, contains=None):
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
                yield image_path

def rewrite_images(basePath, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        print(f'rootDir:{rootDir}')
        # loop over the filenames in the current directory
        for filename in filenames:
            #print(filename)
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            image_path = os.path.join(rootDir, filename)
            if image_path.endswith('jpg'):
                print(image_path)
                img = cv2.imread(image_path)
                cv2.imwrite(image_path, img)


if __name__ == '__main__':
    rewrite_images('./images')
