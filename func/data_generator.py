import os 
import numpy as np
from PIL import Image
import imageio
from itertools import cycle
from random import shuffle
import random
import imgaug as ia
from imgaug import augmenters as iaa
def open_y_images(*paths, im_h=None, im_w=None, gray=False):
    """
    @paths: image path
    @im_h:  image height
    @im_w:  image weight
    @gray:  process grayscale image

    FIXME: grayscale <-> RGB <-> BGR
    """
    
    try:
        data = None
        w, h = im_w, im_h
        if (im_h is not None and isinstance(im_h, int) and
            im_w is not None and isinstance(im_w, int)):
#             print(paths[0])
#             for i in paths:
#                 k = Image.open(i).convert('L').resize((w, h))
#                 k = np.array(k)
#                 print(k.shape)
            data = np.array([np.array(Image.open(i).convert('L').resize((w, h)), dtype='float32') for i in paths])
        else:
            data = np.array([np.array(Image.open(i), dtype='float32') for i in paths])
    except Exception as e:
        print(e)
    return data


def open_images(*paths, im_h=None, im_w=None, gray=False):
    """
    @paths: image path
    @im_h:  image height
    @im_w:  image weight
    @gray:  process grayscale image

    FIXME: grayscale <-> RGB <-> BGR
    """
    
    try:
        data = None
        w, h = im_w, im_h
        if (im_h is not None and isinstance(im_h, int) and
            im_w is not None and isinstance(im_w, int)):
            data = np.array([np.array(Image.open(i).resize((w, h)), dtype='float32') for i in paths])
        else:
            data = np.array([np.array(Image.open(i), dtype='float32') for i in paths])
    except Exception as e:
        print(e)
    return data

class RandomRoundRobin(object):
    def __init__(self, data, bz, random_after_epoch=True):
        self.data = data
        self.batch_size = bz
        self.batch_data = self._get_item()
        self.random_after_epoch = random_after_epoch

    def _get_item(self):
        for i, item in enumerate(cycle(self.data)):
            yield (i, item)

    def next_batch(self):
        data = []
        for i in range(self.batch_size):
            count, item = self.batch_data.__next__()
            data.append(item)
            if count == len(self.data)-1 and self.random_after_epoch:
                shuffle(self.data)
                self.batch_data = self._get_item()
        return data

    
class DataGenerator(object):
    def __init__(self, input_shape=(512, 512, 3), random_after_epoch=True):
        self.input_shape = input_shape
        self.random_after_epoch = random_after_epoch

    def _gen_data(self, x_img, y_label, img_augmenter=None):
        h, w, c = self.input_shape
        basename = os.path.basename(y_label[0])
        filetype = os.path.splitext(basename)[1]
        
        if filetype =='.npy':
            X_ = open_images(*x_img, im_h=h, im_w=w)
            X_ = np.stack(X_)
            Y_ = np.stack([np.load(i) for i in y_label])

            if img_augmenter is not None:
                X_ = my_aug(X_)

            if len(Y_.shape) < len(X_.shape) and len(Y_.shape) == 3:
                Y_ = Y_[..., np.newaxis]

            X_ = X_.astype('float64')
            X_ = X_ / 255

            return X_, Y_
        if filetype == '.png':
            X_ = open_images(*x_img, im_h=h, im_w=w)
            Y_ = open_y_images(*y_label, im_h=h, im_w=w)
            X_ = np.stack(X_)
            Y_ = np.stack(Y_)
            
            if img_augmenter is not None:
                X_ = my_aug(X_)

            if len(Y_.shape) < len(X_.shape) and len(Y_.shape) == 3:
                Y_ = Y_[..., np.newaxis]

            X_ = X_.astype('float64')
            X_ = X_ / 255

            Y_ = Y_.astype('float64')
            Y_ = Y_ / 255

            return X_, Y_

    def gen_train_data(self, x_img, y_label, bz=1, img_augmenter=None):
        pair_image = list(zip(x_img, y_label))
        rrr = RandomRoundRobin(pair_image, bz, self.random_after_epoch)
        while True:
            batch = rrr.next_batch()
            x = [i[0] for i in batch]
            y = [i[1] for i in batch]
            X_, Y_ = self._gen_data(x, y, img_augmenter=img_augmenter)
            yield X_, Y_

    def get_test_data(self, x_img, y_label, bz=1, img_augmenter=None):
        pair_image = list(zip(x_img, y_label))
        rrr = RandomRoundRobin(pair_image, bz, False)
        while True:
            batch = rrr.next_batch()
            x = [i[0] for i in batch]
            y = [i[1] for i in batch]
            X_, Y_ = self._gen_data(x, y, img_augmenter=img_augmenter)
            yield X_, Y_
            
            
    def _gen_data_tesri(self, x_img, img_augmenter=None):
        h, w, c = self.input_shape
    
        X_ = open_images(*x_img, im_h=h, im_w=w)
        #plt.imshow(X_[0])
        X_ = np.stack(X_)
        
        if img_augmenter is not None:
            X_ = img_augmenter.augment_images(X_)
            
        X_ = X_.astype('float64')
        X_ = X_ / 255

        return X_
    def get_tesri_data(self, x_img, bz=1, img_augmenter=None):
        #pair_image = list(zip(x_img))
#         print(len(x_img))
#         print(x_img[0])
        rrr = RandomRoundRobin(x_img, bz, self.random_after_epoch)
        while True:
            batch = rrr.next_batch()
#             print(len(batch))
#             print(batch[0])
            x = [i for i in batch]
            #print(len(batch))
            #print(x)
            X_ = self._gen_data_tesri(x, img_augmenter=img_augmenter)
            yield X_, x
            
    
def my_aug(pics):
    # at labels generating step, we will not useaugmentation that would influence  0 value of background
    '''input  : [img]'''
    '''output : [aug_img]''' 
          
    aug = iaa.Sequential(
        [
            iaa.SomeOf(1, [
                iaa.Noop(),
                iaa.Grayscale(alpha=(0.0, 1.0)),
#                 iaa.Invert(0.5, per_channel=True),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                iaa.SaltAndPepper(0.05),
                iaa.AverageBlur(k=(2, 7)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
            ])
        ])
    # let every round use a same augmentation
    aug_pics = aug.augment_images(pics)
       
    return aug_pics


def get_label_path(imgdir_path, labeldir_path, task_type):
    '''
    #multilabel #mothmask
    '''
    basename = os.path.basename(imgdir_path)
    basename = '.'.join(basename.split('.')[:-1])
    #return os.path.join(labeldir_path, basename+'.png') #multilabel.npy #mothmask.npy
    return os.path.join(labeldir_path, basename, task_type+'.npy') #multilabel.npy #mothmask.npy

def get_filename(path):
    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]
    return file_name