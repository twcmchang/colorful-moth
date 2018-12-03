
# coding: utf-8

# In[1]:


import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import skimage.io as io
from skimage.transform import resize

import numpy as np

import tensorflow as tf

from func.data_generator import DataGenerator
from func.unet_model import NeuralNetwork
from func.tool import get_fname
from func.plot import plt_result


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', default='7')
parser.add_argument('--test_mode', dest='test_mode', default= False , type=bool)

### model using
parser.add_argument('--model_dir', dest='model_dir', default='model/Unet_rmbg/BRCAS2TESRI/')

### data
parser.add_argument('--XX_DIR', dest='XX_DIR', default='data/ori/tesri/')
parser.add_argument('--image-h', dest='im_h', default=256, type=int)
parser.add_argument('--image-w', dest='im_w', default=256, type=int)
parser.add_argument('--image-d', dest='im_c', default=3, type=int)
parser.add_argument('--num_class', dest='num_class', default= 1, type=int)
parser.add_argument('--keep_prob', dest='keep_prob', default= 1, type=float)

### model
parser.add_argument('--bz', '--batch-size', dest='bz', type=int, default=16)

### return parser
args = parser.parse_args()


### set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# In[3]:


md = NeuralNetwork(args = args) 
md.build_graph()
md.attach_saver()


# In[4]:


Xname = os.listdir(args.XX_DIR)
Xpath = [os.path.join(args.XX_DIR, name) for name in Xname]

datagen = DataGenerator(input_shape=(args.im_h, args.im_w, args.im_c))
pred_gen  = datagen.get_tesri_data(Xpath, bz=args.bz)
pred_iter  = int(np.ceil(len(Xpath)/(args.bz))) 


# In[5]:


model_dir = args.model_dir
checkpoint_path = model_dir + 'model_ckpt' 
# checkpoint_path = model_dir

with tf.Session(graph=md.graph) as sess: 
    sess.run(tf.global_variables_initializer()) # Variable initialization.
    meta_to_restore = checkpoint_path+'.meta'
    saver = tf.train.import_meta_graph(meta_to_restore)
    saver.restore(sess,checkpoint_path)
    print('Model Restored')
    
    pred_loss_collector = []
    for pred_batch_i in range(pred_iter):
        print('\r[Predict]-----pred-mini-Batch ({}/{})'.format(pred_batch_i+1, pred_iter), end='\r')
        x_pred_batch, path = next(pred_gen)
        pred_batch = sess.run([md.y_pred_tf], 
                                feed_dict = {md.x_data_tf: x_pred_batch})
        for i in range(len(pred_batch[0])):
            m_mask = pred_batch[0][i][:,:,0]
            norm_mask = (m_mask-m_mask.min())/ (m_mask.max()-m_mask.min())
            ori_img = io.imread(path[i])
            ori_img = resize(ori_img, output_shape=(256,256,3))
            norm_mask3 = np.stack([norm_mask,norm_mask,norm_mask], axis = 2)
            bin_mask3 = np.where(norm_mask3 > 0.5, 1.0, 0.0)
            w_mask = 1-bin_mask3
            w_img = (bin_mask3 * ori_img)+w_mask
            img_list = [ori_img, bin_mask3, w_img]
            title_list = ['Original image', 'U-Net mask', 'w_img']
            fig = plt_result(img_list, title_list)
            save_to_check = os.path.join(model_dir, 'Predict_checking')
            if not os.path.exists(save_to_check):
                os.makedirs(save_to_check)
            save_to_mask = os.path.join(model_dir, 'Predict_mask')
            if not os.path.exists(save_to_mask):
                os.makedirs(save_to_mask)
            fname = get_fname(path[i])

            
            io.imsave(os.path.join(save_to_mask,'%s.png' % (fname)), bin_mask3[:,:,0])
            fig.savefig(os.path.join(save_to_check,'%s.png' % (fname)), dpi=100, format='png',bbox_inches='tight' )

