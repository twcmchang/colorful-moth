
# coding: utf-8

# # Package

# In[1]:


import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from func.data_generator import get_label_path, DataGenerator
from func.unet_model import NeuralNetwork
from func.plot import plt_learning_curve
from func.callbacks import MyCallbacks
from func.data_generator import my_aug
from func.metrics import mean_iou_score
from func.tool import get_fname


# # Parser

# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', default='0')
parser.add_argument('--test_mode', dest='test_mode', default= False , type=bool)

### data
parser.add_argument('--SAVEDIR', dest='SAVEDIR', default='model/Unet_rmbg') # 'Unet_5comps', #Unet_rmbg
parser.add_argument('--labeltype', dest='labeltype', default='.png') # '.npy'  '.png'
parser.add_argument('--num_class', dest='num_class', default= 1, type=int)
parser.add_argument('--XX_DIR', dest='XX_DIR', default='data/ori/tesri/')
parser.add_argument('--YY_DIR', dest='YY_DIR', default='model/Unsup_rmbg/result_sample/predict_mask_postprocessd/')

parser.add_argument('--image-h', dest='im_h', default=256, type=int)
parser.add_argument('--image-w', dest='im_w', default=256, type=int)
parser.add_argument('--image-c', dest='im_c', default=3, type=int)


### model
parser.add_argument('--epoch', dest='epoch', type=int, default=300)
parser.add_argument('--bz', '--batch-size', dest='bz', type=int, default=16)
parser.add_argument('--lr', '--learning-rate', dest='lr', type=float, default=1e-4)
parser.add_argument('--current_best_val_loss', dest='current_best_val_loss', default=float("inf"), type=float)
parser.add_argument('--earlystop_patience', dest='earlystop_patience', default=25, type=float)
parser.add_argument('--min_delta', dest='min_delta', default=0, type=float)
parser.add_argument('--lr_patience', dest='lr_patience', default= 10, type=int)
parser.add_argument('--lr_reduce_factor', dest='lr_reduce_factor', default= 0.5, type=float)
parser.add_argument('--keep_prob', dest='keep_prob', default= 1, type=float)
parser.add_argument('--log_step', dest='log_step', default= 0.2, type=float)
parser.add_argument('--momentum', dest='momentum', default= 0.9, type=float)

### return parser
args = parser.parse_args()

### set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# # Data & Generator

# In[3]:


### DATA PREPARING
XX_DIR = args.XX_DIR
YY_DIR = args.YY_DIR
Xname = os.listdir(YY_DIR)
Xname = [get_fname(name) for name in Xname]

name_train, name_test = train_test_split(Xname,      test_size=2/10, random_state=55)
name_train, name_val  = train_test_split(name_train, test_size=3/10, random_state=55)

Xtrain = [os.path.join(XX_DIR, name+'.jpg') for name in name_train]
Xval   = [os.path.join(XX_DIR, name+'.jpg') for name in name_val]
Xtest  = [os.path.join(XX_DIR, name+'.jpg') for name in name_test]

if args.labeltype =='.npy':
    if args.num_class==1:
        task_type = 'mothmask'
    if args.num_class==5:
        task_type = 'multilabel'
    Ytrain = [get_label_path(i, YY_DIR, task_type = task_type) for i in Xtrain]
    Yval   = [get_label_path(i, YY_DIR, task_type = task_type) for i in Xval]
    Ytest  = [get_label_path(i, YY_DIR, task_type = task_type) for i in Xtest]
if args.labeltype =='.png':
    Ytrain = [os.path.join(YY_DIR, name+'.png') for name in name_train]
    Yval   = [os.path.join(YY_DIR, name+'.png') for name in name_val]
    Ytest  = [os.path.join(YY_DIR, name+'.png') for name in name_test]

print('train: %s, %s' % (len(Xtrain), len(Ytrain)))
print('valid: %s, %s' % (len(Xval),   len(Yval)))
print('test : %s, %s' % (len(Xtest),  len(Ytest)))


# In[4]:


### Generator
datagen = DataGenerator(input_shape=(args.im_h, args.im_w, args.im_c))
train_gen = datagen.gen_train_data(Xtrain, Ytrain, bz=args.bz, img_augmenter=1)
val_gen   = datagen.get_test_data(Xval, Yval, bz=args.bz, img_augmenter=1)

### Epoch & Iter
train_iter = int(np.ceil(len(Xtrain)/(args.bz))) 
val_iter   = int(np.ceil(len(Xval)/(args.bz))) 
if args.test_mode == True:
        train_iter, val_iter = 2, 2


# # Model

# In[5]:


md = NeuralNetwork(args = args) 
md.build_graph()
md.attach_saver()


# # Train

# In[6]:


### Save DIR
save_dir = '%s/' %(args.SAVEDIR)
if not os.path.exists(save_dir):
     os.makedirs(save_dir)
### Model naming
train_logtime = time.strftime("%y%m%d%H%M%S")
if args.test_mode == True:
    train_logtime = train_logtime+'_test'
model_dir = os.path.join(save_dir, train_logtime)

with tf.Session(graph=md.graph) as sess: 
    cb = MyCallbacks(args, model_dir, md, sess)
    l_rate = cb.get_current_lr()
    ### Variable initialization.
    sess.run(tf.global_variables_initializer())

    ### Parameters that should be stored.
    params = {}
    params['train_loss']=[]
    params['valid_loss']=[]
    params['train_score']=[]
    params['valid_score']=[]
    
    while(cb.get_current_earlystop_status() is not True):
        ### Train
        train_loss_collector = []
        stime = time.time()
        for train_batch_i in range(train_iter):
            print('\r[TRAINING]-----train-mini-Batch ({}/{})'.format(train_batch_i+1, train_iter), end='\r')
            x_batch, y_batch= next(train_gen)

            train_loss_batch, train_pred_batch, _, __ = sess.run([md.loss_tf,
                                                                  md.y_pred_tf, 
                                                                  md.train_step_tf, 
                                                                  md.extra_update_ops_tf], 
                                                                  feed_dict={md.x_data_tf: x_batch, 
                                                                             md.y_data_tf: y_batch[:,:,:,0:args.num_class], 
                                                                             md.keep_prob_tf: args.keep_prob, 
                                                                             md.learn_rate_tf: l_rate,
                                                                             md.training_tf: True})
            train_loss_collector = np.append(train_loss_collector, train_loss_batch)

        ### Valid
        valid_loss_collector = []
        for valid_batch_i in range(val_iter):
            print('\r[Validating]-----valid-mini-Batch ({}/{})'.format(valid_batch_i+1, val_iter), end='\r')
            x_valid_batch, y_valid_batch= next(val_gen)
            val_loss_batch, val_pred_batch = sess.run([md.loss_tf, 
                                                       md.y_pred_tf], 
                                                       feed_dict = {md.x_data_tf: x_valid_batch,
                                                                    md.y_data_tf: y_valid_batch[:,:,:,0:args.num_class], 
                                                                    md.keep_prob_tf: 1.0})
            valid_loss_collector = np.append(valid_loss_collector, val_loss_batch)
        ### training log
        train_loss = np.mean(train_loss_collector)
        valid_loss = np.mean(valid_loss_collector)
        params['train_loss'].extend([train_loss])
        params['valid_loss'].extend([valid_loss])
        
        
         ### callbacks
        epoch_counter = cb.update_record(train_loss = train_loss,
                                         valid_loss = valid_loss,
                                         l_rate=l_rate,
                                         save_ckpt = True,
                                         earlystop = True,
                                         reduce_lr = True,
                                         print_out = True)
        l_rate = cb.get_current_lr()

        if epoch_counter >= cb.get_epoch():
            break
print('Training Ended')
print('model_dir: ', model_dir)


# In[7]:


### learning curve
best_val_loss = min(params['valid_loss'])
best_epoch = np.argmin(params['valid_loss'])
                    
sub = 'min val_loss %.4f at epoch %s' % (best_val_loss, best_epoch)

fig = plt_learning_curve(params['train_loss'], params['valid_loss'],
                         title = 'Loss', sub = '%s | %s' %(model_dir, sub))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
fig.savefig(os.path.join(model_dir, 'loss.png'))
print('model_dir: ', model_dir)
fig


# # Test

# In[8]:


### Generator & training setting 
datagen = DataGenerator(input_shape=(args.im_h, args.im_w, args.im_c))
test_gen  = datagen.get_test_data(Xtest, Ytest, bz=args.bz)
test_iter  = int(np.ceil(len(Xtest)/(args.bz))) 
epoch = args.epoch
if args.test_mode == True:
        test_iter = 2
        epoch = 10


# In[9]:


# ### Test on previous model
# model_dir = 'model/Unet_rmbg/181121113804'

### Test on current model
checkpoint_path = os.path.join(model_dir, 'model_ckpt') 

with tf.Session(graph=md.graph) as sess: 
    sess.run(tf.global_variables_initializer()) # Variable initialization.
    meta_to_restore = checkpoint_path+'.meta'
    saver = tf.train.import_meta_graph(meta_to_restore)
    saver.restore(sess,checkpoint_path)
    print('Model Restored')
    
    test_loss_collector = []
    overall_miou = 0
    for test_batch_i in range(test_iter):
        print('\r[Testing]-----test-mini-Batch ({}/{})'.format(test_batch_i+1, test_iter), end='\r')
        #x_test_batch, y_test_batch, path = next(test_gen)
        x_test_batch, y_test_batch = next(test_gen)
        test_loss_batch, test_pred_batch = sess.run([md.loss_tf,md.y_pred_tf], 
                                                feed_dict = {md.x_data_tf: x_test_batch, 
                                                             md.y_data_tf: y_test_batch[:,:,:,0:args.num_class], 
                                                             md.keep_prob_tf: 1.0})

        test_loss_collector = np.append(test_loss_collector, test_loss_batch)
### mIoU        
        for idx in range(args.bz):
            ### making background channel
            plain = np.ones((256, 256))
            for i in range(args.num_class):       
                true_y = y_test_batch[idx]
                plain = plain-true_y[:,:,i]
            plain = np.where(plain>0.5, 1, 0)
            plain = np.expand_dims(plain, axis=2)
            true_y = np.concatenate((true_y,plain), axis = 2)
            
            ### making background channel
            plain = np.ones((256, 256))
            for i in range(args.num_class):
                pred_y = test_pred_batch[idx]
                plain = plain-pred_y[:,:,i]
            plain = np.where(plain>0.5, 1, 0)
            plain = np.expand_dims(plain, axis=2)
            pred_y = np.concatenate((pred_y,plain), axis = 2)

            true_y = np.argmax(true_y, axis=2)
            pred_y = np.argmax(pred_y, axis=2)
            print(mean_iou_score(pred_y, true_y, n_labels = args.num_class+1))
            
            overall_miou += mean_iou_score(pred_y, true_y, n_labels = args.num_class+1)  
    mIoU = overall_miou/(test_iter * args.bz)
    print('overall_miou on BRCAS:', mIoU) 
######################################### 
        
    test_loss = np.mean(test_loss_collector)
    print('\ntest loss: %.4f' % test_loss)


# # SAVE_LOG

# In[10]:


### Save log
summary_save = '%s/training_summary.csv' %(args.SAVEDIR)
# save into dictionary
sav = vars(args)
sav['test_loss'] = test_loss 
sav['mIoU'] = mIoU
sav['model_dir'] = model_dir
sav['best_val_loss'] =  best_val_loss
sav['best_epoch'] = best_epoch

### Append into summary files
dnew = pd.DataFrame(sav, index=[0])
if os.path.exists(summary_save):
    dori = pd.read_csv(summary_save)
    dori = pd.concat([dori, dnew])
    dori.to_csv(summary_save, index=False)
else:
    dnew.to_csv(summary_save, index=False)

print(summary_save)

