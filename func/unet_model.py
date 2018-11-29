import tensorflow as tf
import numpy as np
class NeuralNetwork():

    def __init__(self, args, nn_name='tmp', nn_type='UNet'):
        """Instance constructor."""
        self.args = args
        self.input_shape = [args.im_h, args.im_w, args.im_c] 
        self.output_shape = [args.im_h, args.im_w,args.num_class]

    def weight_variable(self, shape, name=None):
        """ Weight initialization """
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)

    def bias_variable(self, shape, name=None):
        """Bias initialization."""
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)
     
    def conv2d(self, x, W, name=None):
        """ 2D convolution. """
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

    def max_pool_2x2(self, x, name=None):
        """ Max Pooling 2x2. """
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',
                              name=name)
    
    def conv2d_transpose(self, x, filters, name=None):
        """ Transposed 2d convolution. """
        return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=2, 
                                          strides=2, padding='SAME') 
    
    def leaky_relu(self, z, name=None):
        """Leaky ReLU."""
        return tf.maximum(0.01 * z, z, name=name)
    
    def activation(self, x, name=None):
        """ Activation function. """
        a = tf.nn.elu(x, name=name)
        #a = self.leaky_relu(x, name=name)
        #a = tf.nn.relu(x, name=name)
        return a 
    
    def loss_tensor(self):
        """Loss tensor."""
        if True:
            # Dice loss based on Jaccard dice score coefficent.
            axis=np.arange(1,len(self.output_shape)+1)
            offset = 1e-5
            #D_self.y_data_tf = self.y_data_tf
            #D_self.y_pred_tf = self.y_pred_tf
            
            corr = tf.reduce_sum(self.y_data_tf * self.y_pred_tf, axis=axis)
            l2_pred = tf.reduce_sum(tf.square(self.y_pred_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(self.y_data_tf), axis=axis)
            dice_coeff = (2. * corr + 1e-5) / (l2_true + l2_pred + 1e-5)
            # Second version: 2-class variant of dice loss
            #corr_inv = tf.reduce_sum((1.-self.y_data_tf) * (1.-self.y_pred_tf), axis=axis)
            #l2_pred_inv = tf.reduce_sum(tf.square(1.-self.y_pred_tf), axis=axis)
            #l2_true_inv = tf.reduce_sum(tf.square(1.-self.y_data_tf), axis=axis)
            #dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
            #             (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
        if False:
            # Sigmoid cross entropy. 
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.y_data_tf, logits=self.z_pred_tf))
        return loss 
    
    def optimizer_tensor(self):
        """Optimization tensor."""
        # Adam Optimizer (adaptive moment estimation). 
        optimizer = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
                    self.loss_tf, name='train_step_tf')
        return optimizer
   
    def batch_norm_layer(self, x, name=None):
        """Batch normalization layer."""
        if False:
            layer = tf.layers.batch_normalization(x, training=self.training_tf, 
                                                  momentum=self.momentum, name=name)
        else: 
            layer = x
        return layer
    
    def dropout_layer(self, x, name=None):
        """Dropout layer."""
        if True:
            layer = tf.layers.dropout(x, 1-self.args.keep_prob, training=self.training_tf,
                                     name=name) # it's drop rate, not keep rate
        else:
            layer = x
        return layer

    def num_of_weights(self,tensors):
        """Compute the number of weights."""
        sum_=0
        for i in range(len(tensors)):
            m = 1
            for j in range(len(tensors[i].shape)):
              m *= int(tensors[i].shape[j])
            sum_+=m
        return sum_

    def build_UNet_graph(self):
        """ Create the UNet graph in TensorFlow. """
        args = self.args
        # 1. unit 
        with tf.name_scope('1.unit'):
            W1_1 = self.weight_variable([3,3,self.input_shape[2],16], 'W1_1')
            b1_1 = self.bias_variable([16], 'b1_1')
            Z1 = self.conv2d(self.x_data_tf, W1_1, 'Z1') + b1_1
            A1 = self.activation(self.batch_norm_layer(Z1)) # (.,128,128,16)
            A1_drop = self.dropout_layer(A1)
            W1_2 = self.weight_variable([3,3,16,16], 'W1_2')
            b1_2 = self.bias_variable([16], 'b1_2')
            Z2 = self.conv2d(A1_drop, W1_2, 'Z2') + b1_2
            A2 = self.activation(self.batch_norm_layer(Z2)) # (.,128,128,16)
            P1 = self.max_pool_2x2(A2, 'P1') # (.,64,64,16)
        # 2. unit 
        with tf.name_scope('2.unit'):
            W2_1 = self.weight_variable([3,3,16,32], "W2_1")
            b2_1 = self.bias_variable([32], 'b2_1')
            Z3 = self.conv2d(P1, W2_1) + b2_1
            A3 = self.activation(self.batch_norm_layer(Z3)) # (.,64,64,32)
            A3_drop = self.dropout_layer(A3)
            W2_2 = self.weight_variable([3,3,32,32], "W2_2")
            b2_2 = self.bias_variable([32], 'b2_2')
            Z4 = self.conv2d(A3_drop, W2_2) + b2_2
            A4 = self.activation(self.batch_norm_layer(Z4)) # (.,64,64,32)
            P2 = self.max_pool_2x2(A4) # (.,32,32,32)
        # 3. unit
        with tf.name_scope('3.unit'):
            W3_1 = self.weight_variable([3,3,32,64], "W3_1")
            b3_1 = self.bias_variable([64], 'b3_1')
            Z5 = self.conv2d(P2, W3_1) + b3_1
            A5 = self.activation(self.batch_norm_layer(Z5)) # (.,32,32,64)
            A5_drop = self.dropout_layer(A5)
            W3_2 = self.weight_variable([3,3,64,64], "W3_2")
            b3_2 = self.bias_variable([64], 'b3_2')
            Z6 = self.conv2d(A5_drop, W3_2) + b3_2
            A6 = self.activation(self.batch_norm_layer(Z6)) # (.,32,32,64)
            P3 = self.max_pool_2x2(A6) # (.,16,16,64)
        # 4. unit
        with tf.name_scope('4.unit'):
            W4_1 = self.weight_variable([3,3,64,128], "W4_1")
            b4_1 = self.bias_variable([128], 'b4_1')
            Z7 = self.conv2d(P3, W4_1) + b4_1
            A7 = self.activation(self.batch_norm_layer(Z7)) # (.,16,16,128)
            A7_drop = self.dropout_layer(A7)
            W4_2 = self.weight_variable([3,3,128,128], "W4_2")
            b4_2 = self.bias_variable([128], 'b4_2')
            Z8 = self.conv2d(A7_drop, W4_2) + b4_2
            A8 = self.activation(self.batch_norm_layer(Z8)) # (.,16,16,128)
            P4 = self.max_pool_2x2(A8) # (.,8,8,128)
        # 5. unit 
        with tf.name_scope('5.unit'):
            W5_1 = self.weight_variable([3,3,128,256], "W5_1")
            b5_1 = self.bias_variable([256], 'b5_1')
            Z9 = self.conv2d(P4, W5_1) + b5_1
            A9 = self.activation(self.batch_norm_layer(Z9)) # (.,8,8,256)
            A9_drop = self.dropout_layer(A9)
            W5_2 = self.weight_variable([3,3,256,256], "W5_2")
            b5_2 = self.bias_variable([256], 'b5_2')
            Z10 = self.conv2d(A9_drop, W5_2) + b5_2
            A10 = self.activation(self.batch_norm_layer(Z10)) # (.,8,8,256)
            
        # 6. unit
        with tf.name_scope('6.unit'):
            W6_1 = self.weight_variable([3,3,256,128], "W6_1")
            b6_1 = self.bias_variable([128], 'b6_1')
            U1 = self.conv2d_transpose(A10, 128) # (.,16,16,128)
            U1 = tf.concat([U1, A8], 3) # (.,16,16,256)
            Z11 = self.conv2d(U1, W6_1) + b6_1
            A11 = self.activation(self.batch_norm_layer(Z11)) # (.,16,16,128)
            A11_drop = self.dropout_layer(A11)
            W6_2 = self.weight_variable([3,3,128,128], "W6_2")
            b6_2 = self.bias_variable([128], 'b6_2')
            Z12 = self.conv2d(A11_drop, W6_2) + b6_2
            A12 = self.activation(self.batch_norm_layer(Z12)) # (.,16,16,128)
        # 7. unit 
        with tf.name_scope('7.unit'):
            W7_1 = self.weight_variable([3,3,128,64], "W7_1")
            b7_1 = self.bias_variable([64], 'b7_1')
            U2 = self.conv2d_transpose(A12, 64) # (.,32,32,64)
            U2 = tf.concat([U2, A6],3) # (.,32,32,128)
            Z13 = self.conv2d(U2, W7_1) + b7_1
            A13 = self.activation(self.batch_norm_layer(Z13)) # (.,32,32,64)
            A13_drop = self.dropout_layer(A13)
            W7_2 = self.weight_variable([3,3,64,64], "W7_2")
            b7_2 = self.bias_variable([64], 'b7_2')
            Z14 = self.conv2d(A13_drop, W7_2) + b7_2
            A14 = self.activation(self.batch_norm_layer(Z14)) # (.,32,32,64)
        # 8. unit
        with tf.name_scope('8.unit'):
            W8_1 = self.weight_variable([3,3,64,32], "W8_1")
            b8_1 = self.bias_variable([32], 'b8_1')
            U3 = self.conv2d_transpose(A14, 32) # (.,64,64,32)
            U3 = tf.concat([U3, A4],3) # (.,64,64,64)
            Z15 = self.conv2d(U3, W8_1) + b8_1
            A15 = self.activation(self.batch_norm_layer(Z15)) # (.,64,64,32)
            A15_drop = self.dropout_layer(A15)
            W8_2 = self.weight_variable([3,3,32,32], "W8_2")
            b8_2 = self.bias_variable([32], 'b8_2')
            Z16 = self.conv2d(A15_drop, W8_2) + b8_2
            A16 = self.activation(self.batch_norm_layer(Z16)) # (.,64,64,32)
        # 9. unit 
        with tf.name_scope('9.unit'):
            W9_1 = self.weight_variable([3,3,32,16], "W9_1")
            b9_1 = self.bias_variable([16], 'b9_1')
            U4 = self.conv2d_transpose(A16, 16) # (.,128,128,16)
            U4 = tf.concat([U4, A2],3) # (.,128,128,32)
            Z17 = self.conv2d(U4, W9_1) + b9_1
            A17 = self.activation(self.batch_norm_layer(Z17)) # (.,128,128,16)
            A17_drop = self.dropout_layer(A17)
            W9_2 = self.weight_variable([3,3,16,16], "W9_2")
            b9_2 = self.bias_variable([16], 'b9_2')
            Z18 = self.conv2d(A17_drop, W9_2) + b9_2
            A18 = self.activation(self.batch_norm_layer(Z18)) # (.,128,128,16)
        # 10. unit: output layer
        with tf.name_scope('10.unit'):
            W10 = self.weight_variable([1,1,16,args.num_class], "W10")
            b10 = self.bias_variable([args.num_class], 'b10')
            Z19 = self.conv2d(A18, W10) + b10
            A19 = tf.nn.sigmoid(self.batch_norm_layer(Z19)) # (.,128,128,1)
            print(A19.shape)
        
        self.z_pred_tf = tf.identity(Z19, name='z_pred_tf') # (.,128,128,1)
        self.y_pred_tf = tf.identity(A19, name='y_pred_tf') # (.,128,128,1)
        print(self.y_pred_tf.shape)
        
        print('Build UNet Graph: 10 layers, {} trainable weights'.format(
            self.num_of_weights([W1_1,b1_1,W1_2,b1_2,W2_1,b2_1,W2_2,b2_2,
                                 W3_1,b3_1,W3_2,b3_2,W4_1,b4_1,W4_2,b4_2,
                                 W5_1,b5_1,W5_2,b5_2,W6_1,b6_1,W6_2,b6_2,
                                 W7_1,b7_1,W7_2,b7_2,W8_1,b8_1,W8_2,b8_2,
                                 W9_1,b9_1,W9_2,b9_2,W10,b10])))
    
    def build_graph(self):
        """ Build the complete graph in TensorFlow. """
        tf.reset_default_graph()  
        self.graph = tf.Graph()

        with self.graph.as_default():
            
            # Input tensor. [None, h, w, c]
            shape = [None]
            shape = shape.extend(self.input_shape)
            self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=shape, 
                                            name='x_data_tf') # (.,128,128,3)
            
            # Generic tensors.
            self.keep_prob_tf = tf.placeholder_with_default(1.0, shape=(), 
                                                            name='keep_prob_tf') 
            self.learn_rate_tf = tf.placeholder(dtype=tf.float32,
                                                name="learn_rate_tf")
            self.training_tf = tf.placeholder_with_default(False, shape=(),
                                                           name='training_tf')
            # Build U-Net graph.
            self.build_UNet_graph()

            # Target tensor.
            shape = [None]
            shape = shape.extend(self.output_shape)
            self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=shape, 
                                            name='y_data_tf') # (.,128,128,1)
            # Loss tensor
            self.loss_tf = tf.identity(self.loss_tensor(), name='loss_tf')

            # Optimization tensor.
            self.train_step_tf = self.optimizer_tensor()
            
            # Extra operations required for batch normalization.
            self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
               
#             ##### exaimine loss
#             axis=np.arange(1,len(self.output_shape)+1)
#             self.D_corr = tf.reduce_sum(self.y_data_tf * self.y_pred_tf, axis=axis)
#             self.D_l2_pred = tf.reduce_sum(tf.square(self.y_pred_tf), axis=axis)
#             self.D_l2_true = tf.reduce_sum(tf.square(self.y_data_tf), axis=axis)
#             self.D_dice_coeff = (2. * self.D_corr + 1e-5) / (self.D_l2_true + self.D_l2_pred + 1e-5)
#             #loss = tf.subtract(1., tf.reduce_mean(dice_coeff))


    def attach_saver(self):
        """ Initialize TensorFlow saver. """
        with self.graph.as_default():
            self.use_tf_saver = True
            self.saver_tf = tf.train.Saver()


print('Model Built!')


