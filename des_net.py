# Time    : 2018/7/20 下午 3:23
# Author  : Ruandy

import tensorflow as tf
import numpy as np

HPF = np.zeros([5,5,1,16],dtype=np.float32)

HPF[:,:,0,1]=np.array([[0, 0,0, 0,0],[0,0.3266407412, 0.3266407412, 0.3266407412, 0.3266407412] ,[0,0.1352990250, 0.1352990250, 0.1352990250 ,0.1352990250],
[0,-0.1352990250, -0.1352990250, -0.1352990250, -0.1352990250],
[0, -0.3266407412, -0.3266407412, -0.3266407412 ,-0.3266407412]],dtype=np.float32)

HPF[:,:,0,2]=np.array([[0, 0,0, 0,0],
[0,0.3266407412, 0.1352990250, -0.1352990250, -0.3266407412] ,[0,0.3266407412, 0.1352990250, -0.1352990250, -0.3266407412], [0,0.3266407412 ,0.1352990250, -0.1352990250, -0.3266407412],
[0,0.3266407412, 0.1352990250, -0.1352990250, -0.3266407412]
],dtype=np.float32)

HPF[:,:,0,3]=np.array([[0, 0,0, 0,0],[0,0.2500000000, 0.2500000000, 0.2500000000, 0.2500000000],
[0, -0.2500000000, -0.2500000000, -0.2500000000, -0.2500000000],
[0, -0.2500000000, -0.2500000000, -0.2500000000, -0.2500000000 ],
[0,0.2500000000, 0.2500000000, 0.2500000000, 0.2500000000 ]
],dtype=np.float32)

HPF[:,:,0,4]=np.array([[0, 0,0, 0,0],[0,0.4267766953, 0.1767766953, -0.1767766953, -0.4267766953], [0,0.1767766953, 0.0732233047, -0.0732233047, -0.1767766953 ],
[0,-0.1767766953, -0.0732233047, 0.0732233047, 0.1767766953 ],
[0,-0.4267766953, -0.1767766953, 0.1767766953, 0.4267766953 ]
],dtype=np.float32)

HPF[:,:,0,5]=np.array([[0, 0,0, 0,0],[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000 ],[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000 ],[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000],[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000]],dtype=np.float32)
HPF[:,:,0,6]=np.array([[0, 0,0, 0,0],[0,0.1352990250 ,0.1352990250, 0.1352990250, 0.1352990250],
[0, -0.3266407412, -0.3266407412, -0.3266407412, -0.3266407412 ],
[0,0.3266407412, 0.3266407412, 0.3266407412, 0.3266407412],
[ 0,-0.1352990250, -0.1352990250, -0.1352990250, -0.1352990250 ]
],dtype=np.float32)
HPF[:,:,0,7]=np.array([[0, 0,0, 0,0],[0,0.3266407412,0.1352990250, -0.1352990250, -0.3266407412],[0, -0.3266407412, -0.1352990250, 0.1352990250, 0.3266407412],
[0,-0.3266407412, -0.1352990250, 0.1352990250, 0.3266407412], [0,0.3266407412, 0.1352990250, -0.1352990250, -0.3266407412 ]],dtype=np.float32)
HPF[:,:,0,8]=np.array([[0, 0,0, 0,0],[0,0.3266407412, -0.3266407412, -0.3266407412 ,0.3266407412 ],
[0,0.1352990250, -0.1352990250 ,-0.1352990250 ,0.1352990250],
[0, -0.1352990250, 0.1352990250, 0.1352990250, -0.1352990250],
[0, -0.3266407412, 0.3266407412, 0.3266407412 ,-0.3266407412 ]
],dtype=np.float32)

HPF[:,:,0,9]=np.array([[0, 0,0, 0,0],
[0,0.1352990250, -0.3266407412, 0.3266407412 ,-0.1352990250 ],[0,0.1352990250, -0.3266407412, 0.3266407412, -0.1352990250], [0,0.1352990250 ,-0.3266407412, 0.3266407412, -0.1352990250 ],[0,0.1352990250 ,-0.3266407412, 0.3266407412, -0.1352990250 ]
],dtype=np.float32)

HPF[:,:,0,10]=np.array([[0, 0,0, 0,0],[0,0.1767766953, 0.0732233047 ,-0.0732233047, -0.1767766953],
[ 0,-0.4267766953, -0.1767766953, 0.1767766953, 0.4267766953],
[0,0.4267766953 , 0.1767766953, -0.1767766953, -0.4267766953],
[ 0,-0.1767766953 ,-0.0732233047 ,0.0732233047, 0.1767766953]
],dtype=np.float32)

HPF[:,:,0,11]=np.array([[0, 0,0, 0,0],[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000],
[ 0,-0.2500000000 ,0.2500000000, 0.2500000000, -0.2500000000],
[ 0,-0.2500000000, 0.2500000000, 0.2500000000, -0.2500000000] ,[0,0.2500000000, -0.2500000000, -0.2500000000, 0.2500000000 ]
],dtype=np.float32)

HPF[:,:,0,12]=np.array([[0, 0,0, 0,0],[0,0.1767766953, -0.4267766953, 0.4267766953 ,-0.1767766953 ],
[0,0.0732233047, -0.1767766953 ,0.1767766953, -0.0732233047],
[0,-0.0732233047, 0.1767766953, -0.1767766953, 0.0732233047],
[ 0,-0.1767766953, 0.4267766953, -0.4267766953 ,0.1767766953 ]
],dtype=np.float32)

HPF[:,:,0,13]=np.array([[0, 0,0, 0,0],[0,0.1352990250, -0.1352990250 ,-0.1352990250, 0.1352990250],
[ 0,-0.3266407412, 0.3266407412 ,0.3266407412, -0.3266407412 ],
[0,0.3266407412 ,-0.3266407412, -0.3266407412, 0.3266407412 ],
[0,-0.1352990250 ,0.1352990250, 0.1352990250 ,-0.1352990250 ]
],dtype=np.float32)

HPF[:,:,0,14]=np.array([[0, 0,0, 0,0],[0,0.1352990250, -0.3266407412 ,0.3266407412 ,-0.1352990250],
[0, -0.1352990250, 0.3266407412, -0.3266407412, 0.1352990250],
[0, -0.1352990250, 0.3266407412, -0.3266407412, 0.1352990250],
[0,0.1352990250, -0.3266407412, 0.3266407412 ,-0.1352990250]
],dtype=np.float32)

HPF[:,:,0,15]=np.array([[0, 0,0, 0,0],
[0,0.0732233047, -0.1767766953, 0.1767766953, -0.0732233047 ],
[0,-0.1767766953, 0.4267766953 ,-0.4267766953 ,0.1767766953],
[0,0.1767766953, -0.4267766953, 0.4267766953, -0.1767766953 ],
[0,-0.0732233047, 0.1767766953, -0.1767766953, 0.0732233047]
],dtype=np.float32)

def discriminator(cover_tensor,stego_tensor,is_training):# xu's net lite 256

    input_tensor = tf.concat([cover_tensor,stego_tensor],0)
    input_tensor = tf.subtract(tf.divide(input_tensor,128),1) #pre process1

    with tf.variable_scope('dis_net',reuse= tf.AUTO_REUSE):
        hpf_kernels = tf.Variable(HPF,trainable=False,name='dis_HPF')
        conv_pre = tf.nn.conv2d(input_tensor,hpf_kernels,[1,1,1,1],'SAME',name='dis_conv_pre')

        with tf.variable_scope('dis_D1'):
            kernel_D1 = tf.Variable(tf.random_normal([5,5,16,8],mean=0.0,stddev=0.01),name='kernel_D1')
            conv_D1 = tf.nn.conv2d(conv_pre,kernel_D1,[1,1,1,1],padding='SAME',name='conv_D1')
            abs_D1 = tf.abs(conv_D1,name='abs_D1')
            bn_D1 = tf.layers.batch_normalization(abs_D1,is_training,name= 'bn_D1')
            tanh_D1 = tf.nn.tanh(bn_D1,name='tanh_D1')
            pool_D1 = tf.nn.avg_pool(tanh_D1,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME',name='pool_D1')

        with tf.variable_scope('dis_D2'):
            kernel_D2 = tf.Variable(tf.random_normal([5,5,8,16],mean=0.0,stddev=0.01),name='kernel_D2')
            conv_D2 = tf.nn.conv2d(pool_D1,kernel_D2,[1,1,1,1],padding='SAME',name='conv_D2')
            abs_D2 = tf.abs(conv_D2,name='abs_D2')
            bn_D2 = tf.layers.batch_normalization(abs_D2,is_training,name= 'bn_D2')
            tanh_D2 = tf.nn.tanh(bn_D2,name='tanh_D2')
            pool_D2 = tf.nn.avg_pool(tanh_D2,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME',name='pool_D2')

        with tf.variable_scope('dis_D3'):
            kernel_D3 = tf.Variable(tf.random_normal([1,1,16,32],mean=0.0,stddev=0.01),name='kernel_D3')
            conv_D3 = tf.nn.conv2d(pool_D2,kernel_D3,[1,1,1,1],padding='SAME',name='conv_D3')
            abs_D3 = tf.abs(conv_D3,name='abs_D3')
            bn_D3 = tf.layers.batch_normalization(abs_D3,is_training,name= 'bn_D3')
            tanh_D3 = tf.nn.tanh(bn_D3,name='tanh_D3')
            pool_D3 = tf.nn.avg_pool(tanh_D3,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME',name='pool_D3')

        with tf.variable_scope('dis_D4'):
            kernel_D4 = tf.Variable(tf.random_normal([1,1,32,64],mean=0.0,stddev=0.01),name='kernel_D4')
            conv_D4 = tf.nn.conv2d(pool_D3,kernel_D4,[1,1,1,1],padding='SAME',name='conv_D4')
            abs_D4 = tf.abs(conv_D4,name='abs_D4')
            bn_D4 = tf.layers.batch_normalization(abs_D4,is_training,name= 'bn_D4')
            tanh_D4 = tf.nn.tanh(bn_D4,name='tanh_D4')
            pool_D4 = tf.nn.avg_pool(tanh_D4,ksize=[1,5,5,1],strides=[1,2,2,1],padding='SAME',name='pool_D4')

        with tf.variable_scope('dis_D5'):
            kernel_D5 = tf.Variable(tf.random_normal([1,1,64,128],mean=0.0,stddev=0.01),name='kernel_D5')
            conv_D5 = tf.nn.conv2d(pool_D4,kernel_D5,[1,1,1,1],padding='SAME',name='conv_D5')
            abs_D5 = tf.abs(conv_D5,name='abs_D5')
            bn_D5 = tf.layers.batch_normalization(abs_D5,is_training,name= 'bn_D5')
            tanh_D5 = tf.nn.tanh(bn_D5,name='tanh_D5')
            pool_D5 = tf.nn.avg_pool(tanh_D5,ksize=[1,16,16,1],strides=[1,1,1,1],padding='VALID',name='pool_D5')

        with tf.variable_scope('dis_D6'):#dense layer
            pool_shape_D = tf.shape(pool_D5)
            pool_reshape_D = tf.reshape(pool_D5,[pool_shape_D[0], pool_shape_D[1] * pool_shape_D[2] * pool_shape_D[3]])
            weights_D = tf.Variable(tf.random_normal([128, 2], mean=0.0, stddev=0.01), name="weights_D")
            bias_D = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="bias_D")
            label_D_out = tf.add(tf.matmul(pool_reshape_D, weights_D), bias_D)

    return label_D_out




