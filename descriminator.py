import tensorflow as tf

def des_hiding(cover_tensor, container_tensor):

    des_h_input = tf.concat([cover_tensor,container_tensor],0)

        with tf.variable_scope("des_hiding_1"):
            des_h_conv_1 = tf.layers.conv2d(inputs = des_h_input, filters=20, kernel_size=4,strides=4,padding='same',name='des_h_conv_1', kernel_initializer= tf.random_normal_initializer(0, 0.02))
            des_h_bn_1 = tf.layers.batch_normalization(inputs = des_h_conv_1,axis=3,momentum=0.1,epsilon=0.0001,training=True,name='des_h_bn_1')
            des_h_relu_1 = tf.nn.relu(des_h_bn_1,name = 'des_h_relu_1')
        with tf.variable_scope("des_hiding_2"):
            des_h_conv_2 = tf.layers.conv2d(inputs = des_h_relu_1, filters=20, kernel_size=4,strides=4,padding='same',name='des_h_conv_2', kernel_initializer= tf.random_normal_initializer(0, 0.02))
            des_h_bn_2 = tf.layers.batch_normalization(inputs = des_h_conv_2,axis=3,momentum=0.1,epsilon=0.0001,training=True,name='des_h_bn_2')
            des_h_relu_2 = tf.nn.relu(des_h_bn_2,name = 'des_h_relu_2')
        with tf.variable_scope("des_hiding_3"):
            des_h_conv_3 = tf.layers.conv2d(inputs = des_h_relu_2, filters=20, kernel_size=4,strides=4,padding='same',name='des_h_conv_3',kernel_initializer=tf.random_normal_initializer(0,0.02))
            des_h_bn_3 = tf.layers.batch_normalization(inputs = des_h_conv_3,axis=3,momentum=0.1,epsilon=0.0001,training=True,name='des_h_bn_3')
            des_h_relu_3 = tf.nn.relu(des_h_bn_3,name = 'des_h_relu3')
        with tf.variable_scope("des_hiding_4"):
            des_h_conv_4 = tf.layers.conv2d(inputs = des_h_relu_3,filters=20 , kernel_size=4,strides=4,padding='same',name='des_h_conv_4',kernel_initializer=tf.random_normal_initializer(0,0.02))
            des_h_bn_4 = tf.layers.batch_normalization(inputs=des_h_conv_4,axis=3,momentum=0.1,epsilon=0.0001,training=True,name='des_h_bn_4')
            des_h_relu_4 = tf.nn.relu(des_h_bn_4,name = 'des_h_relu_4')
        with tf.variable_scope("des_hiding_fc"):
            des_h_weights = tf.Variable(tf.random_normal([20,2],mean=0.0,stddev=0.01),name='des_h_weights')
            des_h_biases = tf.Variable(tf.random_normal([2],mean=0.0,stddev=0.01),name='des_h_biases')
            des_h_out = tf.nn.softmax(tf.add(tf.multiply(des_h_relu_4,des_h_weights),des_h_biases))
        
        return des_h_out

            
def des_reveal(cover_tensor, container_tensor):
    des_r_input = tf.concat([cover_tensor, container_tensor], 0)

    with tf.variable_scope("des_rev_1"):
        des_r_conv_1 = tf.layers.conv2d(inputs=des_r_input, filters=20, kernel_size=4, strides=4, padding='same',name='des_r_conv_1', kernel_initializer=tf.random_normal_initializer(0, 0.02))
        des_r_bn_1 = tf.layers.batch_normalization(inputs=des_r_conv_1, axis=3, momentum=0.1, epsilon=0.0001,training=True, name='des_r_bn_1')
        des_r_relu_1 = tf.nn.relu(des_r_bn_1, name='des_r_relu_1')
    with tf.variable_scope("des_rev_2"):
        des_r_conv_2 = tf.layers.conv2d(inputs=des_r_relu_1, filters=20, kernel_size=4, strides=4, padding='same',name='des_r_conv_2', kernel_initializer=tf.random_normal_initializer(0, 0.02))
        des_r_bn_2 = tf.layers.batch_normalization(inputs=des_r_conv_2, axis=3, momentum=0.1, epsilon=0.0001,training=True, name='des_r_bn_2')
        des_r_relu_2 = tf.nn.relu(des_r_bn_2, name='des_r_relu_2')
    with tf.variable_scope("des_rev_3"):
        des_r_conv_3 = tf.layers.conv2d(inputs=des_r_relu_2, filters=20, kernel_size=4, strides=4, padding='same',name='des_r_conv_3', kernel_initializer=tf.random_normal_initializer(0, 0.02))
        des_r_bn_3 = tf.layers.batch_normalization(inputs=des_r_conv_3, axis=3, momentum=0.1, epsilon=0.0001,training=True, name='des_r_bn_3')
        des_r_relu_3 = tf.nn.relu(des_r_bn_3, name='des_r_relu3')
    with tf.variable_scope("des_rev_4"):
        des_r_conv_4 = tf.layers.conv2d(inputs=des_r_relu_3, filters=20, kernel_size=4, strides=4, padding='same',name='des_r_conv_4', kernel_initializer=tf.random_normal_initializer(0, 0.02))
        des_r_bn_4 = tf.layers.batch_normalization(inputs=des_r_conv_4, axis=3, momentum=0.1, epsilon=0.0001,training=True, name='des_r_bn_4')
        des_r_relu_4 = tf.nn.relu(des_r_bn_4, name='des_r_relu_4')
    with tf.variable_scope("des_rev_fc"):
        des_r_weights = tf.Variable(tf.random_normal([20, 2], mean=0.0, stddev=0.01), name='des_r_weights')
        des_r_biases = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name='des_r_biases')
        des_r_out = tf.nn.softmax(tf.add(tf.multiply(des_r_relu_4, des_r_weights), des_r_biases))

    return des_r_out

