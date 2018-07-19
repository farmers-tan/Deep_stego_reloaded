# Time    : 2018/7/18 下午 2:51
# Author  : Ruandy

import tensorflow as tf

def hiding_net(cover_tensor,msg_tensor):
    with tf.variable_scope('Hiding_net'):
        concat_input = tf.concat([cover_tensor,msg_tensor],axis=3)

        with tf.variable_scope("3x3_conv_branch"):
            conv_3x3 = tf.layers.conv2d(inputs=concat_input, filters=10, kernel_size=3, padding='same', name="1",activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=10, kernel_size=3, padding='same', name="2",activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=10, kernel_size=3, padding='same', name="3",activation=tf.nn.relu)

        with tf.variable_scope("4x4_conv_branch"):
            conv_4x4 = tf.layers.conv2d(inputs=concat_input, filters=10, kernel_size=4, padding='same', name="1",activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=10, kernel_size=4, padding='same', name="2",activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=10, kernel_size=4, padding='same', name="3",activation=tf.nn.relu)

        with tf.variable_scope("5x5_conv_branch"):
            conv_5x5 = tf.layers.conv2d(inputs=concat_input, filters=10, kernel_size=5, padding='same', name="1",activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=10, kernel_size=5, padding='same', name="2",activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=10, kernel_size=5, padding='same', name="3",activation=tf.nn.relu)

        with tf.variable_scope("concat_conv"):
            concat_output = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3)
            output = tf.layers.conv2d(inputs=concat_output,filters=1, kernel_size=1)