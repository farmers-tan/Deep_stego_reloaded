import tensorflow as tf

def prep_net(secret_tensor):
    with tf.variable_scope("prep_net"):
        with tf.variable_scope("3x3_conv"):
            conv_3x3 = tf.layers.conv2d(inputs=secret_tensor, filters=20, kernel_size=3, padding='same', name='1',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same',name='2',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same',name='3',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same',name='4',activation=tf.nn.relu)

        with tf.variable_scope("4x4_conv"):
            conv_4x4 = tf.layers.conv2d(inputs=secret_tensor, filters=20,kernel_size=4,padding='same',name='1',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='2',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='3',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='4',activation=tf.nn.relu)

        with tf.variable_scope("5x5_conv"):
            conv_5x5 = tf.layers.conv2d(inputs=secret_tensor,filters=20,kernel_size=5,padding='same',name='1',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=20,kernel_size=5,padding='same',name='2',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5,filters=20,kernel_size=5,padding='same',name='3',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='4',activation=tf.nn.relu)

        output = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='output')
        return output


def hiding_net(cover_tensor, prep_tensor):
    with tf.variable_scope("hiding_net"):
        concat_input = tf.concat([cover_tensor,prep_tensor],axis=3,name="hiding_input")

        with tf.variable_scope("3x3_conv"):
            conv_3x3 = tf.layers.conv2d(inputs=concat_input, filters=20, kernel_size=3, padding='same', name='1',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='2',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='3',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='4',activation=tf.nn.relu)

        with tf.variable_scope("4x4_conv"):
            conv_4x4 = tf.layers.conv2d(inputs=concat_input, filters=20, kernel_size=4, padding='same', name='1',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='2',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='3',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='4',activation=tf.nn.relu)

        with tf.variable_scope("5x5_conv"):
            conv_5x5 = tf.layers.conv2d(inputs=concat_input, filters=20, kernel_size=5, padding='same', name='1',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='2',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='3',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='4',activation=tf.nn.relu)

        concat_tmp =  tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name="concat_tmp")

        output = tf.layers.conv2d(inputs=concat_tmp,filters=3,kernel_size=1,padding='same',name="output")
        return output


def rev_net(container_tensor):
    with tf.variable_scope("rev_net"):

        with tf.variable_scope("3x3_conv"):
            conv_3x3 = tf.layers.conv2d(inputs=container_tensor, filters=20, kernel_size=3, padding='same', name='1',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='2',activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='3', activation=tf.nn.relu)
            conv_3x3 = tf.layers.conv2d(inputs=conv_3x3, filters=20, kernel_size=3, padding='same', name='4',activation=tf.nn.relu)

        with tf.variable_scope("4x4_conv"):
            conv_4x4 = tf.layers.conv2d(inputs=container_tensor, filters=20, kernel_size=4, padding='same', name='1',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='2',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='3',activation=tf.nn.relu)
            conv_4x4 = tf.layers.conv2d(inputs=conv_4x4, filters=20, kernel_size=4, padding='same', name='4',activation=tf.nn.relu)

        with tf.variable_scope("5x5_conv"):
            conv_5x5 = tf.layers.conv2d(inputs=container_tensor, filters=20, kernel_size=5, padding='same', name='1',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='2',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='3',activation=tf.nn.relu)
            conv_5x5 = tf.layers.conv2d(inputs=conv_5x5, filters=20, kernel_size=5, padding='same', name='4',activation=tf.nn.relu)

        concat_tmp = tf.concat([conv_3x3, conv_4x4, conv_5x5],axis=3, name="concat_tmp")

        output = tf.layers.conv2d(inputs=concat_tmp, filters= 3, kernel_size=1, padding='same',name='output')
        return output






