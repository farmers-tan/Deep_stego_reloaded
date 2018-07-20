# Time    : 2018/7/18 下午 1:13
# Author  : Ruandy

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import Misc
import hiding_net
import rev_net

# hyper para
FILE_LIST_cover = 'F:\\datasets\\tiny-imagenet-200\\val\\images\\val_'
FILE_LIST_msg = 'F:\\datasets\\tiny-imagenet-200\\test\\images\\test_'
EPOCH = 100
BATCH_SIZE = 2
RANK3_NUM = 3
ITR = int(10000/(BATCH_SIZE*RANK3_NUM))
IMG_SIZE = 64
CHANNEL = 3
LR = 0.001

# main framework
def main(_):
    input_tensor_cover = tf.placeholder(shape=[BATCH_SIZE,IMG_SIZE,IMG_SIZE,CHANNEL*RANK3_NUM],dtype=tf.float32)
    input_tensor_msg = tf.placeholder(shape=[BATCH_SIZE,IMG_SIZE,IMG_SIZE,CHANNEL],dtype=tf.float32)
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    contain_tensor = hiding_net.hiding_net(input_tensor_cover,input_tensor_msg,RANK3_NUM)
    rev_tensor = rev_net.rev_net(contain_tensor)

    loss_hiding = tf.losses.mean_squared_error(input_tensor_cover, contain_tensor)
    loss_rev = tf.losses.mean_squared_error(input_tensor_msg, rev_tensor)

    opt_hiding = tf.train.AdamOptimizer(LR).minimize(loss_hiding,global_step=global_step_tensor)
    opt_rev = tf.train.AdamOptimizer(LR).minimize(loss_rev)

    tf.summary.image('cover',input_tensor_cover[:,:,:,0:3], max_outputs=1)
    tf.summary.image('contain', contain_tensor[:,:,:,0:3], max_outputs=1)
    tf.summary.image('msg',input_tensor_msg, max_outputs=1)
    tf.summary.image('rev',rev_tensor, max_outputs=1)

    tf.summary.scalar('loss_hide',loss_hiding)
    tf.summary.scalar('loss_rev',loss_rev)
    merged_summary = tf.summary.merge_all()




    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('./log', sess.graph)

        for ep in range(EPOCH):
            counter_cover = 0
            counter_msg = 0

            for itr in range(ITR):
                input_np_cover,counter_cover = Misc.get_img_batch_rank3(FILE_LIST_cover,BATCH_SIZE,counter_cover,RANK3_NUM)
                input_np_msg,counter_msg = Misc.get_img_batch(FILE_LIST_msg,BATCH_SIZE,counter_msg)

                _,_,summary,step = sess.run([opt_hiding,opt_rev,merged_summary,global_step_tensor],feed_dict={input_tensor_cover:input_np_cover, input_tensor_msg:input_np_msg})
                writer.add_summary(summary,step)
                print('training ep %d '%ep + 'itr %d' %itr)
                # print(np.shape(input_np_cover))
                #
                #
                # # [output_tmp] = sess.run([input_tensor_cover],feed_dict={input_tensor_cover:input_np_cover})
                # # out = np.array(output_tmp, np.int32)
                # plt.imshow(input_np_cover[0,:,:,1])
                # plt.show()


if __name__ == "__main__":
    tf.app.run()