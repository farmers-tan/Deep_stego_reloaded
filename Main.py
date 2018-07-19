# Time    : 2018/7/18 下午 1:13
# Author  : Ruandy

import tensorflow as tf
import Misc
import matplotlib.pyplot as plt
import numpy as np

# hyper para
FILE_LIST_cover = 'F:\\datasets\\tiny-imagenet-200\\val\\images\\val_'
EPOCH = 1
BATCH_SIZE = 10
RANK3_NUM = 3
ITR = int(10000/(BATCH_SIZE*RANK3_NUM))
IMG_SIZE = 64
CHANNEL = 3

# main framework
def main(_):
    input_tensor_cover = tf.placeholder(shape=[BATCH_SIZE,IMG_SIZE,IMG_SIZE,CHANNEL],dtype=tf.float32)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for ep in range(EPOCH):
            counter = 0

            for itr in range(ITR):
                input_np_cover,counter = Misc.get_img_batch_rank3(FILE_LIST_cover,BATCH_SIZE,counter,RANK3_NUM)
                print(np.shape(input_np_cover))

                # [output_tmp] = sess.run([input_tensor_cover],feed_dict={input_tensor_cover:input_np_cover})
                # out = np.array(output_tmp, np.int32)
                plt.imshow(input_np_cover[0,:,:,0])
                plt.show()




if __name__ == "__main__":
    tf.app.run()