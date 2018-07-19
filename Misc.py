# Time    : 2018/7/18 下午 1:13
# Author  : Ruandy

import numpy as np
from PIL import Image
from os.path import join

def get_img_batch(file_list, batch_size, counter):  # get a numpy tensor with shape [batch,img_size,img_size,channel] return with a counter

    img_batch = []

    for i in range(batch_size):
        path = join(file_list+str(counter+i)+'.JPEG')
        img = np.array(Image.open(path).convert("RGB"),dtype=np.float32)
        img_batch.append(img)

    counter = counter + batch_size

    return img_batch, counter


def get_img_batch_rank3(file_list, batch_size, counter, num_rank3):  # get a numpy tensor with shape [batch,img_size,img_size,channel*num_channel]

    img_batch = []

    for i in range(num_rank3):

        img_batch_tmp = []

        for ii in range(batch_size):
            path = join(file_list+str(counter+ii)+'.JPEG')
            img = np.array(Image.open(path).convert("RGB"),dtype=np.float32)
            img_batch_tmp.append(img)

        if i == 0:
            img_batch = img_batch_tmp
        else:
            img_batch = np.concatenate((img_batch_tmp,img_batch),axis=3)

        counter = counter + batch_size

    return img_batch,counter



