import tensorflow as tf
import numpy as np
import os
import glob
import random
import Image
import net_struct



## sys_para
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = "/HDD/rdy/dataset/color_256" #250


## hyper_para
BATCH_SIZE = 10 # size 256 RGB
LR = .0001
BETA =

## file input

File_list = glob.glob(join(PATH,'*'))

def get_img_batch(file_list,batch_size,img_size = (256,256)):
    secret_batch = []
    cover_batch = []

    for i in range(batch_size):
        secret_path =  random.choice(file_list)
        cover_path = random.choice(file_list)
        secret_img = Image.open(secret_path).convert("RGB")
        cover_img = Image.open(cover_path).convert("RGB")

        secret_img =  np.array(ImageOps.fit(secret_img,img_size),dtype = np.float32)/255
        cover_img = np.array(ImageOps.fit(cover_img, img_size), dtype=np.float32) / 255

        secret_batch.append(secret_img)
        cover_batch.append(cover_img)

    secret_batch = np.array(secret_batch)
    cover_batch = np.array(cover_batch)

    return secret_batch,cover_batch

def add_noise(tensor, std= .1):
    with tf.variable_scope("noise_layer"):
        return tensor + tf.random_normal(shape = tf.shape(tensor), mean= 0.0, stddev=std, dtype=np.float32)

def get_loss(secret_true,secret_pred,cover_ture,cover_pred):
    ##modification in progress

def tensor2img(tensor):
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        return tf.clip_by_value(tensor,0,1)

def training_graph(secret_tensor,cover_tensor,global_step):
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        prep_output = net_struct.prep_net()
        




