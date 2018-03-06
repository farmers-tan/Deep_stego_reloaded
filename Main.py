import tensorflow as tf
import numpy as np
import os
import Image
from PIL import ImageOps
import glob
import random
import Image
from os.path import join
import net_struct
import descriminator
import time

## sys_para
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
PATH = "/HDD/rdy/dataset/color_256" #250

## hyper_para
BATCH_SIZE = 20 # size 256 RGB
LR = .0001
BETA = .5
epoch_step = 10000

## file input

File_list = glob.glob(join(PATH,"*"))

# def normalize_batch(imgs):
#     return(imgs - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

def get_img_batch(file_list,batch_size,img_size = (256,256),normalize = True):
    secret_batch = []
    cover_batch = []

    for i in range(batch_size):
        secret_path =  random.choice(file_list)
        cover_path = random.choice(file_list)
        secret_img = Image.open(secret_path).convert("RGB")
        cover_img = Image.open(cover_path).convert("RGB")

        secret_img =  np.array(ImageOps.fit(secret_img,img_size),dtype = np.float32)/255
        cover_img = np.array(ImageOps.fit(cover_img, img_size), dtype=np.float32)/255

        secret_batch.append(secret_img)
        cover_batch.append(cover_img)

    secret_batch = np.array(secret_batch)
    cover_batch = np.array(cover_batch)

    # if normalize == True:
    #     secret_batch = normalize_batch(secret_batch)
    #     cover_batch = normalize_batch(cover_batch)

    return secret_batch,cover_batch

def add_noise(tensor, std= .1):
    with tf.variable_scope("noise_layer"):
        return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=np.float32)

def get_loss(secret_true,secret_pred,cover_true,cover_pred,beta = BETA):
    ##modification in progress
    with tf.variable_scope("losses"):
        beta = tf.constant(beta, name="beta")
        secret_mse = tf.losses.mean_squared_error(secret_true, secret_pred)
        cover_mse = tf.losses.mean_squared_error(cover_true, cover_pred)
        final_loss = cover_mse + beta * secret_mse
        return final_loss, secret_mse, cover_mse

def tensor2img(tensor):
    with tf.variable_scope("tensor2img",reuse=tf.AUTO_REUSE):
        # t = tensor * tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
        return tf.clip_by_value(tensor,0,1)

def training_graph(secret_tensor,cover_tensor,global_step):
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        prep_output = net_struct.prep_net(secret_tensor)
        hiding_output = net_struct.hiding_net(cover_tensor, prep_output)
        noise_added_output = add_noise(hiding_output)
        rev_output = net_struct.rev_net(noise_added_output)

        loss,rev_loss,cover_loss = get_loss(secret_tensor,rev_output,cover_tensor, hiding_output)

        optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step= global_step)

        tf.summary.scalar("loss",loss,family="train")
        tf.summary.scalar("rev_loss",rev_loss,family="train")
        tf.summary.scalar("cover_loss",cover_loss, family="train")
        tf.summary.image("secret_img", tensor2img(secret_tensor), max_outputs=1, family="train")
        tf.summary.image("cover_img" , tensor2img(cover_tensor), max_outputs=1, family="train")
        tf.summary.image("hidding_img", tensor2img(hiding_output), max_outputs=1, family="train")
        tf.summary.image("reveal_img", tensor2img(rev_output), max_outputs=1, family="train")

        merged_summary = tf.summary.merge_all()

        return optimizer,merged_summary


def test_graph(secret_tensor, cover_tensor):
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        prep_output = net_struct.prep_net(secret_tensor)
        hiding_output = net_struct.hiding_net(cover_tensor, prep_output)
        rev_output = net_struct.rev_net(hiding_output)

        loss, rev_loss, cover_loss = get_loss(secret_tensor, rev_output, cover_tensor, hiding_output)

        tf.summary.scalar("loss", loss, family="test")
        tf.summary.scalar("rev_loss", rev_loss, family="test")
        tf.summary.scalar("cover_loss", cover_loss, family="test")
        tf.summary.image("secret_img", tensor2img(secret_tensor), max_outputs=1, family="test")
        tf.summary.image("cover_img", tensor2img(cover_tensor), max_outputs=1, family="test")
        tf.summary.image("hidding_img", tensor2img(hiding_output), max_outputs=1, family="test")
        tf.summary.image("reveal_img", tensor2img(rev_output), max_outputs=1, family="test")

        merged_summary = tf.summary.merge_all()

        return merged_summary


with tf.Session() as sess:
    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")

    secret_tensor = tf.placeholder(shape=[None, 256, 256, 3],dtype=tf.float32,name="prep_input")
    cover_tensor = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32, name="hiding_input")
    writer = tf.summary.FileWriter("./logger")

    train_op, summary_op = training_graph(secret_tensor, cover_tensor, global_step_tensor)
    test_op = test_graph(secret_tensor,cover_tensor)

    saver = tf.train.Saver(max_to_keep=5)
    print('initializing...')
    print('lr = %f'%LR)
    print('beta = %f'%BETA)
    print('batch size = %d'%BATCH_SIZE)
    time.sleep(5)

    tf.global_variables_initializer().run()
    writer.add_graph(sess.graph)


    for ep in range(10):
        for itr in range(epoch_step):
            secret_img, cover_img = get_img_batch(file_list=File_list,batch_size = BATCH_SIZE)

            sess.run([train_op],feed_dict={secret_tensor:secret_img, cover_tensor:cover_img})
            os.system("clear")
            print("Training_itr_%d"%itr+" epoch %d"%ep)


            if itr % 10 == 0:
                summary,global_step = sess.run([summary_op, global_step_tensor], feed_dict={secret_tensor:secret_img, cover_tensor:cover_img})
                writer.add_summary(summary,global_step)


            if itr % 100 == 0: #test section mulfunction need to debug
                secret_img_test, cover_img_test = get_img_batch(file_list=File_list,batch_size=1)
                summary, global_step = sess.run([test_op, global_step_tensor],feed_dict={secret_tensor:secret_img_test,cover_tensor:cover_img_test})
                writer.add_summary(summary, global_step)

            save_model = saver.save(sess,"./model/"+"%d"%ep+".ckpt")

    writer.close()





