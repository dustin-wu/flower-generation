import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2

class Generator(tf.keras.Model):
    def __init__(self, z_dim, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.gen_linear1 = tf.Variable(tf.random.truncated_normal(shape=[self.z_dim, self.gen_dim], stddev=.1))
        self.gen_b1 = tf.Variable(tf.random.truncated_normal(shape=[self.gen_dim], stddev=.1))

        self.gen_linear2 = tf.Variable(tf.random.truncated_normal(shape=[self.gen_dim, self.gen_dim], stddev=.1))
        self.gen_b2 = tf.Variable(tf.random.truncated_normal(shape=[self.gen_dim], stddev=.1))

        self.gen_linear3 = tf.Variable(tf.random.truncated_normal(shape=[self.gen_dim, self.z_vector_dim], stddev=.1))
        self.gen_b3 = tf.Variable(tf.random.truncated_normal(shape=[self.z_vector_dim], stddev=.1))

        def call(self, z):
            gen1 = tf.matmul(z, self.gen_linear1) + self.gen_b1
            gen1 = tf.nn.leaky_relu(gen1, alpha=0.02)
            gen2 = tf.matmul(gen1, self.gen_linear2) + self.gen_b2
            gen2 = tf.nn.leaky_relu(gen2, alpha=0.02)
            gen3 = tf.matmul(gen2, self.gen_linear3) + self.gen_b3
            return tf.nn.sigmoid(gen3)

class Discriminator(tf.keras.Model):
    def __init__(self, z_dim, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.dis_linear1 = tf.Variable(tf.random.truncated_normal(shape=[self.z_vector_dim, self.dis_dim], stddev=.1))
        self.dis_b1 = tf.Variable(tf.random.truncated_normal(shape=[self.dis_dim], stddev=.1))

        self.dis_linear2 = tf.Variable(tf.random.truncated_normal(shape=[self.dis_dim, self.dis_dim], stddev=.1))
        self.dis_b2 = tf.Variable(tf.random.truncated_normal(shape=[self.dis_dim], stddev=.1))

        self.dis_linear3 = tf.Variable(tf.random.truncated_normal(shape=[self.dis_dim, 1], stddev=.1))
        self.dis_b3 = tf.Variable(tf.random.truncated_normal(shape=[1], stddev=.1))

        def call(self, z_vector):
            dis1 = tf.matmul(z_vector, self.dis_linear1) + self.dis_b1
            dis1 = tf.nn.leaky_relu(dis1, alpha=0.02)
            dis2 = tf.matmul(dis1, self.dis_linear2) + self.dis_b2
            dis2 = tf.nn.leaky_relu(dis2, alpha=0.02)
            dis3 = tf.matmul(dis2, self.dis_linear3) + self.dis_b3
            return dis3

class GAN(tf.keras.Model):
    def __init__(self, z_dim, dataset_name, data_dir, train, epochs, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.epochs = epochs

        self.generator = Generator(z_dim, z_vector_dim, gen_dim, dis_dim)
        self.discriminator = Generator(z_dim, z_vector_dim, gen_dim, dis_dim)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
        self.dis_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)

        self.gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.gen_optimizer, net=self.generator)
        self.gen_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)
        self.dis_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.dis_optimizer, net=self.discriminator)
        self.dis_manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)

        if os.path.exists(self.data_dir+'/'+self.dataset_namez+'.hdf5'):
            self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_namez+'.hdf5', 'r')
            self.data_z = self.data_dict['zs'][:]
            if (self.z_vector_dim!=self.data_z.shape[1]):
                print("error: self.z_vector_dim!=self.data_z.shape")
                exit(0)
        else:
            if train:
                print("error: cannot load "+self.data_dir+'/'+self.dataset_namez+'.hdf5')
                exit(0)
            else:
                print("warning: cannot load "+self.data_dir+'/'+self.dataset_namez+'.hdf5')

    def train(self):
        self.gen_ckpt.restore(self.gen_manager.latest_checkpoint)
        if self.gen_manager.latest_checkpoint:
            print("Restored generator from {}".format(self.gen_manager.latest_checkpoint))
        else:
            print("Initializing generator from scratch.")

        self.dis_ckpt.restore(self.dis_manager.latest_checkpoint)
        if self.dis_manager.latest_checkpoint:
            print("Restored discriminator from {}".format(self.dis_manager.latest_checkpoint))
        else:
            print("Initializing discriminator from scratch.")

        start_time = time.time()

        batch_index_num = len(self.data_z)
        batch_index_list = np.arange(batch_index_num)
        batch_size = 50
        batch_num = int(batch_index_num/batch_size)

        for epoch in range(self.epochs):
            np.random.shuffle(batch_index_list)
            total_gen_err = 0
            total_dis_err = 0

            for idx in range(batch_num):
                batch_z = np.random.normal(0, 0.2, [batch_size, self.z_dim]).astype(np.float32)
                batch_vector_z = self.data_z[idx*batch_size:(idx+1)*batch_size]
                
                with tf.GradientTape() as tape:
                    d_real = self.discriminator(batch_vector_z)
                    gen_output = self.generator(batch_z)
                    d_fake = self.discriminator(gen_output)
                    gen_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

                train_vars = self.discriminator.trainable_variables
                gradients = tape.gradient(gen_loss, train_vars)
                self.gen_optimizer.apply_gradients(zip(gradients, train_vars))

                with tf.GradientTape() as tape:
                    d_real = self.discriminator(batch_vector_z)
                    gen_output = self.generator(batch_z)
                    d_fake = self.discriminator(gen_output)
                    dis_loss = tf.reduce_mean(d_fake)

                train_vars = self.generator.trainable_variables
                gradients = tape.gradient(dis_loss, train_vars)
                self.dis_optimizer.apply_gradients(zip(gradients, train_vars))

                total_gen_err += gen_loss
                total_dis_err += dis_loss

            print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, gen_loss: %.6f, dis_loss: %.6f" % (epoch, self.epochs, time.time() - start_time, total_gen_err/batch_num, total_dis_err/batch_num))

            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 1000 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def get_z(self, num):
        self.gen_ckpt.restore(self.gen_manager.latest_checkpoint)
        if self.gen_manager.latest_checkpoint:
            print("Restored generator from {}".format(self.gen_manager.latest_checkpoint))
        else:
            print("Initializing generator from scratch.")

        self.dis_ckpt.restore(self.dis_manager.latest_checkpoint)
        if self.dis_manager.latest_checkpoint:
            print("Restored discriminator from {}".format(self.dis_manager.latest_checkpoint))
        else:
            print("Initializing discriminator from scratch.")

        batch_z = np.random.normal(0, 0.2, [num, self.z_dim]).astype(np.float32)
        z_vector = self.generator(batch_z)
        return z_vector
         