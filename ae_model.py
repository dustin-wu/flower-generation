import os
import time
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
import mcubes
import h5py
from z2voxel import z2voxel, write_ply_triangle

class Encoder(tf.keras.Model):
    def __init__(self, enc_dim, z_dim):
        self.enc_dim = enc_dim # enc_dim will be used to determine the sizes of the encoder architecture
        self.z_dim = z_dim # z is the latent space vector that we encode our objects to

        self.enc_filter1 = tf.Variable(tf.random.truncated_normal(shape=[4, 4, 4, 1, self.enc_dim], stddev=.1))
        self.enc_b1 = tf.Variable(tf.random.truncated_normal(shape=[self.enc_dim], stddev=.1))

        self.enc_filter2 = tf.Variable(tf.random.truncated_normal(shape=[4, 4, 4, self.enc_dim, self.enc_dim*2], stddev=.1))
        self.enc_b2 = tf.Variable(tf.random.truncated_normal(shape=[self.enc_dim*2], stddev=.1))

        self.enc_filter3 = tf.Variable(tf.random.truncated_normal(shape=[4, 4, 4, self.enc_dim*2, self.enc_dim*4], stddev=.1))
        self.enc_b3 = tf.Variable(tf.random.truncated_normal(shape=[self.enc_dim*4], stddev=.1))

        self.enc_filter4 = tf.Variable(tf.random.truncated_normal(shape=[4, 4, 4, self.enc_dim*4, self.enc_dim*8], stddev=.1))
        self.enc_b4 = tf.Variable(tf.random.truncated_normal(shape=[self.enc_dim*8], stddev=.1))

        self.enc_filter5 = tf.Variable(tf.random.truncated_normal(shape=[4, 4, 4, self.enc_dim*8, self.z_dim], stddev=.1))
        self.enc_b5 = tf.Variable(tf.random.truncated_normal(shape=[self.z_dim], stddev=.1))

    def call(self, vox3d):
        enc1 = tf.nn.conv3d(vox3d, self.enc_filter1, strides=[1,2,2,2,1], padding="SAME")
        enc1 = tf.nn.bias_add(enc1, self.enc_b1)
        mean1, var1 = tf.nn.moments(enc1, [0,1,2,3])
        enc1 = tf.nn.batch_normalization(enc1, mean1, var1, variance_epsilon=1e-5, offset=None, scale=None)
        enc1 = tf.nn.leaky_relu(enc1, alpha=0.02)

        enc2 = tf.nn.conv3d(enc1, self.enc_filter2, strides=[1,2,2,2,1], padding="SAME")
        enc2 = tf.nn.bias_add(enc2, self.enc_b2)
        mean2, var2 = tf.nn.moments(enc2, [0,1,2,3])
        enc2 = tf.nn.batch_normalization(enc2, mean2, var2, variance_epsilon=1e-5, offset=None, scale=None)
        enc2 = tf.nn.leaky_relu(enc2, alpha=0.02)

        enc3 = tf.nn.conv3d(enc2, self.enc_filter3, strides=[1,2,2,2,1], padding="SAME")
        enc3 = tf.nn.bias_add(enc3, self.enc_b3)
        mean3, var3 = tf.nn.moments(enc3, [0,1,2,3])
        enc3 = tf.nn.batch_normalization(enc3, mean3, var3, variance_epsilon=1e-5, offset=None, scale=None)
        enc3 = tf.nn.leaky_relu(enc3, alpha=0.02)

        enc4 = tf.nn.conv3d(enc3, self.enc_filter4, strides=[1,2,2,2,1], padding="SAME")
        enc4 = tf.nn.bias_add(enc4, self.enc_b1)
        mean4, var4 = tf.nn.moments(enc4, [0,1,2,3])
        enc4 = tf.nn.batch_normalization(enc4, mean4, var4, variance_epsilon=1e-5, offset=None, scale=None)
        enc4 = tf.nn.leaky_relu(enc4, alpha=0.02)

        enc5 = tf.nn.conv3d(enc4, self.enc_filter5, strides=[1,1,1,1,1], padding="VALID")
        enc5 = tf.nn.bias_add(enc5, self.enc_b5)
        enc5 = tf.nn.sigmoid(enc5)
        
        z = tf.reshape(enc5, [1, self.z_dim])
        return z


class Decoder(tf.keras.Model):
    def __init__(self, z_dim, point_dim=3, dec_dim=3):
        self.z_dim = z_dim # z is the latent space vector that we encode our objects to
        self.point_dim = point_dim # point_dim is the dimensonality of our points
        self.dec_dim = dec_dim # same but for decoder architecture

        self.dec_linear1 = tf.Variable(tf.random.truncated_normal(shape=[self.z_dim + self.point_dim, self.dec_dim*8], stddev=.1))
        self.dec_b1 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8], stddev=.1))

        self.dec_linear2 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8, self.dec_dim*8], stddev=.1))
        self.dec_b2 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8], stddev=.1))

        self.dec_linear3 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8, self.dec_dim*8], stddev=.1))
        self.dec_b3 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8], stddev=.1))

        self.dec_linear4 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*8, self.dec_dim*4], stddev=.1))
        self.dec_b4 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*4], stddev=.1))

        self.dec_linear5 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*4, self.dec_dim*2], stddev=.1))
        self.dec_b5 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*2], stddev=.1))

        self.dec_linear6 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim*2, self.dec_dim], stddev=.1))
        self.dec_b6 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim], stddev=.1))

        self.dec_linear7 = tf.Variable(tf.random.truncated_normal(shape=[self.dec_dim, 1], stddev=.1))
        self.dec_b7 = tf.Variable(tf.random.truncated_normal(shape=[1], stddev=.1))

    def call(self, points, z_vector, is_training=False):
        z_tiled = tf.tile(z_vector, [self.batch_size, 1])
        points_and_z = tf.concat([self.point_coord, z_tiled]) # concat the z vector to the xyz coords

        dec1 = tf.matmul(points_and_z, self.dec_linear1) + self.dec_b1
        dec1 = tf.nn.leaky_relu(dec1, alpha=0.02)

        dec2 = tf.matmul(dec1, self.dec_linear2) + self.dec_b2
        dec2 = tf.nn.leaky_relu(dec2, alpha=0.02)

        dec3 = tf.matmul(dec2, self.dec_linear3) + self.dec_b3
        dec3 = tf.nn.leaky_relu(dec3, alpha=0.02)

        dec4 = tf.matmul(dec3, self.dec_linear4) + self.dec_b4
        dec4 = tf.nn.leaky_relu(dec4, alpha=0.02)

        dec5 = tf.matmul(dec4, self.dec_linear5) + self.dec_b5
        dec5 = tf.nn.leaky_relu(dec5, alpha=0.02)

        dec6 = tf.matmul(dec5, self.dec_linear6) + self.dec_b6
        dec6 = tf.nn.leaky_relu(dec6, alpha=0.02)

        dec7 = tf.matmul(dec6, self.dec_linear7) + self.dec_b7
        dec7 = tf.maximum(tf.minimum(dec7, 1), 0)

        reconstructed_point_value = tf.reshape(dec7, [self.batch_size, 1])
        return reconstructed_point_value


class Autoencoder(tf.keras.Model):
    def __init__(self, enc_dim=32, dec_dim=128, z_dim=256, point_dim=3):
        
        self.enc_dim = enc_dim # enc_dim will be used to determine the sizes of the encoder architecture
        self.dec_dim = dec_dim # same but for decoder architecture
        self.z_dim = z_dim # z is the latent space vector that we encode our objects to
        self.point_dim = point_dim
        self.encoder = Encoder(self.enc_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim, self.point_dim, self.dec_dim)

    def call(self, vox3d, z_vector, point_coord, is_training=False):
        if is_training:
            z_vector = self.encoder(vox3d)
            reconstructed_point_value = self.decoder(point_coord, z_vector)

        else:
            if vox3d is not None:
                z_vector = self.encoder(vox3d)
            if z_vector is not None and point_coord is not None:
                reconstructed_point_value = self.decoder(point_coord, z_vector)
            else:
                reconstructed_point_value = None

        return z_vector, reconstructed_point_value


class IMAE(tf.keras.Model):
    def __init__(self, sample_vox_size, dataset_name, train, data_dir):
        self.sample_vox_size = sample_vox_size
        self.point_batch_size = 16*16*16
        self.shape_batch_size = 32
        self.input_size = 64
        self.epochs = 200

        self.enc_dim = 32
        self.dec_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        self.dataset_name = dataset_name
        self.dataset_load = self.dataset_name + '_train'
        if not train:
            self.dataset_load = self.dataset_name + 'test'
        self.data_dir = data_dir
        data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'

        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
            self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
            self.data_voxels = data_dict['voxels'][:]
            self.load_point_batch_size = self.data_points.shape[1]
            self.data_voxels = np.reshape(self.data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
        else:
            print("error: cannot load "+data_hdf5_name)
            exit(0)

        self.autoencoder = Autoencoder(self.enc_dim, self.dec_dim, self.z_dim, self.point_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.autoencoder)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)

    def loss(self, point_value, reconstructed_point_value):
        return tf.reduce_mean(tf.square(point_value - reconstructed_point_value))

    def train(self):
        batch_index_list = np.arange(len(self.vox3d))
        batch_num = int(len(self.data_voxels)/self.shape_batch_size)
        point_batch_num = int(self.load_point_batch_size//self.point_batch_size)
        start_time = time.time()

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(self.epochs):
            np.random.shuffle(batch_index_list)
            total_loss = 0
            total_num = 0

            for idx in range(batch_num):
                indices_of_batch = batch_index_list[idx*self.shape_batch_size: (idx+1)*self.shape_batch_size]
                batch_voxels = self.vox3d[indices_of_batch].astype(np.float32)

                if point_batch_num==1:
                    point_coord = self.data_points[indices_of_batch]
                    point_value = self.data_values[indices_of_batch]
                else:
                    sample_idx = np.random.randint(point_batch_num)
                    point_coord = self.data_points[indices_of_batch, sample_idx*self.point_batch_size: (sample_idx+1)*self.point_batch_size]
                    point_value = self.data_values[indices_of_batch, sample_idx*self.point_batch_size: (sample_idx+1)*self.point_batch_size]

                with tf.GradientTape() as tape:
                    _, reconstructed_point_value = self.autoencoder(batch_voxels, None, point_coord, is_training=True)
                    batch_loss = self.loss(point_value, reconstructed_point_value)

                gradients = tape.gradient(batch_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                total_loss += batch_loss
                total_num += 1

            print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (epoch, self.epochs, time.time() - start_time, total_loss/total_num))

            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def get_z(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num,self.z_dim], np.float32)

        for t in range(shape_num):
            batch_voxels = self.data_voxels[t: t+1].astype(np.float32)
            out_z,_ = self.autoencoder(batch_voxels, None, None, is_training=False)
            hdf5_file["zs"][t: t+1, :] = out_z.numpy()

        hdf5_file.close()
        print("Created z vectors dataset")

    # Output mesh as ply file
    def test_mesh(self, st, en, sample_dir):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        
        for t in range(st, min(len(self.data_voxels), en)):
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            model_z,_ = self.autoencoder(batch_voxels, None, None, is_training=False)
            model_float = z2voxel(model_z, self.autoencoder)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
            write_ply_triangle(sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)