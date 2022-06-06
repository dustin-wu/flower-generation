import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py

# Generator: generate latent space vectors to feed to the autoencoder, given a randomly sampled vector from its own latent space
class Generator(tf.keras.Model):
    def __init__(self, z_dim=256, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        super().__init__()
        self.z_dim = z_dim # Generator's latent space dimension
        self.z_vector_dim = z_vector_dim # Autoencoder's latent space dimension
        self.gen_dim = gen_dim # Used to determine size of generator architecture
        self.dis_dim = dis_dim # Used to determine size of discriminator architecture

        glorot_uniform = tf.keras.initializers.GlorotUniform()

        self.gen_linear1 = tf.Variable(glorot_uniform(shape=[self.z_dim, self.gen_dim]))
        self.gen_b1 = tf.Variable(tf.zeros(shape=[self.gen_dim]))

        self.gen_linear2 = tf.Variable(glorot_uniform(shape=[self.gen_dim, self.gen_dim]))
        self.gen_b2 = tf.Variable(tf.zeros(shape=[self.gen_dim]))

        self.gen_linear3 = tf.Variable(glorot_uniform(shape=[self.gen_dim, self.z_vector_dim]))
        self.gen_b3 = tf.Variable(tf.zeros(shape=[self.z_vector_dim]))

    # MLP forward pass, with sigmoid activation 
    def call(self, z):
        gen1 = tf.matmul(z, self.gen_linear1) + self.gen_b1
        gen1 = tf.nn.leaky_relu(gen1, alpha=0.02)
        gen2 = tf.matmul(gen1, self.gen_linear2) + self.gen_b2
        gen2 = tf.nn.leaky_relu(gen2, alpha=0.02)
        gen3 = tf.matmul(gen2, self.gen_linear3) + self.gen_b3
        return tf.nn.sigmoid(gen3) # Since the autoencoder uses a sigmoid activation, the generator should have one too

# Discriminator: assign a "realness" score to the given vector in terms of coming from the distribution of autoencoder latent vectors
# Outputs logits according to Wasserstein GAN (WGAN); would more appropriately be called a critic network
class Discriminator(tf.keras.Model):
    def __init__(self, z_dim=256, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        super().__init__()
        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        glorot_uniform = tf.keras.initializers.GlorotUniform()

        self.dis_linear1 = tf.Variable(glorot_uniform(shape=[self.z_vector_dim, self.dis_dim]))
        self.dis_b1 = tf.Variable(tf.zeros(shape=[self.dis_dim]))

        self.dis_linear2 = tf.Variable(glorot_uniform(shape=[self.dis_dim, self.dis_dim]))
        self.dis_b2 = tf.Variable(tf.zeros(shape=[self.dis_dim]))

        self.dis_linear3 = tf.Variable(glorot_uniform(shape=[self.dis_dim, 1]))
        self.dis_b3 = tf.Variable(tf.zeros(shape=[1]))

    def call(self, z_vector):
        dis1 = tf.matmul(z_vector, self.dis_linear1) + self.dis_b1
        dis1 = tf.nn.leaky_relu(dis1, alpha=0.02)
        dis2 = tf.matmul(dis1, self.dis_linear2) + self.dis_b2
        dis2 = tf.nn.leaky_relu(dis2, alpha=0.02)    
        dis3 = tf.matmul(dis2, self.dis_linear3) + self.dis_b3
        return dis3 # WGAN = no output activation

# Entire network class including training and evaluation functions
class IMGAN(tf.keras.Model):
    def __init__(self, config, z_dim=256, z_vector_dim=256, gen_dim=2048, dis_dim=2048):
        super().__init__()
        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.epochs = config.epoch
        self.data_dir = config.data_dir
        self.do_train = config.train
        self.dataset_namez = config.dataset + '_train_z'

        self.generator = Generator(z_dim, z_vector_dim, gen_dim, dis_dim)
        self.discriminator = Generator(z_dim, z_vector_dim, gen_dim, dis_dim)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1)
        self.dis_optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1)
        self.dis_steps = 5 # WGAN: Train the discriminator for multiple steps before training the generator once
        
        self.gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.gen_optimizer, net=self.generator)
        self.gen_manager = tf.train.CheckpointManager(self.gen_ckpt, config.checkpoint_dir+"/gen", max_to_keep=10)
        self.dis_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.dis_optimizer, net=self.discriminator)
        self.dis_manager = tf.train.CheckpointManager(self.dis_ckpt, config.checkpoint_dir+"/dis", max_to_keep=10)

        if os.path.exists(self.data_dir+'/'+self.dataset_namez+'.hdf5'):
            self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_namez+'.hdf5', 'r')
            self.data_z = self.data_dict['zs'][:]
            if (self.z_vector_dim!=self.data_z.shape[1]):
                print("error: self.z_vector_dim!=self.data_z.shape")
                exit(0)
        else:
            if self.do_train:
                print("error: cannot load "+self.data_dir+'/'+self.dataset_namez+'.hdf5')
                exit(0)
            else:
                print("warning: cannot load "+self.data_dir+'/'+self.dataset_namez+'.hdf5')

    # Gradient penalty encourages the model to keep weight magnitudes low, which is vital to WGANs in preventing mode collapse
    def gradient_penalty(self, batch_size, real, fake):
        alpha = np.random.normal(0, 1, [batch_size, self.z_dim]).astype(np.float32)
        diff = fake - real
        interpolated = real + alpha * diff
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            dis_output = self.discriminator(interpolated)

        grads = tape.gradient(dis_output, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        grad_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return grad_penalty
                
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
        batch_size = 128
        batch_num = int(batch_index_num/batch_size)

        for epoch in range(self.epochs):
            np.random.shuffle(batch_index_list)
            total_gen_err = 0
            total_dis_err = 0

            for idx in range(batch_num):
                batch_vector_z = self.data_z[idx*batch_size:(idx+1)*batch_size]
                
                # Train discriminator for multiple steps per generator step
                for i in range(self.dis_steps):
                    batch_z = np.random.normal(0, 1, [batch_size, self.z_dim]).astype(np.float32)                    
                    with tf.GradientTape() as tape:
                        d_real = self.discriminator(batch_vector_z)
                        gen_output = self.generator(batch_z)
                        d_fake = self.discriminator(gen_output)
                        dis_loss = -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)) # WGAN critic loss
                        grad_penalty = self.gradient_penalty(batch_size, batch_vector_z, gen_output)
                        dis_loss += 10 * grad_penalty

                    train_vars = self.discriminator.trainable_variables
                    gradients = tape.gradient(dis_loss, train_vars)
                    self.dis_optimizer.apply_gradients(zip(gradients, train_vars))
                
                # Generator step
                batch_z = np.random.normal(0, 1, [batch_size, self.z_dim]).astype(np.float32)
                with tf.GradientTape() as tape:
                    gen_output = self.generator(batch_z)
                    d_fake = self.discriminator(gen_output)
                    gen_loss = -tf.reduce_mean(d_fake) # WGAN generator loss

                train_vars = self.generator.trainable_variables
                gradients = tape.gradient(gen_loss, train_vars)
                self.gen_optimizer.apply_gradients(zip(gradients, train_vars))

                total_gen_err += gen_loss
                total_dis_err += dis_loss

            print("Epoch: [%2d/%2d] time: %4.4f, gen_loss: %.6f, dis_loss: %.6f" % (epoch, self.epochs, time.time() - start_time, total_gen_err/batch_num, total_dis_err/batch_num))

            self.gen_ckpt.step.assign_add(1)
            if int(self.gen_ckpt.step) % 10 == 0:
                save_path = self.gen_manager.save()
                print("Saved generator checkpoint for step {}: {}".format(int(self.gen_ckpt.step), save_path))
            
            self.dis_ckpt.step.assign_add(1)
            if int(self.dis_ckpt.step) % 10 == 0:
                save_path = self.dis_manager.save()
                print("Saved discriminator checkpoint for step {}: {}".format(int(self.dis_ckpt.step), save_path))

    # Get a certain number of randomly sampled latent space vectors for creating novel objects with the autoencoder
    def generate_samples(self, num):
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

        batch_z = np.random.normal(0, 1, [num, self.z_dim]).astype(np.float32)
        z_vector = self.generator(batch_z)
        return z_vector