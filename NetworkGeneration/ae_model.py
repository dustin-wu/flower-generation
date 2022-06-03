import os
import time
import tensorflow as tf
import numpy as np
import mcubes
import h5py
from z2voxel import z2voxel, write_ply_triangle

# Code draws from the Pytorch implementation of the IM-net paper: https://github.com/czq142857/IM-NET-pytorch
# Adapted to use tensorflow 2, which is used in cs1470

# Encoder: Encode voxelized models into latent "z" vectors
class Encoder(tf.keras.Model):
    def __init__(self, enc_dim, z_dim):
        super().__init__()
        self.enc_dim = enc_dim # Used to determine the sizes of the encoder architecture
        self.z_dim = z_dim # Latent space dimension
        
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
        
        
    # The forward pass is akin to a typical CNN, just with 3D convolutions
    def call(self, vox3d):
        enc1 = tf.nn.conv3d(vox3d, self.enc_filter1, strides=[1,2,2,2,1], padding="SAME")
        mean1, var1 = tf.nn.moments(enc1, [0,1,2,3]) 
        enc1 = tf.nn.batch_normalization(enc1, mean1, var1, variance_epsilon=1e-5, offset=None, scale=None)
        enc1 = tf.nn.leaky_relu(enc1, alpha=0.02)

        enc2 = tf.nn.conv3d(enc1, self.enc_filter2, strides=[1,2,2,2,1], padding="SAME")
        mean2, var2 = tf.nn.moments(enc2, [0,1,2,3])
        enc2 = tf.nn.batch_normalization(enc2, mean2, var2, variance_epsilon=1e-5, offset=None, scale=None)
        enc2 = tf.nn.leaky_relu(enc2, alpha=0.02)

        enc3 = tf.nn.conv3d(enc2, self.enc_filter3, strides=[1,2,2,2,1], padding="SAME")
        mean3, var3 = tf.nn.moments(enc3, [0,1,2,3])
        enc3 = tf.nn.batch_normalization(enc3, mean3, var3, variance_epsilon=1e-5, offset=None, scale=None)
        enc3 = tf.nn.leaky_relu(enc3, alpha=0.02)

        enc4 = tf.nn.conv3d(enc3, self.enc_filter4, strides=[1,2,2,2,1], padding="SAME")
        mean4, var4 = tf.nn.moments(enc4, [0,1,2,3])
        enc4 = tf.nn.batch_normalization(enc4, mean4, var4, variance_epsilon=1e-5, offset=None, scale=None)
        enc4 = tf.nn.leaky_relu(enc4, alpha=0.02)

        enc5 = tf.nn.conv3d(enc4, self.enc_filter5, strides=[1,1,1,1,1], padding="VALID")
        enc5 = tf.nn.bias_add(enc5, self.enc_b5)
        enc5 = tf.nn.sigmoid(enc5)
        
        z = tf.reshape(enc5, [-1, self.z_dim])
        return z

# Decoder: Recovers 3D object from latent vector by learning a function that maps points to point values: 0 for outside the object, 1 for inside 
class Decoder(tf.keras.Model):
    def __init__(self, z_dim, point_dim=3, dec_dim=3):
        super().__init__()
        self.z_dim = z_dim # Latent space dimension
        self.point_dim = point_dim # Dimensonality of the points
        self.dec_dim = dec_dim # Used to determine the sizes of the decoder architecture

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

    # Forward pass is akin to a typical MLP, at the end the output is clipped to be roughly 0 or 1
    def call(self, points, z_vector, is_training=False):
        z_repeated = tf.reshape(z_vector, [-1, 1, self.z_dim])
        z_repeated = tf.repeat(z_repeated, repeats=points.shape[1], axis=1) # repeat z so that it can be applied to each of the points in batch
        points_and_z = tf.concat([points, z_repeated], axis=2) # concat the z vector to the xyz points

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
        dec7 = tf.maximum(tf.minimum(dec7, dec7*0.01*0.99), dec7*0.01) # approximate clipping helps with convergence

        return dec7

# Autoencoder class for wrapping Encoder and Decoder in one call
class Autoencoder(tf.keras.Model):
    def __init__(self, enc_dim=32, dec_dim=128, z_dim=256, point_dim=3):
        super().__init__()
        self.enc_dim = enc_dim 
        self.dec_dim = dec_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.encoder = Encoder(self.enc_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim, self.point_dim, self.dec_dim)

    def call(self, vox3d, z_vector, point_coord, is_training=False):
        if is_training: # Training mode: return the point value given the voxel model and point
            z_vector = self.encoder(vox3d)
            reconstructed_point_value = self.decoder(point_coord, z_vector)

        else:
            if vox3d is not None: # Encoding mode: just return the latent vector associated with the voxel model
                z_vector = self.encoder(vox3d)
            if z_vector is not None and point_coord is not None: # Decoding mode: obtain the point value from a given latent vector
                reconstructed_point_value = self.decoder(point_coord, z_vector)
            else:
                reconstructed_point_value = None

        return z_vector, reconstructed_point_value

# Entire network class including training and evaluation functions
class IMAE(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.sample_vox_size = config.sample_vox_size # Resolution to sample voxels at, one of 16, 32, or 64
        self.point_batch_size = 16*16*16 # Batch sizes are constrained by heavy memory usage of voxel models
        self.shape_batch_size = 16
        self.input_size = 64 # Input voxel models are always 64x64x64
        self.epochs = config.epoch

        self.enc_dim = 32
        self.dec_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_name + '_test'
        self.data_dir = config.data_dir
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=0.999)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.autoencoder)
        self.manager = tf.train.CheckpointManager(self.ckpt, config.checkpoint_dir, max_to_keep=10)
        
        self.sampling_threshold = 0.5
        self.real_size = 256
        
    # MSE: we want decoder-generated point values to be as close to ground-truth as possible
    def loss(self, point_value, reconstructed_point_value):
        return tf.reduce_mean(tf.square(point_value - reconstructed_point_value))

    def train(self):
        batch_index_list = np.arange(len(self.data_voxels)) # Indices of data examples used to divy out batches
        batch_num = int(len(self.data_voxels)/self.shape_batch_size) # Number of batches per epoch
        point_batch_num = int(self.load_point_batch_size/self.point_batch_size) # Number of points to sample per batch
        start_time = time.time()

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        
        for epoch in range(self.epochs):
            np.random.shuffle(batch_index_list) # Always shuffle your batches!
            total_loss = 0
            total_num = 0

            for idx in range(batch_num):
                indices_of_batch = batch_index_list[idx*self.shape_batch_size: (idx+1)*self.shape_batch_size]
                batch_voxels = self.data_voxels[indices_of_batch].astype(np.float32)
                batch_voxels = tf.transpose(batch_voxels, perm=[0, 2, 3, 4, 1]) # Needed to line up data with model's expected input shape

                sample_idx = np.random.randint(point_batch_num) # For each batch of voxel models we sample a random batch of points
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

    # Call just the encoder network to get z vectors from training examples for use in GAN
    def get_z(self, config):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        hdf5_path = self.dataset_name+'_train_z.hdf5'
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num,self.z_dim], np.float32)

        for t in range(shape_num):
            batch_voxels = self.data_voxels[t: t+1].astype(np.float32)
            batch_voxels = tf.transpose(batch_voxels, perm=[0, 2, 3, 4, 1])
            out_z,_ = self.autoencoder(batch_voxels, None, None, is_training=False)
            hdf5_file["zs"][t: t+1, :] = out_z.numpy()

        hdf5_file.close()
        print("Created z vectors dataset")

    # Generate meshes corresponding to z-vectors generated by the encoder from training examples
    def test_mesh(self, config):
        st = config.start
        en = config.end
        sample_dir = config.sample_dir
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        
        for t in range(st, min(len(self.data_voxels), en)):
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = tf.transpose(batch_voxels, perm=[0, 2, 3, 4, 1])
            model_z,_ = self.autoencoder(batch_voxels, None, None, is_training=False)
            model_float = z2voxel(model_z, self.autoencoder)
            
            # Use the marching cubes algorithm to convert voxel model to polygonal mesh
            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
            write_ply_triangle(sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)
    
    # Given a batch of z-vectors, generate corresponding meshes
    def test_z(self, config, batch_z, dim):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        
        for t in range(batch_z.shape[0]):
            model_z = batch_z[t:t+1]
            model_float = z2voxel(model_z, self.autoencoder)
            
            # Use the marching cubes algorithm to convert voxel model to polygonal mesh
            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
            write_ply_triangle(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)