import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DataLoader:

	def __init__(self, simulator, verbose=False):

		self.verbose = verbose

		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []
		self.y_reg_train = []
		self.y_reg_test = []

		self.timesteps = 0
		self.sim = simulator

		self.maxs = []
    
	def load_data(self):

		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

		#normalize the images
		x_train = np.expand_dims(x_train/255.0, axis=-1)
		x_test = np.expand_dims(x_test/255.0, axis=-1)

		#discretize the images
		x_train = np.where(x_train<0.5, 0, 1)
		x_test = np.where(x_test<0.5, 0, 1)

		#create (simulate) a synthetic "time series" data vector (y) for each of the input (x) such that y=Gx and G is linear
		#self.sim  represents some abstract function (i.e. fluid flow simulator)

		y_dim = self.sim.shape[-1]
		y_reg_train = np.zeros([y_train.shape[0], y_dim])
		y_reg_test = np.zeros([y_test.shape[0], y_dim])

		#simulate Y = GX
		for i in range(y_train.shape[0]):
			y_reg_train[i:i+1, :] = np.reshape((x_train[i:i+1, :, :, 0]), [1, x_train.shape[1]*x_train.shape[2]])@self.sim 

		for i in range(y_test.shape[0]):
			y_reg_test[i:i+1, :] = np.reshape((x_test[i:i+1, :, :, 0]), [1, x_test.shape[1]*x_test.shape[2]])@self.sim 
		    
		#normalize data
		self.maxs = np.max(y_reg_train, axis=0)
		y_reg_train = y_reg_train/self.maxs
		y_reg_test = y_reg_test/self.maxs

		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
		self.y_reg_train = np.expand_dims(y_reg_train, axis=-1)
		self.y_reg_test = np.expand_dims(y_reg_test, axis=-1)

		if self.verbose: 
			print("Loaded training data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_train.shape), str(self.y_reg_train.shape), str(self.y_train.shape)))
			print("Loaded testing data x {:s} and y {:s} and y_labels {:s}".format(str(self.x_test.shape), str(self.y_reg_test.shape), str(self.y_test.shape)))
		    
		return self.x_train, self.x_test, self.y_train, self.y_test, self.y_reg_train, self.y_reg_test
        
	def simulator(self, ms):
		'''simulate observations for a given set of discretized models
		'''
		#discretize the images
		ms = np.where(ms<0.5, 0, 1)

		#ms : 10,000 x 784
		d_dim = self.sim.shape[-1]
		ds = np.zeros([ms.shape[0], d_dim])

		for i in range(ms.shape[0]):
			ds[i:i+1, :] = np.reshape((ms[i:i+1, :, :, 0]), [1, ms.shape[1]*ms.shape[2]])@self.sim 
		ds = ds/self.maxs

		return np.expand_dims(ds, axis=-1)
             
 