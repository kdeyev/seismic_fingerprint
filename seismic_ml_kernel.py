def mn (X_train, y_train, X_test, y_test):
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Dense # the two types of neural network layer we will be using
	from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
	import numpy as np

	print ('image has', X_train.shape[1]*X_train.shape[2], 'pixels')
	
	batch_size = 128 # in each iteration, we consider 128 training examples at once
	num_epochs = 20 # we iterate twenty times over the entire training set
	hidden_size = 512 # there will be 512 neurons in both hidden layers

	num_train = X_train.shape[0] # there are 60000 training examples in MNIST
	num_test = X_test.shape[0] # there are 10000 test examples in MNIST

	height = X_test[0].shape[0]
	width = X_test[0].shape[1]
	
	#height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
	num_classes = len(np.unique(y_train)) # there are 10 classes (1 per digit)

	X_train = X_train.reshape(num_train, height * width) # Flatten data to 1D
	X_test = X_test.reshape(num_test, height * width) # Flatten data to 1D
	X_train = X_train.astype('float32') 
	X_test = X_test.astype('float32')
	X_train /= 255 # Normalise data to [0, 1] range
	X_test /= 255 # Normalise data to [0, 1] range

	Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
	Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

	inp = Input(shape=(height * width,)) # Our input is a 1D vector of size 784
	hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
	hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
	out = Dense(num_classes, activation='softmax')(hidden_2) # Output softmax layer

	model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy

	model.fit(X_train, Y_train, # Train the model using the training set...
			  batch_size=batch_size, nb_epoch=num_epochs,
			  verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
	print(model.evaluate(X_test, Y_test, verbose=1)) # Evaluate the trained model on the test set!
	
def ci (X_train, y_train, X_test, y_test, num_epochs = 200):
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
	from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
	import numpy as np

	print ('image has', X_train.shape[1]*X_train.shape[2], 'pixels')
	
	batch_size = 32 # in each iteration, we consider 32 training examples at once
	#num_epochs = 200 # we iterate 200 times over the entire training set
	kernel_size = 3 # we will use 3x3 kernels throughout
	pool_size = 2 # we will use 2x2 pooling throughout
	conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
	conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
	hidden_size = 512 # the FC layer will have 512 neurons

	num_train, depth, height, width = X_train.shape # there are 50000 training examples in CIFAR-10 
	num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
	num_classes = np.unique(y_train).shape[0] # there are 10 image classes
	
	X_train = X_train.astype('float32') 
	X_test = X_test.astype('float32')
	X_train /= np.max(X_train) # Normalise data to [0, 1] range
	X_test /= np.max(X_train) # Normalise data to [0, 1] range

	Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
	Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels


	inp = Input(shape=(depth, height, width)) # N.B. depth goes first in Keras!
	# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
	conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
	conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)
	pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
	drop_1 = Dropout(drop_prob_1)(pool_1)
	# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
	conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(drop_1)
	conv_4 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_3)
	pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
	drop_2 = Dropout(drop_prob_1)(pool_2)
	# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
	flat = Flatten()(drop_2)
	hidden = Dense(hidden_size, activation='relu')(flat)
	drop_3 = Dropout(drop_prob_2)(hidden)
	out = Dense(num_classes, activation='softmax')(drop_3)

	model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy

	model.fit(X_train, Y_train, # Train the model using the training set...
			  batch_size=batch_size, nb_epoch=num_epochs,
			  verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
	print(model.evaluate(X_test, Y_test, verbose=1)) # Evaluate the trained model on the test set!

def ci_vision_model (X_train):
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
	import keras
	
	kernel_size = 3 # we will use 3x3 kernels throughout
	pool_size = 2 # we will use 2x2 pooling throughout
	conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
	conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	
	num_train, depth, height, width = X_train.shape # there are 50000 training examples in CIFAR-10 
	
	print ('image has', X_train.shape[1]*X_train.shape[2], 'pixels')
	
	inp = Input(shape=(depth, height, width)) # N.B. depth goes first in Keras!
	# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
	conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
	conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)
	pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
	drop_1 = Dropout(drop_prob_1)(pool_1)
	# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
	conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(drop_1)
	conv_4 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_3)
	pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
	drop_2 = Dropout(drop_prob_1)(pool_2)
	# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
	flat = Flatten()(drop_2)

	vision_model = Model(input=inp, output=flat) # To define a model, just specify its input and output layers
	
	keras.utils.plot_model(vision_model, to_file='outputs/vision_model.png')

	vision_model_inp = inp
	vision_model_out = vision_model (inp)
	return vision_model_inp, vision_model_out
	
	
def ci_multi_train (X_train, y_train, num_epochs = 20):
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
	from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
	import numpy as np
	import keras
	
	import os
	if os.path.exists('outputs/classification_model.h5'):
		print ('Model already exists')
		classification_model = keras.models.load_model('outputs/classification_model.h5')
		return classification_model
	
	
	vision_model_inputs = []
	vision_model_outputs = []
	for i in range(len(X_train)):
		X_train[i] = X_train[i].astype('float32') 
		X_train[i] /= np.max(X_train[i]) # Normalise data to [0, 1] range
		vision_model_inp, vision_model_out = ci_vision_model (X_train[i])
		vision_model_inputs.append(vision_model_inp)
		vision_model_outputs.append(vision_model_out)
	
	num_classes = np.unique(y_train).shape[0] # there are 10 image classes
	Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
			
	print ('number of vision models', len (X_train))
	
	batch_size = 32 # in each iteration, we consider 32 training examples at once
	#num_epochs = 200 # we iterate 200 times over the entire training set
	hidden_size = 512 # the FC layer will have 512 neurons
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
			
	print ('Vision models are ready')
	
	concatenated_inp = keras.layers.concatenate(vision_model_outputs)	
	hidden = Dense(hidden_size, activation='relu')(concatenated_inp)
	drop_3 = Dropout(drop_prob_2)(hidden)
	out = Dense(num_classes, activation='softmax')(drop_3)

	classification_model = Model(vision_model_inputs, out)

	print ('Classification model is ready')
	
	classification_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy

	print ('Classification model compiled')
	
	keras.utils.plot_model(classification_model, to_file='outputs/classification_model.png')

	tbCallBack = keras.callbacks.TensorBoard(log_dir='outputs/Graph', histogram_freq=0, write_graph=True, write_images=True)
	esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

	classification_model.fit(X_train, Y_train, # Train the model using the training set...
			  batch_size=batch_size, epochs=num_epochs,
			  verbose=1, validation_split=0.1, # ...holding out 10% of the data for validation
			  callbacks=[tbCallBack, esCallBack])
			  
	classification_model.save('outputs/classification_model.h5')  # creates a HDF5 file 'my_model.h5'

def ci_multi_test (classification_model, X_test, y_test):
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
	from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
	import numpy as np
	import keras
	
	for i in range(len(X_test)):
		X_test[i] = X_test[i].astype('float32')
		X_test[i]  /= np.max(X_test[i]) # Normalise data to [0, 1] range
	
	
	num_classes = np.unique(y_test).shape[0] # there are 10 image classes
	Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
	num_test = X_test[0].shape[0] # there are 10000 test examples in CIFAR-10
	
	batch_size = 32 # in each iteration, we consider 32 training examples at once

	print(classification_model.evaluate(X_test, Y_test, verbose=2)) # Evaluate the trained model on the test set!
	
def ci_speed (X_train, y_train, X_test, y_test):
	from keras.datasets import mnist # subroutines for fetching the MNIST dataset
	from keras.models import Model # basic class for specifying and training a neural network
	from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, merge
	from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
	from keras.regularizers import l2 # L2-regularisation
	from keras.layers.normalization import BatchNormalization # batch normalisation
	from keras.preprocessing.image import ImageDataGenerator # data augmentation
	from keras.callbacks import EarlyStopping # early stopping
	import numpy as np
	
	print ('image has', X_train.shape[1]*X_train.shape[2], 'pixels')
		
	batch_size = 128 # in each iteration, we consider 128 training examples at once
	num_epochs = 50 # we iterate at most fifty times over the entire training set
	kernel_size = 3 # we will use 3x3 kernels throughout
	pool_size = 2 # we will use 2x2 pooling throughout
	conv_depth = 32 # use 32 kernels in both convolutional layers
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
	hidden_size = 128 # there will be 128 neurons in both hidden layers
	l2_lambda = 0.0001 # use 0.0001 as a L2-regularisation factor
	ens_models = 3 # we will train three separate models on the data

	num_train = X_train.shape[0] # there are 60000 training examples in MNIST
	num_test = X_test.shape[0] # there are 10000 test examples in MNIST

	height, width, depth = X_train.shape[1], X_train.shape[2], 3 # MNIST images are 28x28 and greyscale
	num_classes = np.unique(y_train).shape[0]  # there are 10 classes (1 per digit)

	X_train = X_train.reshape(X_train.shape[0], depth, height, width)
	X_test = X_test.reshape(X_test.shape[0], depth, height, width)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
	Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

	# Explicitly split the training and validation sets
	ss = int (num_train*0.8)
	X_val = X_train[ss:]
	Y_val = Y_train[ss:]
	X_train = X_train[:ss]
	Y_train = Y_train[:ss]

	inp = Input(shape=(depth, height, width)) # N.B. Keras expects channel dimension first
	inp_norm = BatchNormalization(axis=1)(inp) # Apply BN to the input (N.B. need to rename here)

	outs = [] # the list of ensemble outputs
	for i in range(ens_models):
		# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer), applying BN in between
		conv_1 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode='same', init='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(inp_norm)
		conv_1 = BatchNormalization(axis=1)(conv_1)
		conv_2 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode='same', init='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(conv_1)
		conv_2 = BatchNormalization(axis=1)(conv_2)
		pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
		drop_1 = Dropout(drop_prob_1)(pool_1)
		flat = Flatten()(drop_1)
		hidden = Dense(hidden_size, init='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(flat) # Hidden ReLU layer
		hidden = BatchNormalization(axis=1)(hidden)
		drop = Dropout(drop_prob_2)(hidden)
		outs.append(Dense(num_classes, init='glorot_uniform', W_regularizer=l2(l2_lambda), activation='softmax')(drop)) # Output softmax layer

	out = merge(outs, mode='ave') # average the predictions to obtain the final output

	model = Model(input=inp, output=out) # To define a model, just specify its input and output layers

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy

	datagen = ImageDataGenerator(
			width_shift_range=0.1,	# randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1)	 # randomly shift images vertically (fraction of total height)
	datagen.fit(X_train)

	# fit the model on the batches generated by datagen.flow()---most parameters similar to model.fit
	model.fit_generator(datagen.flow(X_train, Y_train,
							batch_size=batch_size),
							samples_per_epoch=X_train.shape[0],
							nb_epoch=num_epochs,
							validation_data=(X_val, Y_val),
							verbose=1,
							callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]) # adding early stopping

	print(model.evaluate(X_test, Y_test, verbose=1)) # Evaluate the trained model on the test set!