def mn (X_train, y_train, X_test, y_test):
    from keras.models import Model # basic class for specifying and training a neural network
    from keras.layers import Input, Dense # the two types of neural network layer we will be using
    from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
    import numpy as np

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