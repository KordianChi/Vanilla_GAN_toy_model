'''
This is first toymodel Generative Adversial Neural Network. Main goal of this
approach is reconstructed simply 2d shape (eg. circle) by generator and
discriminator pair.
In this file there are all functions
'''
### --- FUNCTIONS --- ###
### --- DATA PREPARATION --- ###

'''
This is functions for generate artificial, "true" data for our model - this is
2-D shape of thin circle, disc. First function is for control that random point
is on disc, second create "true" data set
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential


### --- GENERATE REAL DATA --- ###
    
def create_real_data(size):
    
    X = np.reshape(
                    uniform.rvs(loc=-1, scale=2, size=size),
                    (size, 1))
    Y = np.reshape(X ** 2, (size, 1))
    
    data = np.hstack((X, Y))


    return data

'''
This function create fake data set for training out discriminator network
'''

### --- GENERATE FAKE DATA --- ###

def create_fake_data(size):
    
    x = np.reshape(
        uniform.rvs(loc=-1, scale=2, size=size),
        (size, 1)
        )
    
    y = np.reshape(
        uniform.rvs(loc=0, scale=1, size=size),
        (size, 1)
        )
    
    data = np.hstack((x, y))
    
    return data

###-----------------------------###
### --- LATENT SPACE POINTS --- ###

'''
This function generate points from latent space for generator. Latent space
dimension is chosen arbitrally - reasonable is dim about 5.
'''

def latent_space_points(space_dim, size):
    
    points = uniform.rvs(loc=-1, scale=2, size=size * space_dim)
    points = np.reshape(points, (size, space_dim))
    
    return points
    

###---------------------###
### --- GAN NETWORK --- ###

''' 
This function is discriminator part of GAN network, for our simply case dense
ANN with 1 hidden layer is sufficient. We use functional api for good manners.
'''

### --- DISCRIMINATOR NETWORK --- ###

def discriminator(input_shape=2):
    
    inputs = Input(shape=input_shape)
    x = Dense(25)(inputs)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)

### ------------------------- ###
### --- GENERATOR NETWORK --- ###

def generator(latent_dim, n_outputs=2):
    
    inputs = Input(shape=latent_dim)
    x = Dense(15, kernel_initializer='he_uniform')(inputs)
    x = Activation('relu')(x)
    x = Dense(n_outputs)(x)
    outputs = Activation('linear')(x)
    
    return Model(inputs=inputs, outputs=outputs)

###---------------------------###
### --- GENERATOR WORKING --- ###

# use the generator to generate n fake examples and plot the results
def visualise_generator_points(generator, latent_dim, size):
    inputs = latent_space_points(latent_dim, size)
    X = generator.predict(inputs)
    '''
    x_min = X[:, 0].min().round() - 1
    x_max = X[:, 0].max().round() + 1
    y_min = X[:, 1].min().round() - 1
    y_max = X[:, 1].max().round() + 1
    '''
    
    x = np.linspace(-1, 1, 100)
    y = x ** 2
    
    plt.scatter(X[:, 0], X[:, 1], s=0.5, c='red')
    plt.plot(x,y)
    plt.axis([-1.5, 1.5, -0.5, 2])
    plt.show()
 
def generator_fake_points(generator, latent_dim, size):
    inputs = latent_space_points(latent_dim, size)
    X = generator.predict(inputs)
    return X
###---------------------------###

### --- GENERATIVE ADVERSIAL NETWORK --- ###

def generative_adversial_network(generator, discriminator):
    
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    return gan

