import sys
import numpy as np
from yann.network import network
from yann.utils.dataset import setup_dataset
import logging

to_create_dataset = False
#dataset created 22177

if (to_create_dataset):
    data_params = {
        "source": 'matlab',
        #"name": 'svhn_yann',
        "location": '/home/praz/_datasets/to_convert',
        "height": 32,
        "width": 32,
        "channels": 3,
        "batches2test": 42,
        "batches2train": 56,
        "batches2validate": 28,
        "mini_batch_size": 500
    }
    preprocess_params = {
        "normalize": True,
        "ZCA": False,
        "grayscale": False,
        "zero_mean": False
    }
    save_directory = '/home/praz/_datasets'

    dataset = setup_dataset(dataset_init_args=data_params,
                            save_directory=save_directory,
                            preprocess_init_args=preprocess_params,
                            verbose=3)
# learning rates is a tuple, the first indicates an annealing of a linear rate
# the second is the initial learning reate of the first era, and the third
# is the learning rate of the second era


else:

    #learning_rates = (0.001, 0.0002)
    # creating the dataset

    net = network()

    # dataset is hardcoded to one that was created
    dataset_params = {"dataset": "/home/praz/_datasets/_dataset_22177",
                      "id": "svhn_yann",
                      "n_classes": 10}


    optimizer_params = {
        "momentum_type": 'polyak',
        "momentum_params": (0.9, 0.95, 30),
        #"regularization": (0.0001, 0.0002),
        "optimization_type": 'rmsprop',
        "id": 'polyak-rms'
    }

    net.add_module(type='optimizer', params=optimizer_params)

    net.add_layer(type="input", id="input", dataset_init_args=dataset_params)


    #add a convolutional layer #1: Applies 32 5*5 filters (extracs 5*5 regions with ReLu Activation)
    net.add_layer ( type = "conv_pool",
                origin = "input",
                id = "conv_pool_1",
                num_neurons = 10,
                filter_size = (5,5),
                pool_size = (2,2),
                activation = 'relu',
                stride = (2,2),
                verbose = 'verbose'
            )


    # region Description
    #Convolutional and pooling layer #2
    net.add_layer(type = 'conv_pool',
                  origin = 'conv_pool_1',
                  id = "conv_pool_2",
                  num_neurons = 5,
                  filter_size = (5,5),
                  activation = 'relu',
                  pool_size = (2,2),
                  #stride = (2,2),
                  verbose = 'verbose')
    # endregion



    net.add_layer(type="fully_connected", id='fully_connected', origin= 'conv_pool_2', verbose = 'verbose', activation = 'relu')

    #net.add_layer(type="fully_connected", id='fully_connected_2', origin= 'fully_connected', verbose = 'verbose', activation = 'relu')


    net.add_layer(type="classifier",
                  id="softmax",
                  origin="fully_connected",
                  num_classes=10,
                  activation="softmax"
                  )

    net.add_layer(type="objective",
                  id="nll",
                  origin="softmax")



    net.cook(optimizer='polyak-rms',
             objective_layer='nll',
             datastream='svhn_yann',
             classifier='softmax',
             )

    net.train(epochs=(20, 20),
              validate_after_epochs=1,
              training_accuracy=True,
              #learning_rates=learning_rates,
              show_progress=True,
              early_terminate=True
              )

    net.test()
