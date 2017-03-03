import sys
import numpy as np
from yann.network import network
import logging
#sys.stdout = open('logger', 'w')


sys.stdout = open('file2.log', 'w')
print("Arguments received are {}".format(sys.argv))


# receive the string to read the dataset from

# default dataset


if len(sys.argv) == 1:
    print (["No command line arguments give, using default:_dataset/_dataset_30385"])
    dataset_location = "/home/praz/_datasets/_dataset_30385"
else:
    dataset_location = sys.argv[1]

dataset_params = {"dataset": dataset_location, "id": 'mnist', "n_classes": 10}


net = network()
net.add_layer(type="input", id="input", dataset_init_args=dataset_params)

# adding fully connected hidden layers

net.add_layer(type="dot_product",
              origin="input",
              id="dot_product_1",
              num_nuerons=800,
              regularize=True,
              activation='relu'
              )

net.add_layer(type="dot_product",
              origin="dot_product_1",
              id="dot_product_2",
              num_neruons=800,
              regularize=True,
              activation='relu'
              )

net.add_layer(type="classifier",
              id="softmax",
              origin="dot_product_2",
              num_classes=10,
              activation='softmax'
              )

net.add_layer(type="objective",
              id="nll",
              origin="softmax"
              )
print("Network architecture: {}".format(net.layers))

# todo place the possible parameter options for the optimization modules here

momentum_types = [ 'polyak']
optimizer_types = ['adagrad' ]

log_filename = "for_submission_2.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG)
submission_logger = logging.getLogger("Submissionlogs")
submission_logger.setLevel(logging.DEBUG)



for momentum_type in momentum_types:
    for optimizer_type in optimizer_types:
        optimizer_params = {
                "momentum_type": momentum_type,
                "momentum_params": (0.9, 0.95, 30),
                "regularization": (0.0001, 0.0002),
                "optimizer_type": optimizer_type,
                "id": 'polyak-rms'
            }
        print(optimizer_params)
        net.add_module (type = 'optimizer', params = optimizer_params)
        net.cook(optimizer = 'polyak-rms',
         objective_layer= 'nll',
         datastream = 'mnist',
         classfier = 'softmax')
        net.train(epocs= (20, 20),
          validate_after_epochs= 2,
          training_accuracy= True,
          learning_rates= (0.001, 0.001, 0.001),
          show_progress = True,
          early_terminate= True)
        net.test()
