#from yann.utils.pickle import pickle
#from yann.utils.pickle import load

from yann.network import network
from datasets import cook_mnist_bg_normalized
from yann.special.datasets import cook_mnist



#steps
#0 make the datasets: (datasets seem to be created in local directory to the code)
##  cook_mnist_rotated_normalized : Created as 61713
## Error on creating
#train the network on one dataset
#1 create another network from the parameters
#2 make sure the parameters are transfered correctly
#3 set it up with all the layers but the softmax frozen (learnable = false)
#4 train only the softmax on this layer
#5 Note the performance down


net1 = network()

#cooking all databases

#1 bg_normalized

#cook_mnist_noisy_normalized()
cook_mnist_bg_normalized()

#cook_mnist()
#cook_mnist_noisy_normalized()

#two convolutinal, two dense layer network

#1st layer: 20 neurons, 5X5 with a pooling of 2X2

#2nd is a 50 neurons, of 3X3 with a pooling of 2X2

#3rd dot product layer with 800 nodes each and droput_rate = 0.5 , L1 and L2 on all layers of 0.0001, RMSPROP with Nesterov momemtum

#4th dot product layer with 800 nodes each and droput_rate = 0.5 , L1 and L2 on all layers of 0.0001, RMSPROP with Nesterov momemtum


#to save a network's parameters down

#pickle(net1, network.pkl)



#to load the network parameters as a dictionary
#params = load('network.pkl')
