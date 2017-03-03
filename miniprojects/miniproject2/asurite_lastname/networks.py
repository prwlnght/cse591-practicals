#sample_submission.py
import numpy as np



class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """



    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        self.w0 = np.random.rand(self.x.shape[1],100)
        self.b0 = np.random.rand(1)
        self.w1 = np.random.rand(100,1)
        self.b1 = np.random.rand(1)
        self.params = [self.w0,self.b0, self.w1, self.b1]  # [(w,b),(w,b)]
        self.learning_rate = 0.01
        self.epochs = 50
        self.sum_error = 0

    def threshold (self, value):
        return np.asarray(value > self.b0, dtype ='int').T[0]

    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params




    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        #v_threshold = np.vectorize(threshold)
        return self.threshold(np.dot((np.dot(x, self.w0)  + self.b0), self.w1) +self.b1)

class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """

    def __init__ (self, data, labels):
        super(mlnn,self).__init__(data, labels)





if __name__ == '__main__':
    pass 
