import numpy as np

class ReLU:
    '''
    An implementation of rectified linear units(ReLU)
    '''
    def __init__(self):
        self.cache = None
        self.dx= None

    def forward(self, x):
        '''
        The forward pass of ReLU. Save necessary variables for backward
        :param x: input data
        :return: output of the ReLU function
        '''
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        zeroes= np.zeros(x.shape)
        out=np.maximum(zeroes,x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''

        :param dout: the upstream gradients
        :return:
        '''
        dx, x = None, self.cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
            # dout and x have the same size

        grad_relu=x
        grad_relu[grad_relu > 0]=1
        grad_relu[grad_relu < 0]=0
        dx=dout*grad_relu
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.dx = dx