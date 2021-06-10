import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
            # by default stide=1 and paddidng=0 but whatever
        self.Conv2d=nn.Conv2d(3,32,(7,7),1,0)
        self.MaxPool2d=nn.MaxPool2d((2,2),2)
            # Input image 3*32*32, throught conv2d and Maxpool we got this
            #input dim
        self.linear=nn.Linear(32*13*13,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        conv1_kernels=self.Conv2d(x)
        a1=torch.relu(conv1_kernels)
        h1=self.MaxPool2d(a1)
        outs=self.linear(torch.flatten(h1,start_dim=1))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs