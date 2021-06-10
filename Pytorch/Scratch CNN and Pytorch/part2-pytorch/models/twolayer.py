import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        # Architecture of 2 fully connected layers
        self.linear1=nn.Linear(input_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        z1=self.linear1(torch.flatten(x.clone().detach(),start_dim=1))
        h1=torch.sigmoid(z1)
        out=self.linear2(h1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out