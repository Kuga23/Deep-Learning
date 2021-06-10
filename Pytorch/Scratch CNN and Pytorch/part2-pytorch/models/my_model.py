import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.Conv2d_1=nn.Conv2d(3,32,(5,5),1,2)
        self.MaxPool2d_1=nn.MaxPool2d((2,2),2)
        self.Conv2d_2=nn.Conv2d(32,64,(5,5),1,2)
        self.MaxPool2d_2=nn.MaxPool2d((2,2),2)
        self.Conv2d_3=nn.Conv2d(64,128,(5,5),1,2)
        self.MaxPool2d_3=nn.MaxPool2d((2,2),2)
        self.linear1=nn.Linear(128*4*4,512)
        self.linear2=nn.Linear(512,128)
        self.linear3=nn.Linear(128,10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        conv1=self.Conv2d_1(x)
        a1=torch.relu(conv1)
        h1=self.MaxPool2d_1(a1)
        
        conv2=self.Conv2d_2(h1)
        a2=torch.relu(conv2)
        h2=self.MaxPool2d_2(a2)
        
        conv3=self.Conv2d_3(h2)
        a3=torch.relu(conv3)
        h3=self.MaxPool2d_3(a3)
    
        z4=self.linear1(torch.flatten(h3,start_dim=1))
        h4=torch.relu(z4)
        z5=self.linear2(h4)
        h5=torch.relu(z5)
        outs=self.linear3(h5)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs