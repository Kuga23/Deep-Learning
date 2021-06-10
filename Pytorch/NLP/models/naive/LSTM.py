import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        # Tutorial Pytorch Doc

        #i_t: input gate
        """self.i_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.i_t.wih= torch.nn.Parameter(torch.zeros(self.input_size+self.hidden_size,self.hidden_size))
        self.i_t.bih= torch.nn.Parameter(torch.zeros(self.hidden_size))

        self.f_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.f_t.wih= torch.nn.Parameter(torch.zeros(self.input_size+self.hidden_size,self.hidden_size))
        self.f_t.bih= torch.nn.Parameter(torch.zeros(self.hidden_size))

        self.g_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.g_t.wih= torch.nn.Parameter(torch.zeros(self.input_size+self.hidden_size,self.hidden_size))
        self.g_t.bih= torch.nn.Parameter(torch.zeros(self.hidden_size))

        self.o_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.o_t.wih= torch.nn.Parameter(torch.zeros(self.input_size+self.hidden_size,self.hidden_size))
        self.o_t.bih= torch.nn.Parameter(torch.zeros(self.hidden_size))"""



        self.wii = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.whi = torch.nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bii = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.bhh = torch.nn.Parameter(torch.Tensor(self.hidden_size))


        # f_t: the forget gate
        #self.f_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.wif = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.whf = torch.nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bif = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.bhf = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        # g_t: the cell gate
        #self.g_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.wig = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.whg = torch.nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.big = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.bhg = torch.nn.Parameter(torch.Tensor(self.hidden_size))
            
        # o_t: the output gate
        #self.o_t= nn.Linear(self.input_size+self.hidden_size,self.hidden_size)
        self.wio = torch.nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.who = torch.nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bio = torch.nn.Parameter(torch.Tensor(self.hidden_size))
        self.bho = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        # Tutorial Pytorch inspiration
        b,s,f= x.shape
        h_t, c_t = None, None
        # init if not given by init_states
        if init_states==None:
            h_t= torch.zeros(b,self.hidden_size)
            c_t=torch.zeros(b,self.hidden_size)
        else:
            h_t,c_t= init_states
        """for t in range(s):
            x_t=x[:,t,:]
            i_t=torch.sigmoid(self.i_t(torch.cat((x_t,h_t),1)))
            f_t=torch.sigmoid(self.i_t(torch.cat((x_t,h_t),1)))
            g_t=torch.tanh(self.i_t(torch.cat((x_t,h_t),1)))
            o_t=torch.sigmoid(self.i_t(torch.cat((x_t,h_t),1)))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
        # loop on the sequence for each step time t"""
        for t in range(s):
            i_t = torch.sigmoid(x[:,t,:]@self.wii + h_t@self.whi + self.bii+self.bii)
            f_t = torch.sigmoid(x[:,t,:]@self.wif + h_t @self.whf + self.bif+self.bif)
            g_t = torch.tanh(x[:,t,:]@self.wig + h_t @self.whg + self.big+self.big)
            o_t = torch.sigmoid(x[:,t,:]@self.wio + h_t@self.who + self.bif+self.bif)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

