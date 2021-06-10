import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.embedding= torch.nn.Embedding(self.input_size,self.emb_size)
        self.rec_layer= torch.nn.RNN(self.emb_size,self.encoder_hidden_size,1,batch_first=True)
        if self.model_type=="LSTM":
            self.rec_layer= torch.nn.LSTM(self.emb_size,self.encoder_hidden_size,1,batch_first=True)
        self.linear1=torch.nn.Linear(self.encoder_hidden_size,self.encoder_hidden_size)
        self.relu_lay = torch.nn.ReLU()
        self.linear2=torch.nn.Linear(self.encoder_hidden_size,self.decoder_hidden_size)
        self.dropout_lay = torch.nn.Dropout(p=dropout);

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len, input_size)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        #b,s= input.shape
        #inp = input.reshape(s,b)
        output, hidden = None, None
        emb= self.dropout_lay(self.embedding(input))
        output,hidden= self.rec_layer(emb)

        if self.model_type=="RNN":
            hidden=self.linear2(self.relu_lay(self.linear1(hidden)))
            hidden= torch.tanh(hidden)
        else:
            newh=self.linear2(self.relu_lay(self.linear1(hidden[0])))
            #newc=self.linear2(self.relu_lay(self.linear1(hidden[1])))
            newh= torch.tanh(newh)
            #newc= torch.tanh(newc)
            hidden=(newh,hidden[1])
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden