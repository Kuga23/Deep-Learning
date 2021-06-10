import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.embedding= torch.nn.Embedding(self.output_size,self.emb_size)
        self.rec_layer= torch.nn.RNN(self.emb_size,self.decoder_hidden_size,1,batch_first=True)
        if self.model_type=="LSTM":
            self.rec_layer= torch.nn.LSTM(self.emb_size,self.decoder_hidden_size,1,batch_first=True)
        self.linear= torch.nn.Linear(self.decoder_hidden_size,self.output_size)
        self.softmax_layer= nn.LogSoftmax(dim=2)
        self.dropout_lay = torch.nn.Dropout(p=dropout);

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, sea_len); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
            # [seq,batchsize] to feed in embedding
        #b,s=input.shape
        #inp = input.reshape(s,b)
        #output, hidden = None, None
        emb= self.dropout_lay(self.embedding(input))
        if self.model_type == "RNN":
            output, hidden = self.rec_layer(emb, hidden)
        if self.model_type == "LSTM":
            output,hidden = self.rec_layer(emb,hidden)
        output=self.softmax_layer(self.linear(output)).squeeze(1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
            
