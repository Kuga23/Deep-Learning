import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
            # padding zeros with same size for each edge
        zero_pad=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        x_pad=np.pad(x, zero_pad, 'constant')
        N,C,H,W=x_pad.shape
            # If kernel size not equal for H and W
        if type(self.kernel_size) is int:
            k1=k2=self.kernel_size
        else:
            k1=self.kernel_size[0]
            k2=self.kernel_size[1]
            # output size considering kernel size stride and padding
        H_=(H-k1)//self.stride+1
        W_=(W-k2)//self.stride+1
        out=np.zeros((N,self.out_channels,H_,W_))

        for n in range(N):
            for c_out in range (self.out_channels):
                for h,i in enumerate(range (0,H-k1+1,self.stride)): 
                    for w,j in enumerate(range (0,W-k2+1,self.stride)):
                        out[n,c_out,h,w]=np.sum(self.weight[c_out,:,:,:]*x_pad[n,:,i:i+k1,j:j+k2])+self.bias[c_out]



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
            # padding zeros with same size for each edge
        zero_pad=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        x_pad=np.pad(x, zero_pad, 'constant')
        N,C,H,W=x_pad.shape
            # If kernel size not equal for H and W
        if type(self.kernel_size) is int:
            k1=k2=self.kernel_size
        else:
            k1=self.kernel_size[0]
            k2=self.kernel_size[1]


        dw=np.zeros((self.out_channels, self.in_channels, k1, k2))
        dx_xpad=np.zeros(x_pad.shape)
        db=np.zeros(self.out_channels)

            # Compute each gradient necessary to conv layer and upstream gradient
        for n in range(N):
            for c_out in range (self.out_channels):
                tmp_db=np.zeros((self.in_channels,k1,k2))
                for h,i in enumerate(range (0,H-k1+1,self.stride)): 
                    for w,j in enumerate(range (0,W-k2+1,self.stride)):
                         dw[c_out]+=x_pad[n,:,i:i+k1,j:j+k2]*dout[n,c_out,h,w]
                         db[c_out]+=dout[n,c_out,h,w]
                         dx_xpad[n,:,i:i+k1,j:j+k2]+=dout[n,c_out,h,w]*self.weight[c_out,:,:,:]
        self.dw=dw
        self.dx=dx_xpad[:,:,self.padding:H-self.padding,self.padding:W-self.padding]
        self.db=db

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
