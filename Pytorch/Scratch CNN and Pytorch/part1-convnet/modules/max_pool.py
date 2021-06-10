import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N,C,H,W= x.shape
            #square kernel or not
        if type(self.kernel_size) is int:
            k1=k2=self.kernel_size
        else:
            k1=self.kernel_size[0]
            k2=self.kernel_size[1]

        H_out=W_out=[]
        out=np.zeros((N,C,(H-k1)//self.stride+1,(W-k2)//self.stride+1))
        for n in range(N):
            for c in range (C):
                for h,i in enumerate(range (0,H-k1+1,self.stride)): 
                    for w,j in enumerate(range (0,W-k2+1,self.stride)):
                        out[n,c,h,w]=np.max(x[n,c,i:i+k1,j:j+k2])
                        H_out=out[n,c,h]
                        W_out=out[n,c,:,w]
                        
                 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N,C,H,W= x.shape
        if type(self.kernel_size) is int:
            k1=k2=self.kernel_size
        else:
            k1=self.kernel_size[0]
            k2=self.kernel_size[1]
        dx= np.zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for h,i in enumerate(range (0,H-k1+1,self.stride)):
                    for w,j in enumerate(range (0,W-k2+1,self.stride)):
                        max_h,max_w= np.unravel_index(np.argmax(x[n,c,i:i+k1,j:j+k2]),(k1,k2))
                        dx[n,c,i+max_h,j+max_w]=dout[n,c,h,w]
        self.dx=dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
