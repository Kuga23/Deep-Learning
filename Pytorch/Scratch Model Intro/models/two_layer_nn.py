# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        N=len(X)

            # 2 fully connected network
        z1=np.dot(X,self.weights['W1'])+self.weights['b1']
        h1=self.sigmoid(z1)
        z2= np.dot(h1,self.weights['W2'])+self.weights['b2']
        out=self.softmax(z2)

        loss=self.cross_entropy_loss(out,y)
        accuracy=self.compute_accuracy(out,y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        if mode != 'train':
            return loss, accuracy
        
            # dL/dz2 the correspind label activation -1
        grad_L_z2=out;
        for i in range (N):
            grad_L_z2[i,y[i]]=grad_L_z2[i,y[i]]-1
        grad_z2_w2= h1
        grad_L_w2= np.dot(grad_z2_w2.T,grad_L_z2)/N
        grad_L_b2= np.dot(np.ones(N).T,grad_L_z2)/N     # db/dbj flatten vector of ones
        self.gradients['W2']=grad_L_w2
        self.gradients['b2']=grad_L_b2

            #upstream layer dL/dz2 * dz2/dh1
        grad_L_h1=np.dot(grad_L_z2,self.weights['W2'].T)
        grad_h1_z1= self.sigmoid_dev(z1)
        grad_z1_w1=X
        grad_L_w1=np.dot(grad_z1_w1.T,grad_L_h1*grad_h1_z1)/N
        grad_L_b1=np.dot(np.ones(N).T,grad_L_h1*grad_h1_z1)/N
        self.gradients['W1']=grad_L_w1
        self.gradients['b1']=grad_L_b1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


