# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
            # numerically stable version
            # shift to negative so the maximum will be 0 not infinite
        c=np.max(scores)
        score_exp=np.exp(scores- c)
        sum_score_exp=np.sum(score_exp,axis=1)
            # each element of row divided by sum of exp
            # we get a matrix of size N,num_classes
        prob= np.zeros(scores.shape)
        for i in range(scores.shape[0]):
            prob[i,:]=score_exp[i,:]/sum_score_exp[i]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        
            # already applied sofmax, normalized score
        N,num_classes= x_pred.shape
            # get p_i for each data
        L_i=[x_pred[i,y[i]] for i in range (N)]
        loss= np.sum(-np.log(L_i))/N
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        
        N,num_classes=x_pred.shape
            # get p_i for each data
        label_value=[x_pred[i,y[i]] for i in range (N)]
        nb_correct=0;
        for i in range (N):
                # argmax
            if np.max(x_pred[i,:])==label_value[i]:
                nb_correct+=1
                # accuracy between [0:1]
        acc= nb_correct/N
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        out = None
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        out= 1/(1+ np.exp(-X))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        ds = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
            #sig(1-sig)
        ds= np.multiply(self.sigmoid(x),(1-self.sigmoid(x)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        out = None
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
            # max(0,X)
        zeroes= np.zeros(X.shape)
        out= np.maximum(zeroes,X)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        out = None
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        
        out=np.zeros(X.shape)
        N,num_classes= X.shape
            # 1 if input positive or 0        
        for i in range(N):
            for j in range (num_classes):
                if X[i,j]>0:
                    out[i,j]=1   # change if input >0

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
