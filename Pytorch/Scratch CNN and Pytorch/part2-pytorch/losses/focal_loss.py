import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights=[]
    for n_i in cls_num_list:
        E_i= (1-beta**n_i)/(1-beta)
        alpha_i=1/E_i
        per_cls_weights.append(alpha_i)
        #normalization
    per_cls_weights=len(cls_num_list)*(per_cls_weights/np.sum(per_cls_weights))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
            #Class balanced CB focal loss
             
        softmax=nn.Softmax(dim=1)
        predict=softmax(input)
                #CB focal loss
        loss=0
        for i in range(len(target)):
            # focal loss for softmax
            loss+=-self.weight[target[i]]*torch.pow((1-predict[i,target[i]]),self.gamma)*torch.log(predict[i,target[i]])

        loss=loss/len(target)
        #loss=torch.tensor(loss/len(target))
        #print(loss)
        #loss=np.sum(w_i*((1-p_i)**self.gamma)*np.log(p_i))/len(target)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss