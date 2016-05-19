import theano.tensor.nnet
import theano.tensor as T
import numpy as np
from ..import init
from ..import nonlinearities
from ..utils import as_tuple
from ..theano_extensions import conv, padding

from .base import Layer
from .. import updates

__all__ = [
    "MultiGaussianMixtureClassification",
]




class MultiGaussianMixtureClassification(Layer):
    """ Build the multi-class gaussian mixture mode.
    
    .. math:: L = \sum_y t_y log(\sum_k \pi_{y, k} f(X, theta_{y, k}))
        where f(X, theta_{y, k}) is a multivariate gaussian distribution with 
        diagonal covariance matrix (For now.)

    Parameters
    -------------------------------
    X: the theano 2d tensor
        X is a theano 2d tensor passed from the last layer with dimension of (n, d), 
        where n is the batch size and d is the dimension of each data point. 
        We model a gaussian mixture model for X.
    targets: Theano 2D tensor or 1D tensor
        Either targets in [0,1] matching the layout of `predictions`, 
        or a vector of int giving the correct class index per data point.

    Returns
    -------------------------------
    Theano 1D tensor
    """
    def __init__(self, incoming, num_components, n_classes, _means = init.GlorotUniform(), weights = None, sigma = None, **kwargs):
        super(MultiGaussianMixtureClassification, self).__init__(incoming, **kwargs)

        self.dim = self.input_shape[1]
        self.num_components = num_components
        self.num_models = n_classes
        #_means = init.Constant(0)
        self._means = self.add_param(_means, (self.num_models, self.num_components, self.dim), name = "Means", regularizable = False, trainable = True)
        if weights is None:
            weights = init.Constant(1.0)

        self.weights = self.add_param(weights, (self.num_models, self.num_components,), name = "Weights", regularizable=False, trainable = True)
        
        if sigma is None:
            sigma = init.Constant(0.0)
        self.sigma = self.add_param(sigma, (self.num_models, self.num_components, self.dim), name = "Sigmas", regularizable = True, trainable = True)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)            

    def get_output_for(self,input, **kwargs):

        if input.ndim > 2:
            input = inpu.flatten(2)

        inputData = input * 10
        inputData.name = 'inputData'
        
        inputData_reshape = inputData.dimshuffle(0, 'x', 'x', 1)
        inputData_reshape.name = 'inputData_reshape'
        inputData_reshape = T.patternbroadcast(inputData_reshape, (False, True, True, False))
        #mean_reshape has dimension: (1, NumofClass, NumofComponent, p)
        mean_reshape = self._means.dimshuffle('x', 0, 1, 2)
        mean_reshape = T.patternbroadcast(mean_reshape, (True, False, False,False))
        mean_reshape.name = 'mean_reshape'

        #self.sigma = nonlinearities.rectify(self.sigma) + T.ones_like(self.sigma)
        sigma = T.exp(self.sigma)
        sigma_reshape = sigma.dimshuffle('x', 0, 1, 2)
        sigma_reshape = T.patternbroadcast(sigma_reshape, (True, False, False, False))
        sigma_reshape.name = 'sigma_reshape'

        #self.weights = nonlinearities.rectify(self.weights) + 1e-16
        weights = T.exp(self.weights)
        weights_sum = T.sum(weights, axis = 1)
        weights_sum = T.patternbroadcast(weights_sum.dimshuffle(0,'x'), (False, True))
        weights = weights / weights_sum
        
        weights_reshape = weights.dimshuffle('x', 0, 1)
        weights_reshape = T.patternbroadcast(weights_reshape, (True, False, False))
        weights_reshape.name = 'weights_reshape' 
        sigma_inverse_sqrt = T.sqrt(1.0/sigma_reshape)
        sigma_inverse_sqrt.name = 'sigma_inverse_sqrt'

        # positive: 
        sqrtTemp = T.sqr((inputData_reshape - mean_reshape) * sigma_inverse_sqrt).sum(axis = 3) 
        
        # negative: 784 * log(sigma) ? sigma = 0.1 -> -1805, else positive.
        sigmaTemp = T.log(sigma_reshape).sum(axis = 3)
        

        # positive:28x28 dimension, then we have 784 * log(2\pi) = 1440
        dimTemp = T.ones((self.num_models, self.num_components), 'float32') * self.dim * T.log(2.0 * np.pi)
        
        logComponentOutput = - 1.0 / 2 * (sqrtTemp + sigmaTemp + dimTemp)
        #logComponentOutput = -1.0/2 * sqrtTemp
        logComponentOutput.name = 'logComponentOutput'
        logComponentSum = logComponentOutput + T.log(weights_reshape) 
        logComponentSum.name = 'logComponentSum'
        logComponentSum_max = logComponentSum.max(axis = 2)
        logComponentSum_max_reshape = logComponentSum_max.dimshuffle(0, 1, 'x')
        componentSum_before = T.exp(logComponentSum - logComponentSum_max_reshape)
        componentSum_before_sum = componentSum_before.sum(axis = 2)
        addLog =  T.log(componentSum_before_sum + T.ones_like(componentSum_before_sum)) + logComponentSum_max
        #addLog = (componentSum_before + T.ones_like().sum(axis = 2)
        #return logComponentOutput, sqrtTemp, sigmaTemp, dimTemp, logComponentSum, logComponentSum_mean_reshape, componentSum_before, addLog, classSum
        addLog_max = addLog.max(axis = 1).dimshuffle(0, 'x')
        addLog_processed = addLog - addLog_max
        addLog_processed = T.exp(addLog_processed)
        softMaxPrediction = addLog_processed / (addLog_processed.sum(axis = 1).dimshuffle(0, 'x'))
        return addLog, softMaxPrediction
                













        
        
