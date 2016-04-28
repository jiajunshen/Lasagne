import theano.tensor.nnet
import theano.tensor as T
from ..import init
from ..import nonlinearities
from ..utils import as_tuple
from ..theano_extensions import conv, padding

from .base import Layer


__all__ = [
    "MultiGaussianMixture",
]




class MultiGaussianMixture(Layer):
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
        super(MultiGaussianMixture, self).__init__(incoming, **kwargs)

        self.dim = self.input_shape[1] - n_classes
        self.num_components = num_components
        self.num_models = n_classes
        self._means = self.add_param(_means, (self.num_models, self.num_components, self.dim), name = "Means", regularizable = False)
        if weights is None:
            weights = init.Constant(1.0/num_components)
            self.weights = self.add_param(weights, (self.num_models, self.num_components,), name = "Weights", regularizable=False)
        if sigma is None:
            sigma = init.Constant(1.0)
            self.sigma = self.add_param(sigma, (self.num_models, self.num_components, self.dim), name = "Sigmas", regularizable = False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)            

    def get_output_for(self,input, **kwargs):

        if input.ndim > 2:
            input = inpu.flatten(2)

        inputData = input[:,:-self.num_models]
        inputLabel = input[:,-self.num_models:]
        #inputData_reshape has dimension: (n, 1, 1, p)
        inputData_reshape = inputData.dimshuffle(0, 'x', 'x', 1)
        inputData_reshape = T.patternbroadcast(inputData_reshape, (False, True, True, False))
        #mean_reshape has dimension: (1, NumofClass, NumofComponent, p)
        mean_reshape = self._means.dimshuffle('x', 0, 1, 2)
        mean_reshape = T.patternbroadcast(mean_reshape, (True, False, False,False))
        sigma_reshape = self.sigma.dimshuffle('x', 0, 1, 2)
        sigma_reshape = T.patternbroadcast(sigma_reshape, (True, False, False, False))
        weights_reshape = self.weights.dimshuffle('x', 0, 1)
        weights_reshape = T.patternbroadcast(weights_reshape, (True, False, False))
        
        
        sigma_inverse_sqrt = T.sqrt(1.0/sigma_reshape)

        sigma_determinant = T.prod(sigma_reshape, axis = 3)
        
        logComponentSum = -T.sqr((inputData_reshape - mean_reshape) * sigma_inverse_sqrt).sum(axis = 3) - T.log(sigma_determinant) + T.log(weights_reshape)
         
        componentSum = T.exp(logComponentSum)
        componentSum = componentSum.clip(a_min = 1e-32, a_max = 1e32) 
        classSum = -(T.log(componentSum.sum(axis = 2)) * inputLabel).sum(axis = 1)
        
        return classSum
                















        
        
