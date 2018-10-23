from numpy import *
from utils.Logger import *


def vec_transpose(x, p):
    m, n = x.shape
    return x.reshape(int(m / p), p, n).T.reshape(int(n * p), int(m / p))


class Model(object):
    def __init__(self, data_x, data_y, alpha):
        self.data_x = data_x
        self.data_y = data_y
        self.alpha = alpha

    def get_indiv_gradient(self, w, index):
        Logger.log("Model get_indiv_gradient not implement")
        raise NotImplementedError
        
    def get_gradient(self, w, index):
        Logger.log("Model get_gradient not implement")
        raise NotImplementedError

    def get_full_gradient(self, w):
        Logger.log("Model get_full_gradient not implement")
        raise NotImplementedError
        
    def get_indiv_hessian(self, w, index):
        Logger.log("Model get_indiv_hessian not implement")
        raise NotImplementedError
        
    def get_hessian(self, w, index):
        Logger.log("Model get_hessian not implement")
        raise NotImplementedError

    def get_indiv_hessian_vector(self, w, index, u):
        Logger.log("Model get_hessian_vector_product not implement")
        raise NotImplementedError
        
    def get_hessian_vector(self, w, index, u):
        Logger.log("Model get_hessian_vector_product not implement")
        raise NotImplementedError

    def get_loss(self, w):
        Logger.log("Model get_loss not implement")
        raise NotImplementedError

