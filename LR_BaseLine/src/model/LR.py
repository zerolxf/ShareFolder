from numpy import *
import numpy as np
from model.model import *
from scipy.special import expit
import scipy.misc
class LR(Model):
    def __init__(self, data_x, data_y, alpha, mode="debug"):
        Model.__init__(self, data_x, data_y, alpha)
        self.n = self.data_x.shape[1]
        self.mode = mode

    def softmax(self, x):
        return expit(x)
    
    def get_indiv_gradient(self, w, idx):
        sf = expit(w.T * self.data_x[:, idx])
        return self.data_x[:, idx] * (sf - self.data_y[:, idx]).T + self.alpha*w
    
    def _debug_get_gradient(self, w, index):
        batch_size = index.shape[0]
        sf = expit(w.T * self.data_x[:, index])
        grad = self.data_x[:, index] * (sf - self.data_y[:, index]).T / batch_size
        grad = grad + self.alpha*w
        return grad
    
    def _release_get_gradient(self, w, index):
        batch_size = index.shape[0]
        grad = mat(zeros(w.shape))
        for i in index:
            grad = grad + self.get_indiv_gradient(w, i)
        grad = grad / batch_size
        return grad
        
    def get_gradient(self, w, index):
        if self.mode == "release":
            return self._release_get_gradient(w, index)
        elif self.mode == "debug":
            return self._debug_get_gradient(w, index)

    def get_full_gradient(self, w):
        index = arange(self.n)
        return self.get_gradient(w, index)
        
    def _get_indiv_hessian_outer(self, w, idx):
        sf =  expit(w.T * self.data_x[:, idx])
        hessian_value = sqrt(sf * (1 - sf))
        return  self.data_x[:,idx]*hessian_value

    def get_indiv_hessian(self, w, idx):
        d = w.shape[0]
        sf =  np.asscalar(expit(w.T * self.data_x[:, idx]))
        hessian_value = sf * (1 - sf)
        print(hessian_value)
        return  np.outer(self.data_x[:,idx],self.data_x[:,idx])*hessian_value + self.alpha * np.identity(d)
#         return  self.data_x[:,idx]*self.data_x[:,idx].T*hessian_value + self.alpha * np.identity(d)
        
    def _debug_get_hessian(self, w, index):
        batch_size = index.shape[0]
        tmp = expit(w.T * self.data_x[:,index])
        tmp = sqrt(np.multiply(tmp, 1-tmp))
        tmp = multiply(tmp, self.data_x[:,index])
        hessian = (tmp*tmp.T) / batch_size
        hessian = hessian + self.alpha * np.mat(np.eye(w.shape[0]))
        return hessian
    
    def _release_get_hessian(self, w, index):
        batch_size = index.shape[0]
        d = w.shape[0]
        batch_hess = np.zeros((d,d))
        for i in index:
            tmp = self._get_indiv_hessian_outer(w, i)
            batch_hess += tmp*tmp.T
        avg_batch_hess = batch_hess / batch_size
        return avg_batch_hess + self.alpha * np.identity(d)
        
    def get_hessian(self, w, index, batch_size=0):
        if self.mode == "release":
            return self._release_get_hessian(w, index)
        elif self.mode == "debug":
            return self._debug_get_hessian(w, index)
    
    def get_full_hessian(self, w):
        index = arange(self.n)
        return self.get_hessian(w, index)
    
    def get_loss(self, w):
        n = self.data_x.shape[1]
        wx = w.T * self.data_x
        loss = -wx*self.data_y.T
        zero = np.zeros((1, n))
        tmp = np.concatenate((wx, zero), axis=0)
        loss = np.asscalar((scipy.special.logsumexp(tmp, axis=0).sum()+ loss) / n)
        penalty = self.alpha * np.asscalar(w.T * w)
        loss = loss + penalty / 2.0
        return loss
    
    def get_indiv_hessian_vector(self, w, idx, u):
        sf = expit(w.T * self.data_x[:, idx])
        print(sf*(1-sf))
        dx = sf*(1-sf)*self.data_x[:, idx].T
        return self.data_x[:, idx]*(dx*u) + self.alpha*u
    
    
    def _debug_get_hessian_vector(self, w, index, u):
        batch_size = index.shape[0]
        sf = expit(w.T * self.data_x[:, index])
        sf = multiply(sf, 1-sf)
        sf_x = multiply(sf, self.data_x[:, index])
        res = self.data_x[:, index]*(sf_x.T*u)/batch_size
        res = res + self.alpha*u
        return res
    
    def _release_get_hessian_vector(self, w, index, u):
        res = mat(zeros(w.shape))
        for i in range(index):
            res += get_indiv_hessian_vector(w, i, u);
        res = res / len(index)
        return res
        
    def get_hessian_vector(self, w, index, u):
        if self.mode == "release":
            return self._release_get_hessian_vector(w, index, u)
        elif self.mode == "debug":
            return self._debug_get_hessian_vector(w, index, u)

            
