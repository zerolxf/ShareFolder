from numpy import *
from model.model import *
from numpy.linalg import *
import numpy as np
# from sklearn.utils.extmath import (log_logistic, safe_sparse_dot, 
#                              squared_norm)

class LRSoftmax(Model):
    def __init__(self, data_x, data_y, alpha):
        Model.__init__(self, data_x, data_y, alpha)

    def get_indiv_gradient(self, w, idx):
        tmp = w.T * self.data_x[:, idx]
        tmp = exp(tmp - tmp.max(axis=0))
        sf = tmp/tmp.sum(axis=0)
        return self.data_x[:, idx] * (sf - self.data_y[:, idx]).T+ self.alpha*w

    def get_full_gradient(self, w):
#         batch_size = self.data_x.shape[1]
#         index = arange(batch_size)
#         grad = mat(zeros(w.shape))
#         for i in index:
#             grad = grad + self.get_indiv_gradient(w, i)
#         grad = grad / batch_size 
#         return grad
        num = self.data_x.shape[1]
        sf = self.softmax(w.T * self.data_x)
        grad = self.data_x * (sf - self.data_y).T / num
        grad = grad + self.alpha*w
        return grad

    def softmax(self, x):
        ex = exp(x-x.max(axis=0))
        return ex/ex.sum(axis=0)

    def get_gradient(self, w, index):
#         num = len(index)
#         grad = mat(zeros(w.shape))
#         for i in index:
#             grad = grad + self.get_indiv_gradient(w, i)
#         grad = grad/batch_size 
#         return grad
        data_x = self.data_x[:,index]
        data_y = self.data_y[:,index]
        num = data_x.shape[1]
        sf = self.softmax(w.T * data_x)
        grad = data_x * (sf - data_y).T / num
        grad = grad + self.alpha*w
        return grad

    def get_hessian(self, w, index, batch_size):
        d, k = w.shape
        sf = self.softmax(w.T * self.data_x[:,index])
        hessian = mat(zeros((d * k, d * k)))
        for i in range(k):
            row_i = multiply(sf[i, :], self.data_x[:, index])
            for j in range(k):
                col_j = multiply(sf[j,:], self.data_x[:,index])
                hessian[i*d:(i+1)*d,j*d:(j+1)*d] = -row_i*col_j.T/batch_size
                if i == j:
                    col_j = multiply(sqrt(sf[i, :]), self.data_x[:, index])
                    hessian[i * d:(i + 1) * d, j * d:(j + 1) * d] = hessian[i * d:(i + 1) * d, j * d:(j + 1) * d] \
                                                                    + col_j*col_j.T/batch_size+self.alpha*eye(d)
            # end j
        # end i
        return hessian

    def get_loss(self, w):
        num = self.data_x.shape[1]
        sf = self.softmax(w.T*self.data_x)
        loss = -multiply(self.data_y, log(sf)).sum(axis=0).mean(axis=1)
        # wx = w.T*self.data_x
        # loss = -multiply(self.data_y, wx).sum(axis=0).sum(axis=1)
        # loss = loss + log(exp(wx).sum(axis=0)).sum(axis=1)
        loss = loss + 1/2*self.alpha*multiply(w, w).sum(axis=0).sum(axis=1)
        return asscalar(loss)

    def get_hessian_vector_product(self, w, index, batch_size, u):
        d, k = w.shape
        sf = self.softmax(w.T * self.data_x[:, index])
        # d, k = w.shape
        res = self.alpha*u
        # vt = vec_transpose(u, d)
        sf = self.softmax(w.T * self.data_x[:, index])
        # sf_x = multiply(kron(sf, ones((d, 1))), kron(ones((k, 1)), self.data_x[:, index]))
        # res = res - vec_transpose(sf_x*(sf_x.T*vt), d)/batch_size
        # res = res + multiply(sf.T, self.data_x[:, index] * (self.data_x[:, index].T * u))/batch_size
        # return res
        for i in range(k):
            row_i = multiply(sf[i, :], self.data_x[:, index])
            for j in range(k):
                col_j = multiply(sf[j, :], self.data_x[:, index])
                res[:, i] = res[:, i] - row_i*(col_j.T*u[:, j])/batch_size
                if i == j :
                    col_j = multiply(sqrt(sf[j, :]), self.data_x[:, index])
                    tmp = col_j*(col_j.T*u[:,j])/batch_size
                    res[:, i] = res[:, i] + tmp
                    # print("batchsize", batch_size)
                    # print("sf sum is", sf.sum(axis=0).mean(axis=1))
                    # print("tmp sum is", tmp.sum(axis=0))
                    # print("res[:,i] sum is",(res[:,i].sum(axis=0)))
            # end j
        # end i
        return res
        # ans = self.alpha*u
        # vt = vec_transpose(u, d)
        # sf = self.softmax(w.T * self.data_x[:,index])
        # ans = ans + multiply(sf.T, self.data_x[:,index]*(self.data_x[:,index].T*u))
        # tmp = kron(sf, self.data_x[:,index])
        # ans = ans - vec_transpose(tmp*(tmp.T*vt)/batch_size,d)
        # return ans

    def test_grad(self):
        print("-----------test grad start-----------")
        d, n = self.data_x.shape
        k = self.data_y.shape[0]
        w = np.mat(np.random.random((d, k)))
        for i  in range(2):
            print("start test",i)
            new_w = w.copy()
            l = np.random.randint(0, d)
            r = np.random.randint(0, k)
            delta = 1e-5
            new_w[l, r] = new_w[l, r] + delta
            grad = self.get_full_gradient(new_w)
            delta_loss = self.get_loss(new_w) - self.get_loss(w)
            print("delta loss is",(delta_loss))
            print("grad :", grad[l, r])
            print("grad*delta: ",grad[l, r]*delta)
            print("grad*delta-delta_loss: ",grad[l, r]*delta-delta_loss)
            print("grad*delta-delta_loss relative error: ",(grad[l, r]*delta-delta_loss)/delta_loss)

    def test_grad2(self):
        print("---------test grad2 start---------")
        d, n = self.data_x.shape
        k = self.data_y.shape[0]
        w = np.mat(np.random.random((d, k)))
        scale = 1e-6
        for i  in range(5):
            print("start test",i)
            delta_w = random.random((d,k))*scale
            new_w = w.copy() + delta_w
            grad = self.get_full_gradient(new_w)
            grad_vt = asscalar(multiply(grad, delta_w).sum(axis=1).sum(axis=0))
            delta_loss = self.get_loss(new_w) - self.get_loss(w)
            print("delta loss is",(delta_loss))
            print("grad :", norm(grad, ord=2))
            print("grad_vt: ",grad_vt)
            print("grad_vt-delta_loss: ",grad_vt-delta_loss)
            print("grad_vt-delta_loss relative error: ",(grad_vt-delta_loss)/delta_loss)
            
    def test_hessian(self):
        d, n = self.data_x.shape
        k = self.data_y.shape[0]
        w = mat(random.random((d, k)))
        scale = 1e-6
        num = 1000
        idx = random.choice(n, num, replace=False)
        hessian = self.get_hessian(w, idx, num)
        for i in range(20):
            print("start test",i)
            new_w = w.copy()+ mat(random.random((d, k)))*scale
            grad = self.get_gradient(w, idx, num)
            new_grad = self.get_gradient(new_w, idx, num)
            delta_grad = vec_transpose(new_grad - grad,d)
            hessian = self.get_hessian(w, idx, num)
            hessian_vt = hessian * vec_transpose(new_grad-grad, d)
            print("grad norm is:", norm(grad,ord=2))
            print("new_grad norm is:", norm(new_grad,ord=2))
            print("delta grad norm is:",norm(new_grad-grad,ord=2))
            print("hessian_vt norm is:", norm(hessian_vt,ord=2))
            print("hessian_vt&delta grad norm is: ",norm(hessian_vt-delta_grad,ord=2))
            print("hessian_vt&delta grad relative error is: ",norm(hessian_vt-delta_grad,ord=2)/norm(delta_grad,ord=2))
        
    def test_hessian_vector(self):
        d, n = self.data_x.shape
        k = self.data_y.shape[0]
        w = mat(random.random((d, k)))
        vt = mat(random.random((d, k)))        
        num = 1000    
        idx = random.choice(n, num, replace=False)
        hessian = self.get_hessian(w, idx, num)
        delta_w = hessian*vec_transpose(vt, d)
        delta_w2 = self.get_hessian_vector_product(w, idx, num, vt)
        delta_w2 = vec_transpose(delta_w2, d)
        print("w1 norm is", linalg.norm(delta_w, ord=2))
        print("w2 norm is", linalg.norm(delta_w2, ord=2))
        print("diff delta_w norm", linalg.norm(delta_w2 - delta_w, ord=2))








