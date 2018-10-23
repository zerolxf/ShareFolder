from numpy import *
import unittest
from scipy.special import expit
import scipy.misc
import sys
# sys.path.append('../')
# sys.path.insert(0, '/home/lxf/workstation/ML/LR/src/UnitTest/')
from Factory import *
# import Factory
import numpy as np
# from model import *
# from .. import *
class LRTest(unittest.TestCase):
    path="../resource/"
    model_name = "LR"
    fac = Factory(path, model_name)
    model = fac.model
#     def setUpClass(cls):
#         print("test LR setup")
#         path="../resource/"
#         model_name = "LR"
#         self.fac = Factory(path, model_name)
#         self.model = self.fac.model
        
    def __init__(self,*args, **kwargs):

        super(LRTest, self).__init__(*args, **kwargs)
        
#     def setUp(self):
#         print("test LR setup")
#         path="../resource/"
#         model_name = "LR"
#         self.fac = Factory(path, model_name)
#         self.model = self.fac.model
        
    def test_get_gradient(self):
        max_error = 1e-5
        scale = 1e-5
        mode_list = ["release", "debug"]
        d, n = self.model.data_x.shape
        k = self.model.data_y.shape[0]
        for mode in mode_list:
            self.mode = mode
            print("----------start test LR get gradient compute in ", self.mode, " mode")
            for i in range(20):
                idx = arange(self.model.n)
                w = mat(np.random.random((d, k)))
                delta_w = mat(np.random.random((d, k)))*scale
                new_w = w + delta_w
                grad = self.model.get_gradient(w, idx)
                delta_loss = self.model.get_loss(new_w) - self.model.get_loss(w)
                approx_delta_loss = np.asscalar(delta_w.T*grad)
                diff_loss = delta_loss- approx_delta_loss
                relative_error = diff_loss/delta_loss
                if relative_error > max_error:
                    print("*******************************")
                    print("test LR get gradient compute in ", self.mode, " mode error")
                    print("scale w", scale)
                    print("approx_delta_loss:", approx_delta_loss)
                    print("delta_loss:", delta_loss)
                    print("diff_loss:", diff_loss)
                    print("relative_error:", relative_error)
                    print("*******************************")
                    self.assertTrue(relative_error<max_error, "test get gradient error")
            print("end start test LR get gradient compute in ", self.mode, " mode ----------")
        
    def test_get_indiv_hessian(self):
        print("------------start test LR get_indiv_hessian")
        d, n = self.model.data_x.shape
        k = self.model.data_y.shape[0]
        scale_w = 1e-5
        max_error=2e-5
        test_num = 20
        for i in arange(test_num):
            w = np.random.random((d, k))
            idx = np.random.randint(self.model.n)
            delta_w = mat(np.random.random(w.shape))*1e-5
            new_w = w + delta_w
            delta_grad = self.model.get_indiv_gradient(new_w, idx) - self.model.get_indiv_gradient(w, idx)
            hessian = self.model.get_indiv_hessian(w, idx)
            delta_grad_hess = hessian*delta_w
            diff_grad = delta_grad_hess - delta_grad
            relative_error = norm(delta_grad-delta_grad_hess)/norm(delta_grad)
            if (relative_error > max_error):
                print("****************************************")
                print("scale_w:", scale_w)
                print("delta_grad norm is", norm(delta_grad))
                print("delta_grad_hess norm is", norm(delta_grad_hess))
                print("delta_grad diff norm is", norm(delta_grad-delta_grad_hess))
                print("delta_grad diff relative error is", norm(delta_grad-delta_grad_hess)/norm(delta_grad))
                print("****************************************")
                self.assertTrue(relative_error<max_error, "test get indiv hessian error")
        print("end test LR get_indiv_hessian-----------")
    
    def test_get_hessian(self):
        mode_list = ["release", "debug"]
        d, n = self.model.data_x.shape
        k = self.model.data_y.shape[0]
        scale_w = 1e-5
        max_error=1e-5
        test_num = 20
        for mode in mode_list:
            self.mode = mode
            print("----------start test LR get hessian compute in ", self.mode, " mode")
            for i in arange(test_num):
                w = np.random.random((d, k))
                idx = arange(self.model.n)
                delta_w = mat(np.random.random(w.shape))*1e-5
                new_w = w + delta_w
                delta_grad = self.model.get_gradient(new_w, idx) - self.model.get_gradient(w, idx)
                hessian = self.model.get_hessian(w, idx)
                delta_grad_hess = hessian*delta_w
                diff_grad = delta_grad_hess - delta_grad
                relative_error = norm(delta_grad-delta_grad_hess)/norm(delta_grad)
                if (relative_error > max_error):
                    print("****************************************")
                    print("scale_w:", scale_w)
                    print("delta_grad norm is", norm(delta_grad))
                    print("delta_grad_hess norm is", norm(delta_grad_hess))
                    print("delta_grad diff norm is", norm(delta_grad-delta_grad_hess))
                    print("delta_grad diff relative error is", norm(delta_grad-delta_grad_hess)/norm(delta_grad))
                    print("****************************************")
                    self.assertTrue(relative_error<max_error, "test get hessian error")
            print("end test LR get_hessian compute in ", self.mode, " mode-----------")   
    
    def test_get_indiv_hessian_vector(self):
        d, n = self.model.data_x.shape
        k = self.model.data_y.shape[0]
        max_error=1e-9
        test_num = 20
        print("------------start test LR get indiv hessian vector")
        for i in arange(test_num):
            u = np.random.random((d, k))
            w = np.random.random((d, k))
            idx = np.random.randint(self.model.n)
            hessian = self.model.get_indiv_hessian(w, idx)
            print(type(hessian))
            print(hessian.shape)
            hessian_vt_exact = np.dot(hessian,u)
            print(type(hessian_vt_exact))
            print(hessian_vt_exact.shape)
            hessian_vt = self.model.get_indiv_hessian_vector(w, idx, u)
            diff_hessian_vt = hessian_vt_exact - hessian_vt
            if (norm(diff_hessian_vt) > max_error):
                print("****************************************")
                print("hessian norm is", norm(hessian))
                print("hessian_vt_exact norm is", norm(hessian_vt_exact))
                print("hessian_vt norm is", norm(hessian_vt))
                print("diff_hessian_vt error is", norm(diff_hessian_vt))
                print("****************************************")
                self.assertTrue(norm(diff_hessian_vt)<max_error, "test get indiv hessian vector error")
        print("end test LR get indiv hessian vector-----------")
    
    
    def test_get_hessian_vector(self):
        d, n = self.model.data_x.shape
        k = self.model.data_y.shape[0]
        mode_list = ["release", "debug"]
        max_error=1e-8
        test_num = 20
        for mode in mode_list:
            self.mode = mode
            print("------------start test LR test_get_hessian_vector in", self.mode, " mode")
            for i in arange(test_num):
                u = np.random.random((d, k))
                w = np.random.random((d, k))
                idx = arange(self.model.n)
                hessian = self.model.get_hessian(w, idx)
                hessian_vt_exact = hessian*vec_transpose(u, d)
                hessian_vt = self.model.get_hessian_vector(w, idx, u)
                diff_hessian_vt = hessian_vt_exact - hessian_vt
                if (norm(diff_hessian_vt) > max_error):
                    print("****************************************")
                    print("hessian norm is", norm(hessian))
                    print("hessian_vt_exact norm is", norm(hessian_vt_exact))
                    print("hessian_vt norm is", norm(hessian_vt))
                    print("diff_hessian_vt error is", norm(diff_hessian_vt))
                    print("****************************************")
                    self.assertTrue(norm(diff_hessian_vt)<max_error, "test get indiv hessian vector error")
            print("end test LR test_get_hessian_vector in", self.mode, " mode-----------")
        
if __name__ == '__main__':
    unittest.main()
    
#     def test_LR(self):
#         self.test_get_gradient()
#         self.test_get_indiv_hessian()
#         self.test_get_hessian()
#         self.test_get_indiv_hessian_vector()
#         self.test_get _hessian_vector()
        
            
            
                
                







