
import numpy as np
from model.model import *
from solver.Solver import *
import time
from scipy.sparse.linalg import svds
from utils.Logger import *
from solver.first_order.sgd import *

class NewSamp(Solver):
    def __init__(self, model, rank, step_size, sample_size):
        Solver.__init__(self, model, step_size)
        self.rank = rank
        self.sample_size = sample_size
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.name = "NewSamp"

    def get_params(self):
        return {"rank":self.rank, "step_size":self.step_size, "sample_size":self.sample_size}

    def run(self, max_epoch_num):
        start_svrg = True
        record = None
        w = None
        epoch_cnt = None
        start = True

        if not start_svrg:
            w = np.mat(np.random.randn(self.d, self.k))
            epoch_cnt: int = 0  # count the number of epoch
            start = time.time()
            record = Record([], [], [])
        else:
            sgd = Sgd(self.model, 0.1)
            record = sgd.run(2)
            w = record.get_w().copy()
            epoch_cnt = max(record.epoch_list) 
            start = time.time() - max(record.time_list)
        eps = 1e-10
        last_w = np.mat(np.random.random((self.d, self.k))) * 10
        self.print_params()
        
        
        
#         eps = 1e-10
#         last_w = np.mat(np.random.random((self.d, self.k))) * 10
#         w = np.mat(np.random.rand(self.d, self.k))
#         epoch_cnt = 0
#         record = Record([], [], [])
#         start = time.time()
#         self.print_params()

        Logger.log_start(self.name)
        while np.linalg.norm(last_w - w, ord=2) >= eps:
            last_w = w
            # select data
            idx = np.random.choice(self.n, self.sample_size)
            # compute gradient and hessian
            grad = self.get_full_gradient(w)
            epoch_cnt = epoch_cnt + 1
            hessian = self.get_hessian(w, idx)
            rd = np.random.rand()*1e-16
            # hessian = hessian + rd * eye(self.d * self.k)
            # TruncatedSVD
            u, s, v = svds(hessian, k=self.rank + 1)
            matrix_q = np.mat(np.eye(self.d * self.k)) / s[0] + np.mat(u[:, 1:]) * \
                       (np.mat(np.diagflat(1 / s[1:])) - 1 / s[0] * np.mat(np.eye(self.rank))) \
                       * np.mat(v[1:, :])
            grad_vt = vec_transpose(grad, self.d)
            delta_w = matrix_q * grad_vt
            delta_w = vec_transpose(delta_w, self.d)
            # print("w sum",w.sum(axis=0))
            # print("grad sum", grad.sum(axis=0))
            # print("delta_w sum", delta_w.sum(axis=0))
            w = w - self.step_size * delta_w
#             epoch_cnt = epoch_cnt + 1

            # record the loss and epoch_cnt and time
            start_loss_time = time.time()
            loss = self.get_loss(w)
            end_loss_time = time.time()
            start = start + end_loss_time - start_loss_time
            now = time.time() - start
            record.append(epoch_cnt, now, loss)
            Logger.log("EpochCnt :" + str(epoch_cnt) + " time:"
                       + str(now) + "loss:" + str(loss))
            if epoch_cnt >= max_epoch_num:
                Logger.log("epoch_cnt is done")
                break
            if loss >= 100:
                Logger.log("bad iteration")
                break
        # end while
        Logger.log_end(self.name)
        record.set_w(np.mat(w))
        record.get_best()
        return record

