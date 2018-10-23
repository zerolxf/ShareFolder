
import numpy as np
import time
from solver.first_order.svrg import *
from solver.first_order.sgd import *
from model.model import *
from solver.Solver import *

class LiSSA(Solver):
    def __init__(self, model, t1, s1, s2, step_size):
        Solver.__init__(self, model, step_size)
        self.t1 = t1
        self.s1 = s1
        self.s2 = s2
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.name = "LiSSA"

    def get_params(self):
        params = {"t1":self.t1, "s1":self.s1, "s2":self.s2, "step_size":self.step_size}
        return params

    def run(self, max_epoch_num):
        # get a closer w by svrg
        sgd = Sgd(self.model, 0.1)
        record = sgd.run(1)

        w = record.get_w().copy()
        epoch_cnt = max(record.epoch_list)
        start = time.time() - max(record.time_list)

        self.print_params()
        Logger.log_start(self.name)
        while epoch_cnt < max_epoch_num:
            u = []
            grad = self.get_full_gradient(w)
            epoch_cnt = epoch_cnt + 1
            for i in range(self.s1):
                # set u[i] = grad
                u.append(grad)
                for j in range(self.s2):
                    idx = np.random.randint(self.n)
                    vt = self.get_hessian_vector_product(w, idx, u[i])
                    u[i] = (grad + u[i] - vt)
                # end j
            # end i
            delta = np.mat(np.mean(u, axis=0))
            w = w - delta

            # record the loss and epoch_cnt and time
#             epoch_cnt = epoch_cnt + 1.0*(self.s1*self.s2)/self.n
            start_loss_time = time.time()
            loss = self.get_loss(w)
            end_loss_time = time.time()
            start = start + end_loss_time - start_loss_time
            now = time.time() - start
            Logger.log("iterNum :" + str(epoch_cnt) + " time:"
                       + str(now) + "loss:" + str(loss))
            record.append(epoch_cnt, now, loss)
            if loss >= 100:
                Logger.log("bad iteration")
                break

        # end t
        Logger.log_end(self.name)
        record.set_w(w)
        record.get_best()
        return record

