import sys
import numpy as np
from solver.Solver import *
from model.model import *
import time
from utils.Logger import *

# svrg
class Svrg(Solver):
    def __init__(self, model, step_size, m):
        Solver.__init__(self, model, step_size)
        self.m = m
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.name = "Svrg"

    def get_params(self):
        params = {"step_size": self.step_size, "m": self.m}
        return params

    def run(self, max_epoch_num):


        m = int(self.m * self.n)
        w = np.mat(np.random.randn(self.d, self.k))
        epoch_cnt: int = 0      # count the number of epoch
        t = 0           # count the numbers of iteration
        start = time.time()
        self.print_params()
        record = Record([], [], [])

#         sgd = Sgd(self.model, 0.1)
#         record = sgd.run(4)

#         w = record.get_w().copy()
#         epoch_cnt = max(record.epoch_list)
#         start = time.time() - max(record.time_list)

        Logger.log_start(self.name)
        while epoch_cnt < max_epoch_num:
            u = self.get_full_gradient(w)
            epoch_cnt = epoch_cnt + 1
            wt = w.copy()
            wt = wt - self.step_size*u
            for _ in range(m):
                t = t + 1
                idx = np.random.randint(self.n)
                delta = self.get_indiv_gradient(wt, idx) - self.get_indiv_gradient(w, idx)
                delta = delta + u
                wt = wt - self.step_size * delta

                # record the loss and epoch_cnt and time
#                 if t % self.n == 0:
                if _ == m-1:
#                     epoch_cnt = epoch_cnt + 1
                    start_loss_time = time.time()
                    loss = self.get_loss(wt)
                    end_loss_time = time.time()
                    start = start + end_loss_time - start_loss_time
                    now = time.time() - start
                    Logger.log("iterNum :"+str(epoch_cnt)+" time:"
                               + str(now) + "loss:"+str(loss))
                    record.append(epoch_cnt, now, loss)
                    t = 0
                    # if epoch_cnt >= max_epoch_num:
                    #     break
                    # if loss >= 100:
                    #     Logger.log("bad iteration")
                    #     break
            w = wt.copy()
        Logger.log_end(self.name)
        record.set_w(np.mat(w))
        record.get_best()
        return record
