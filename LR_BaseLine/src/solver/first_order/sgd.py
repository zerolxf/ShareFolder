import sys
import numpy as np
from solver.Solver import *
from model.model import *
import time
from utils.Logger import *

# svrg
class Sgd(Solver):
    def __init__(self, model, step_size):
        Solver.__init__(self, model, step_size)
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.name = "Sgd"

    def get_params(self):
        params = {"step_size": self.step_size}
        return params

    def run(self, max_epoch_num):
        w = np.mat(np.random.randn(self.d, self.k))
        epoch_cnt: int = 0      # count the number of epoch
        t = 0           # count the numbers of iteration

        self.print_params()
        record = Record([], [], [])


        Logger.log_start(self.name)
        start = time.time()
        while epoch_cnt < max_epoch_num:
            epoch_cnt = epoch_cnt + 1
            for _ in range(self.n):
                t = t + 1
                idx = np.random.choice(self.n,1)
                w = w - self.step_size * self.get_gradient(w, idx)

            # record the loss and epoch_cnt and time
            start_loss_time = time.time()
            loss = self.get_loss(w)
            end_loss_time = time.time()
            start = start + end_loss_time - start_loss_time
            now = time.time() - start
            Logger.log("iterNum :"+str(epoch_cnt)+" time:"
                       + str(now) + "loss:"+str(loss))
            record.append(epoch_cnt, now, loss)
        Logger.log_end(self.name)
        record.set_w(np.mat(w))
        record.get_best()
        return record
