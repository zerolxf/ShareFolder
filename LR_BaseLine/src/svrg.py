
import numpy as np
from Algorithm import *
import time
from Logger import *


# svrg
class Svrg(Algorithm):
    def __init__(self, train_x, train_y, alpha, epoch_num, step_size, m):
        Algorithm.__init__(self, train_x, train_y, alpha, epoch_num)
        self.step_size = step_size
        self.m = m

    def get_params(self):
        return [self.step_size, self.m]

    def print_params(self):
        Logger.log("Svrg parameters:\nstepSize:"+
                   str(self.step_size) + " m:"+str(self.m))

    def run(self):

        d, n = self.train_x.shape
        k = self.train_y.shape[0]
        m = int(self.m * n)
        w = np.mat(np.random.rand(d, k))
        epoch_cnt: int = 0      # count the number of epoch
        t = 0           # count the numbers of iteration
        start = time.time()
        self.print_params()
        record = Record([], [], [])

        Logger.log("-------Svrg start---------")
        while epoch_cnt < self.epoch_num:
            u = self.get_gradient(w, self.train_x, self.train_y, self.alpha)
            epoch_cnt = epoch_cnt + 1
            wt = w
            for _ in range(m):
                t = t + 1
                idx = np.random.randint(n)
                data_x = self.train_x[:, idx]
                data_y = np.mat(self.train_y[:, idx])
                delta = self.get_gradient(wt, data_x, data_y, self.alpha) \
                        - self.get_gradient(w, data_x, data_y, self.alpha)
                delta = delta + u
                wt = wt - self.step_size * delta

                # record the loss and epoch_cnt and time
                if t % n == 0:
                    epoch_cnt = epoch_cnt + 1
                    start_loss_time = time.time()
                    loss = self.get_loss(wt, self.train_x, self.train_y, self.alpha)
                    end_loss_time = time.time()
                    start = start + end_loss_time - start_loss_time
                    now = time.time() - start
                    Logger.log("iterNum :"+str(epoch_cnt)+" time:"
                               + str(now) + "loss:"+str(loss))
                    record.append(epoch_cnt, now, loss)
                    if epoch_cnt >= self.epoch_num:
                        break
            w = wt
        Logger.log("---------Svrg end-------------")
        record.set_w(np.mat(w))
        record.get_best()
        return record
