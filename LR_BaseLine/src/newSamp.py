
import numpy as np
from Algorithm import *
import time
from scipy.sparse.linalg import svds
from Logger import *


class NewSamp(Algorithm):
    def __init__(self, train_x, train_y, alpha, epoch_num, rank, step_size, sample_size):
        Algorithm.__init__(self, train_x, train_y, alpha, epoch_num)
        self.step_size = step_size
        self.rank= rank
        self.sample_size = sample_size

    def get_params(self):
        return [self.rank, self.step_size, self.sample_size]

    def print_params(self):
        Logger.log("NewSamp parameters:\nstep_size:" + str(self.step_size)
                   + " rank:" + str(self.rank) + " sample_size:" + str(self.sample_size))

    def run(self):

        d, n = self.train_x.shape
        k = self.train_y.shape[0]
        eps = 1e-10
        last_w = np.mat(np.random.random((d, k))) * 10
        w = np.mat(np.random.rand(d, k))
        epoch_cnt = 0
        record = Record([], [], [])
        start = time.time()
        self.print_params()

        Logger.log("--------NewSamp start-------")
        while np.linalg.norm(last_w - w, ord=2) >= eps:
            last_w = w
            # select data
            idx = np.random.choice(n, self.sample_size)
            data_x = self.train_x[:, idx]
            data_y = self.train_y[:, idx]
            # compute gradient and hessian
            grad = self.get_gradient(w, self.train_x, self.train_y, self.alpha)
            epoch_cnt = epoch_cnt + 1
            hessian = self.get_hessian(w, data_x, data_y, self.alpha)
            rd = np.mat(np.random.random((d, d))) * 1e-8
            rd = rd * rd.T
            hessian = hessian + rd
            # TruncatedSVD
            u, s, v = svds(hessian, k=self.rank + 1)
            matrix_q = np.mat(np.eye(d)) / s[0] + np.mat(u[:, 1:]) * \
                       (np.mat(np.diagflat(1 / s[1:])) - 1 / s[0] * np.mat(np.eye(self.rank))) \
                       * np.mat(v[1:, :])
            w = w - self.step_size * matrix_q * grad
            epoch_cnt = epoch_cnt + 1.0*self.sample_size/n

            # record the loss and epoch_cnt and time
            start_loss_time = time.time()
            loss = self.get_loss(w, self.train_x, self.train_y, self.alpha)
            end_loss_time = time.time()
            start = start + end_loss_time - start_loss_time
            now = time.time() - start
            record.append(epoch_cnt, now, loss)
            Logger.log("EpochCnt :" + str(epoch_cnt) + " time:"
                + str(now) + "loss:" + str(loss))
            if epoch_cnt >= self.epoch_num:
                Logger.log("epoch_cnt is done")
                break
            if loss >= 100:
                Logger.log("bad iteration")
                break
        # end while
        Logger.log("NewSamp end")
        record.set_w(np.mat(w))
        record.get_best()
        return record

