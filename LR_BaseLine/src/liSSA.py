
import numpy as np
from Algorithm import *
from svrg import *
import time
from scipy.sparse.linalg import svds


class LiSSA(Algorithm):
    def __init__(self, train_x, train_y, alpha, epoch_num, t1, s1, s2, step_size):
        Algorithm.__init__(self, train_x, train_y, alpha, epoch_num)
        self.step_size = step_size
        self.t1 = t1
        self.s1 = s1
        self.s2 = s2

    def get_params(self):
        return [self.step_size, self.t1, self.s1, self.s2]

    def print_params(self):
        Logger.log("Lissa:\nstepSize:" + str(self.step_size) + " t1:"
              + str(self.t1) + " s1:" + str(self.s1) + " s2:" + str(self.s2))

    def run(self):
        # get a closer w by svrg
        svrg = Svrg(self.train_x, self.train_y, self.alpha, self.t1, self.step_size, 1)
        record = svrg.run()

        w = record.get_w().copy()
        d, n = self.train_x.shape
        k = self.train_y.shape[0]
        epoch_cnt = max(record.epoch_list)
        start = time.time() - max(record.time_list)

        self.print_params()
        Logger.log("----------lissa start-------------")
        while epoch_cnt < self.epoch_num:
            u = []
            grad = self.get_gradient(w, self.train_x, self.train_y, self.alpha)
            epoch_cnt = epoch_cnt + 1
            for i in range(self.s1):
                # set u[i] = grad
                u.append(grad)
                # index = np.random.choice(n, self.s2)
                # data_x = self.train_x[:, index]
                # data_y = np.mat(self.train_y[:, index])
                # hessian = self.get_hessian(w, data_x, data_y, self.alpha)
                bt_sz = 1
                sz = int(self.s2/bt_sz)
                for _ in range(sz):
                    tmp = np.mat(np.zeros((d, 1)))
                    for k in range(bt_sz):
                        idx = np.random.randint(n)
                        data_x = self.train_x[:, idx]
                        smj = np.asscalar(self.softmax(w.T * data_x))
                        hessian_value = smj * (1 - smj)
                        tmp = tmp + hessian_value * data_x * (data_x.T * u[i])
                    tmp = tmp /bt_sz
                        # idx = np.random.randint(n)
                        # data_x = self.train_x[:, idx]
                        # smj = np.asscalar(self.softmax(w.T * data_x))
                        # hessian_value = smj * (1 - smj)
                    # u[i] = (grad + u[i] - hessian_value * data_x * (data_x.T * u[i]))
                    u[i] = (grad + u[i] - tmp)
                # end j
            # end i
            delta = np.mat(np.mean(u, axis=0))
            w = w - delta

            # record the loss and epoch_cnt and time
            epoch_cnt = epoch_cnt + 1.0*(self.s1*self.s2)/n
            start_loss_time = time.time()
            loss = self.get_loss(w, self.train_x, self.train_y, self.alpha)
            end_loss_time = time.time()
            start = start + end_loss_time - start_loss_time
            now = time.time() - start
            Logger.log("iterNum :" + str(epoch_cnt) + " time:"
                       + str(now) + "loss:" + str(loss))
            record.append(epoch_cnt, now, loss)

        # end t
        Logger.log("-------------Lissa end---------")
        record.set_w(w)
        record.get_best()
        return record

