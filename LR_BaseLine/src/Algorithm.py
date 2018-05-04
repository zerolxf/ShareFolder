
import numpy as np
import pandas as pd


class Algorithm(object):
    def __init__(self, train_x, train_y, alpha, epoch_num):
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = alpha
        self.epoch_num = epoch_num
        pass

    def get_params(self):
        print("algorithm get_params is not implemented")

    def run(self):
        print("algorithm run is not implemented")

    def print_params(self):
        print("algorithm print is not implemented")

    @staticmethod
    def softmax(data_x):
        ex = np.exp(data_x)
        return ex / (ex + 1)

    @staticmethod
    def get_loss(w, data_x, data_y, alpha):
        n = data_x.shape[1]
        oneN = np.mat(np.ones((n, 1)))
        loss = np.asscalar((np.log(np.exp(w.T * data_x) + 1) * oneN
                            - w.T * data_x * data_y.T) / n)
        penalty = alpha * np.asscalar(w.T * w)
        loss = loss + penalty / 2.0
        return loss

    def get_gradient(self, w, data_x, data_y, alpha):
        size = data_x.shape[1]
        if size <= 0:
            print("size can't be 0")
            return
        return data_x * (self.softmax(w.T * data_x) - data_y).T / size + alpha * w

    def get_hessian(self, w, data_x, data_y, alpha):
        d, size = data_x.shape
        sm = self.softmax(w.T * data_x)
        sqrt_sm = np.power(sm, 0.5)
        tmp1 = np.multiply(data_x, sqrt_sm)
        tmp2 = np.multiply(data_x, sm)
        hessian = (tmp1 * tmp1.T - tmp2 * tmp2.T) / size
        hessian = hessian + alpha * np.mat(np.eye(d))
        return hessian


def first(the_iterable, condition=lambda x: True):
    for idx, v in enumerate(the_iterable):
        if condition(v):
            return idx


class Record(object):

    def __init__(self, epoch_list, time_list, loss_list):
        self.w = []
        self.epoch_list = epoch_list
        self.time_list = time_list
        self.loss_list = loss_list
        self.best_loss = 1e9
        self.best_epoch = 1e9
        self.best_time = 1e9

    def get_best(self):
        self.best_loss = min(self.loss_list)
        idx = first(self.loss_list, lambda x: abs(x - self.best_loss) <= 1e-15)
        self.best_time = self.time_list[idx]
        self.best_epoch = self.epoch_list[idx]

    def append(self, epoch, t, loss):
        self.epoch_list.append(epoch)
        self.time_list.append(t)
        self.loss_list.append(loss)

    def set_w(self, w):
        self.w = w

    def get_w(self):
        return self.w

    def __lt__(self, other):
        return ((self.best_loss < other.best_loss - 1e-15) or
                ((abs(self.best_loss - other.best_loss) < 1e-15) and
                 self.best_time < other.best_time))
