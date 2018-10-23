from numpy import *
from model.model import *


class Solver(object):
    def __init__(self, model, step_size):
        self.model = model
        self.step_size = step_size

    def get_full_gradient(self, w):
        return self.model.get_full_gradient(w)

    def get_params(self):
        Logger.log("Solver get_params not implement")
        raise NotImplementedError

    def print_params(self):
        params = self.get_params()
        Logger.log_params(params)

    def run(self, max_epoch_num):
        Logger.log("Solver run not implement")
        raise NotImplementedError

    def get_indiv_gradient(self, w, index):
        return self.model.get_indiv_gradient(w, index)
    
    def get_gradient(self, w, index, batch_size=1):
        return self.model.get_gradient(w, index)

    def get_indiv_hessian(self, w, index):
        return self.model.get_indiv_hessian(w, index)
    
    def get_hessian(self, w, index):
        return self.model.get_hessian(w, index)

    def get_loss(self, w):
        return self.model.get_loss(w)

    def get_indiv_hessian_vector(self, w, index, u):
        return self.model.get_indiv_hessian_vector(w, index, u)
    
    def get_hessian_vector(self, w, index, u):
        return self.model.get_hessian_vector(w, index, u)


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
        return ((self.best_loss < other.best_loss - 1e-14) or
                ((abs(self.best_loss - other.best_loss) < 1e-14) and
                 self.best_epoch < other.best_epoch))