import numpy as np
from svrg import *
from newSamp import *
from liSSA import *
from slbfgs import *
class AlgorithmFactory(object):
    def __init__(self, train_x, train_y, alpha, epoch_num):
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = alpha
        self.epoch_num = epoch_num

    def get_algorithm(self, algorithm):
        if algorithm == 'svrg':
            step_size_lists = 0.25 * (np.arange(8) + 4)
            m_lists = np.arange(3) + 1
            idx1 = np.random.randint(8)
            idx2 = np.random.randint(3)
            return Svrg(self.train_x, self.train_y, self.alpha, self.epoch_num,
                        step_size_lists[idx1], m_lists[idx2])
        elif algorithm == "NewSamp":
            rank_lists = 10 * (np.arange(5) + 3)
            step_size_lists = 0.25 *(np.arange(5)+1)
            sample_sz_lists = 2500 *(np.arange(10)+1)
            idx1 = np.random.randint(5)
            idx2 = np.random.randint(5)
            idx3 = np.random.randint(10)
            return NewSamp(self.train_x, self.train_y, self.alpha, self.epoch_num,
                           rank_lists[idx1], step_size_lists[idx2], sample_sz_lists[idx3])
        elif algorithm == "liSSA":
            t1_lists = np.arange(5) + 1
            s1_lists = np.arange(1) + 1
            s2_lists = 2500 * (np.arange(10) + 2)
            step_size_lists = 0.25 * (np.arange(4) + 6)
            idx1 = np.random.randint(5)
            idx2 = np.random.randint(1)
            idx3 = np.random.randint(10)
            idx4 = np.random.randint(4)
            # t1, s1, s2, step_size
            return LiSSA(self.train_x, self.train_y, self.alpha, self.epoch_num,
                         t1_lists[idx1], s1_lists[idx2], s2_lists[idx3], step_size_lists[idx4])
        elif algorithm == "slbfgs":
			# 
            pass
        else:
            print(algorithm+" not implement")

    def get_algorithm_with_params(self, algorithm, params):
        if algorithm == 'svrg':
            return Svrg(self.train_x, self.train_y, self.alpha, self.epoch_num,
                        params[0], params[1])
        elif algorithm == "newSamp":
            return NewSamp(self.train_x, self.train_y, self.alpha, self.epoch_num,
                           params[0], params[1], params[2])
        elif algorithm == "liSSA":
            # t1, s1, s2, step_size
            return LiSSA(self.train_x, self.train_y, self.alpha, self.epoch_num,
                         params[0], params[1], params[2], params[3])
        elif algorithm == "slbfgs":
            return slbfgs(self.train_x, self.train_y, self.alpha, self.epoch_num,
                          params[0], params[1], params[2], params[3], params[4], params[5])
        else:
            print(algorithm+" not implement")