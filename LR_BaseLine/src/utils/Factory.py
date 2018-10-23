from numpy import *
from model.model import *
from solver.Solver import *
from utils.loadData import *
from model.LRSoftmax import *
from model.LR import *
import solver.second_order.Lant as Lant
from solver.second_order.Lant import *
import solver.second_order.liSSA as liSSA
import solver.second_order.newSamp as newSamp
import solver.first_order.svrg as svrg
from solver.distributed_system.CcoSgd import *
from solver.distributed_system.MemSgd import *
from solver.distributed_system.RestartSgd import *
from solver.distributed_system.MiniSgd import *

import solver.distributed_system.CcoSgd as CcoSgd
import solver.distributed_system.MemSgd as MemSgd
import solver.distributed_system.RestartSgd as RestartSgd
import solver.distributed_system.MiniSgd as MiniSgd

class Factory(object):
    def __init__(self, path="../resource/", model_name = "LR"):
        self.path = path
        self.model = self.get_model(model_name)
        self.model_name = model_name

    def get_model(self, model_name):
        if model_name == "LR" :
            data_x, data_y = getMnist49(path=self.path)
            return LR(data_x, data_y, 1e-4)
        elif model_name == "LRSoftmax":
            data_x, data_y = getMnistWithNumber(path=self.path, number = array([0,1,2,3,4,5,6,7,8,9]))
            # print("data_x mean",data_x.mean(axis=1))
            return LRSoftmax(data_x, data_y, 1e-4)
        else:
            pass

    def get_solver(self, solver_name, params):

        if solver_name == "Svrg":
            return svrg.Svrg(self.model, params[0], params[1])
        elif solver_name == "NewSamp":
            return newSamp.NewSamp(self.model, params[0], params[1], params[2])
        elif solver_name == "LiSSA":
            return liSSA.LiSSA(self.model, params[0], params[1], params[2], params[3])
        elif solver_name == "Slbfgs":
            return slbfgs.Slbfgs(self.model, params[0], params[1], params[2], params[3], params[4], params[5])
        else:
            Logger.log(solver_name+" not implement")
            pass

    def get_solver_with_rand_params(self, solver_name):
        w_scale_lists = np.power(0.1, np.arange(3) + 3)
        w_scale = asscalar(random.choice(w_scale_lists, 1))
        if solver_name == 'Svrg':
            # parameters in paper
            if self.model_name == "LRSoftmax":
                # step_size = asscalar(random.choice((arange(22) + 8) * 0.05, 1))
                return self.get_solver(solver_name, [0.025, 2])
            else:
                step_size = asscalar(random.choice((arange(26) + 4) * 0.05, 1))
                s1 = asscalar(random.choice(np.arange(1) + 1, 1))
                return self.get_solver(solver_name, [step_size, s1])
        elif solver_name == "NewSamp":
            step_size = asscalar(random.choice((arange(24)+6)*0.05, 1))
            rank = asscalar(random.choice((np.arange(10)+2)*10, 1))
            sample_sz = asscalar(random.choice((np.arange(10)+2)*1000, 1))
            return self.get_solver(solver_name, [rank, step_size, sample_sz])
        elif solver_name == "LiSSA":
            t1 = random.choice(arange(5) + 1)
            s1 = random.choice(np.arange(1) + 1)
            s2 = random.choice(1000 * (np.arange(10) + 2))
            if self.model_name == "LRSoftmax":
                return self.get_solver(solver_name, [6, s1, s2, 0.025])
            else:
                step_size = asscalar(random.choice((arange(20) + 8) * 0.05, 1))
                return self.get_solver(solver_name, [t1, s1, s2, step_size])
            # t1, s1, s2, step_size
        else:
            pass



