import numpy as np
import solver.second_order.Lant
from solver.second_order.Lant import *
import solver.second_order.liSSA
import solver.second_order.newSamp
import solver.first_order.svrg
from solver.distributed_system.CcoSgd import *
from solver.distributed_system.MemSgd import *
from solver.distributed_system.RestartSgd import *
from solver.distributed_system.MiniSgd import *

import utils.loadData
from  utils.Factory import *
import copy


# 通过下面的方式进行简单配置输出方式与日志级别

class SolverContext(object):

    def __init__(self, path="../resource/", model_name = "LR"):
        self.fac = Factory(path, model_name)

    def get_best_params(self, name, test_num, max_epoch_num):
        best_record = Record([], [], [])
        for i in range(test_num):
            Logger.log("------" + name + " "+ str(i)+" test start------------")
            algorithm = self.fac.get_solver_with_rand_params(name)
            algorithm.model.mode = "release"
            record = algorithm.run(max_epoch_num)
            record.get_best()
            Logger.log_now(record)
            if record < best_record:
                if record.best_epoch < best_record.best_epoch:
                    Logger.log("*****Get a better iter number******")
                if record.best_loss < best_record.best_loss - 1e-15:
                    Logger.log("*****Get a better loss******")
                best_record = record
                b_params = algorithm.get_params()
                Logger.log_best(record)
            Logger.log("------"+ name + " "+ str(i)+" test end------------")
        Logger.log_params(b_params)
        Logger.log_best(best_record)
        return b_params, best_record

def run_newSamp(context):
    newSamp_params = [500, 1.2, 40000]
    newSamp = context.fac.get_solver("NewSamp", newSamp_params)
    newSamp_record = newSamp.run(40)
    pd.DataFrame([newSamp_record.epoch_list, newSamp_record.time_list, newSamp_record.loss_list]) \
        .to_csv("../resource/newSamp.csv", header=None)
    
def run_Svrg(context):
    svrg_params = [0.02, 1]
    svrg = context.fac.get_solver("Svrg", svrg_params)
    svrg_record = svrg.run(100)
    pd.DataFrame([svrg_record.epoch_list, svrg_record.time_list, svrg_record.loss_list]) \
        .to_csv("../resource/svrg_mnist_tmp.csv", header=None)

def run_Lissa(context):
        # Lissa
    lissa_params = [1, 1, 7500, 2]
    lissa = context.fac.get_solver("LiSSA", lissa_params)
    lissa_record = lissa.run()
    pd.DataFrame([lissa_record.epoch_list, lissa_record.time_list, lissa_record.loss_list])\
        .to_csv("../resource/lissa.csv", header=None)
    
    
    
if __name__ == "__main__":
    
    context = SolverContext("../resource/", "LR")
    run_Svrg(context)
    run_newSamp(context)
    run_Lissa(context)
#     algo = context.fac.get_solver("Lant", [1e-4, 60, 0.7, 60])
#     algo.model.test_LR()
#     context.get_best_params("NewSamp", 200, 30)

