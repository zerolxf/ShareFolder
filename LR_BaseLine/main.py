import numpy as np
import liSSA
import newSamp
import svrg
import loadData
from AlgorithmFactory import *
import copy
# 通过下面的方式进行简单配置输出方式与日志级别


class AlgorithmContext(object):

    def __init__(self, path, alpha, epoch_num):
        self.train_x, self.train_y = loadData.getMnist49(path)
        self.algorithm_fac = AlgorithmFactory(self.train_x, self.train_y, alpha, epoch_num)

    def get_best_params(self, name, test_num):
        best_record = Record([], [], [])
        for i in range(test_num):
            Logger.log("------" + name + " "+ str(i)+" test start------------")
            algorithm = self.algorithm_fac.get_algorithm(name)
            record = algorithm.run()
            record.get_best()
            Logger.log_now(record)
            if record < best_record:
                if record.best_epoch < best_record.best_epoch:
                    Logger.log("*****Get a better epoch number******")
                if record.best_loss < best_record.best_loss - 1e-15:
                    Logger.log("*****Get a better loss******")
                best_record = copy.deepcopy(record)
                b_params = algorithm.get_params()
                Logger.log_best(record)
            Logger.log("------"+ name + " "+ str(i)+" test end------------")
        return b_params, best_record

    def get_algorithm_with_params(self, name, params):
        return self.algorithm_fac.get_algorithm_with_params(name, params)


if __name__ == "__main__":
    context = AlgorithmContext("../", 0.0001, 50)
    # m, store_num, update_period,
    # b, bh, step_size
    slbfgs = context.get_algorithm_with_params("slbfgs", [1.0/100, 10, 10, 100, 1000, 0.02])
    slbfgs_record = slbfgs.run()
    Logger.log("result: epoch" + str(slbfgs_record.best_epoch)
               + " time:" + str(slbfgs_record.best_time) + "s loss:" + str(slbfgs_record.best_loss))
			   
			   
