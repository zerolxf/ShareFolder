
import numpy as np
from Algorithm import *
import time
from Logger import *


class slbfgs(Algorithm):
    # 对于所有实验, 我们设置批处理大小b的为20或者100, 我们设置Hessian的批处理大小b_H为10b或者20b, \
    # 设置Hessian更新间隔L为10, 设置存储大小M为10, 设置随机更新次数m为N / b.我们通过网格搜索来优化学习率
    def __init__(self, train_x, train_y, alpha, epoch_num, m, store_num, update_period,
                 b, bh, step_size):
        Algorithm.__init__(self, train_x, train_y, alpha, epoch_num)
        self.store_num = store_num
        self.update_period = update_period
        self.b = b
        self.bh = bh
        self.step_size = step_size
        self.m = m

    def get_params(self):
        return [self.m, self.store_num, self.update_period,
                self.b, self.bh, self.step_size]

    def print_params(self):
        Logger.log("SLBFGS parameters:\nm:" + str(self.m) +" M:" + str(self.store_num) +
                   " L: " + str(self.update_period) + " b:" + str(self.b) +
                   " bh:" + str(self.bh) + "stepSize:" + str(self.step_size))

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
        r: int = 0
        data_cnt: int = 0
        Logger.log("-------SLBFGS start---------")
        s_array = np.mat(np.zeros((d, self.store_num)))
        y_array = np.mat(np.zeros((d, self.store_num)))
        x_array = np.mat(np.zeros((d, self.update_period)))
        last_u = w
        while epoch_cnt < self.epoch_num:
            # y_array, s_array record last M (sr, yr), last idx = (r - 1)%M
            u = self.get_gradient(w, self.train_x, self.train_y, self.alpha)
            epoch_cnt = epoch_cnt + 1
            wt = w
            for i in range(m):
                t = t + 1
                idx = np.random.choice(n, self.b, replace=False)
                data_cnt = data_cnt + self.b
                data_x = self.train_x[:, idx]
                data_y = self.train_y[:, idx]
                delta = self.get_gradient(wt, data_x, data_y, self.alpha) \
                        - self.get_gradient(w, data_x, data_y, self.alpha)
                delta = delta + u
                old_w = wt.copy()
                if r == 0:
                    wt = wt - self.step_size*delta
                else:
                    last_num = min(r, self.store_num)
                    tmp_sum = np.mat(np.zeros(delta.shape))
                    for j in range(last_num+1):
                        right_vt = delta
                        # compute right product with delta
                        for jj in range(j):
                            idx_jj = (r-1-jj)%self.store_num
                            rho_jj = 1 / (np.mat(s_array[:,idx_jj]).T * np.mat(y_array[:,idx_jj]))
                            right_vt = right_vt - s_array[:,idx_jj] * rho_jj * \
                                       (y_array[:,idx_jj].T * right_vt)
                        # compute (last_num -j) element with right_vt
                        if j == last_num:
                            idx_j = (r-1)%self.store_num
                            init_hessi = np.mat(s_array[:,idx_j]).T * np.mat(y_array[:,idx_j])\
                                         /(np.mat(y_array[:,idx_j]).T * np.mat(y_array[:,idx_j]))
                            right_vt = right_vt*init_hessi
                        else:
                            idx_j = (r - 1 - j) % self.store_num
                            rho_j = 1 / (np.mat(s_array[:,idx_j]).T * np.mat(y_array[:,idx_j]))
                            right_vt =  s_array[:,idx_j] * rho_j * (s_array[:,idx_j].T * right_vt)
                        # compute left product with right_vt
                        for jj in range(j):
                            idx_jj = (r - 1 - jj) % self.store_num
                            rho_jj = 1 / (np.mat(s_array[:,idx_jj]).T * np.mat(y_array[:,idx_jj]))
                            right_vt = right_vt -  y_array[:,idx_jj] * \
                                       (rho_jj * s_array[:,idx_jj].T  * right_vt)
                        tmp_sum = tmp_sum + right_vt
                    wt = wt - self.step_size * tmp_sum

                # update hessian approx, with last L w_t,
                if t % self.update_period == 0 and t != 0:
                    ur = np.mat(x_array.mean(axis=1))
                    idx = np.random.choice(n, self.bh, replace=False)
                    data_cnt = data_cnt + self.bh
                    data_x = self.train_x[:, idx]
                    data_y = self.train_y[:, idx]
                    hessian = self.get_hessian(wt, data_x, data_y, self.alpha)
                    sr = ur - last_u
                    yr = hessian*sr
                    s_array[:, r % self.store_num] = sr
                    y_array[:, r % self.store_num] = yr
                    last_u = ur
                    r = r + 1
                # update after using last L w_t, because ut not include this iteration w_t(old_w)
                x_array[:,t%self.update_period] = old_w

                # record the loss and epoch_cnt and time
                if data_cnt >= n:
                    epoch_cnt = epoch_cnt + int(data_cnt/n)
                    data_cnt = data_cnt % n
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
        Logger.log("---------SLBFGS end-------------")
        record.set_w(np.mat(w))
        record.get_best()
        return record
