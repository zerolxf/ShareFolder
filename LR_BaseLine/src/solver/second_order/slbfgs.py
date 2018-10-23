import numpy as np
from solver.Solver import *
import time
from utils.Logger import *


class Slbfgs(Solver):
    # 对于所有实验, 我们设置批处理大小b的为20或者100, 我们设置Hessian的批处理大小b_H为10b或者20b, \
    # 设置Hessian更新间隔L为10, 设置存储大小M为10, 设置随机更新次数m为N / b.我们通过网格搜索来优化学习率
    def __init__(self, model, m, store_num, update_period,
                 b, bh, step_size):
        Solver.__init__(self, model, step_size)
        self.d = self.model.data_x.shape[0]
        self.n = self.model.data_x.shape[1]
        self.k = self.model.data_y.shape[0]
        self.store_num = store_num
        self.update_period = update_period
        self.b = b
        self.bh = bh
        self.step_size = step_size
        self.m = m
        self.name = "S-LBFGS"

    def get_params(self):
        params = {"store_num":self.store_num, "update_period":self.update_period}
        params["b"] = self.b
        params["bh"]  = self.bh
        params["step_size"] = self.step_size
        params["m"] = self.m
        return params

    def run(self, max_epoch_num):
        m = int(self.m * self.n)
        w = np.mat(np.random.rand(self.d, self.k))
        epoch_cnt: int = 0      # count the number of epoch
        t = 0           # count the numbers of iteration
        start = time.time()
        self.print_params()
        record = Record([], [], [])
        r: int = 0
        data_cnt: int = 0
        Logger.log_start(self.name)
        s_array = np.mat(np.zeros((self.d*self.k, self.store_num)))
        y_array = np.mat(np.zeros((self.d*self.k, self.store_num)))
        x_array = np.mat(np.zeros((self.d*self.k, self.update_period)))
        last_u = vec_transpose(w, self.d)
        while epoch_cnt < max_epoch_num:
            # y_array, s_array record last M (sr, yr), last idx = (r - 1)%M
            u = self.get_full_gradient(w)
            epoch_cnt = epoch_cnt + 1
            wt = w
            for i in range(m):
                t = t + 1
                idx = np.random.choice(self.n, self.b, replace=False)
                data_cnt = data_cnt + self.b
                delta = self.get_gradient(wt, idx, self.b) - self.get_gradient(w, idx, self.b)
                delta = vec_transpose(delta + u, self.d)
                old_w = vec_transpose(wt, self.d)
                if r == 0:
                    wt = wt - self.step_size*vec_transpose(delta, self.d)
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
                    wt = wt - self.step_size * vec_transpose(tmp_sum, self.d)

                # update hessian approx, with last L w_t,
                if t % self.update_period == 0 and t != 0:
                    ur = np.mat(x_array.mean(axis=1))
                    idx = np.random.choice(self.n, self.bh, replace=False)
                    data_cnt = data_cnt + self.bh
                    sr = ur - last_u
                    # hessian = self.get_hessian(wt, idx, self.bh)
                    # yr = hessian*sr
                    yr = self.get_hessian_vector_product(wt, idx, self.bh, vec_transpose(sr, self.d))
                    s_array[:, r % self.store_num] = sr
                    y_array[:, r % self.store_num] = vec_transpose(yr, self.d)
                    # y_array[:, r % self.store_num] = yr
                    last_u = ur
                    r = r + 1
                # update after using last L w_t, because ut not include this iteration w_t(old_w)
                x_array[:,t%self.update_period] = old_w

                # record the loss and epoch_cnt and time
                if data_cnt >= self.n:
                    epoch_cnt = epoch_cnt + int(data_cnt/self.n)
                    data_cnt = data_cnt % self.n
                    start_loss_time = time.time()
                    loss = self.get_loss(wt)
                    end_loss_time = time.time()
                    start = start + end_loss_time - start_loss_time
                    now = time.time() - start
                    Logger.log("iterNum :"+str(epoch_cnt)+" time:"
                               + str(now) + "loss:"+str(loss))
                    record.append(epoch_cnt, now, loss)
                    if epoch_cnt >= max_epoch_num:
                        break
            w = wt
        Logger.log_end(self.name)
        record.set_w(np.mat(w))
        record.get_best()
        return record
