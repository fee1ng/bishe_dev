import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from concurrent.futures._base import as_completed, wait

class LFM(object):
    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10,
                 columns=["uid", "iid", "rating"]):
        # 学习率
        self.alpha = alpha
        # P 的lambda
        self.reg_p = reg_p
        # q 的lambda
        self.reg_q = reg_q
        # 矩阵分解隐式类别数量
        self.number_LatentFactors = number_LatentFactors
        # 迭代次数
        self.number_epochs = number_epochs
        self.columns = columns

    def fit(self, dataset):
        # 加载数据
        self.dataset = pd.DataFrame(dataset)
        # 用户评分数据集
        self.user_ratings = dataset.groupby('userId').agg([list])
        # 电影评分数据集
        self.item_ratings = dataset.groupby('movieId').agg([list])
        # 平均评分
        self.global_mean = self.dataset['rating'].mean()
        # 计算的P和Q
        self.P, self.Q = self.sgd()

    def init_matrix(self):
        # 初始化P 和 Q, P

        P = dict(zip(self.user_ratings.index,
                     np.random.rand(len(self.user_ratings), self.number_LatentFactors).astype(np.float32)))
        Q = dict(zip(self.item_ratings.index,
                     np.random.rand(len(self.item_ratings), self.number_LatentFactors).astype(np.float32)))
        return P, Q

    def sgd(self):
        # 进行矩阵拆解
        # 初始化P和Q
        P, Q = self.init_matrix()
        # 迭代更新 P 和 Q 矩阵
        for i in range(self.number_epochs):
            print("iter%d" % i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                # 用户
                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(r_ui - np.dot(v_pu, v_qi))
                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)
                # 更新P Q
                Q[uid] = v_pu
                P[iid] = v_qi
            error_list.append(err ** 2)
        return P, Q

    #def predict(self, uid, iid):
    def predict(self,uid,iid):
        # 预测用户对物品的评分
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.user_ratings.index or iid not in self.item_ratings.index:
            #return self.globalMean
            return self.global_mean
        p_u = self.Q[uid]
        q_i = self.Q[iid]
        return np.dot(p_u, q_i)

    def test(self, testset):
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as  e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


def main():
    print("Working")
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("/usr/ratings.csv", usecols=range(3), dtype=dict(dtype))

    #TODO bug!!!
    #uid = (float)(input("input uid:"))
    #iid = (float)(input("input iid:"))
    uid = 1
    iid = 1
    parm_list = []
    parm_list.append(uid)
    parm_list.append(iid)
    print(parm_list)

    lfm = LFM(0.02, 0.01, 0.01, 10, 20, ["userId", "movieId", "rating"])
    lfm.fit(dataset)

    #线程池多线程处理n个预测
    executor = ThreadPoolExecutor(max_workers=5)
    # 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
    predict_rating_list = []
    #predict_rating = executor.submit(lfm.predict, parm_list)
    predict_rating = executor.submit(lambda p: lfm.predict(*p),parm_list)
    predict_rating_list.append(predict_rating)

    for future in as_completed(predict_rating_list):
        data = future.result()
        print(data)

main()

    #predict_rating = lfm.predict(1, 1)
    #predict_rating = lfm.predict(uid,iid)
    #print(predict_rating)