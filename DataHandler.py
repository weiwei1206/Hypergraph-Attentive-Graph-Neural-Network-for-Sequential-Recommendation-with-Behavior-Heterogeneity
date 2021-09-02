import torch
import torch.utils.data as data
import pickle
import numpy as np
import scipy.sparse as sp
from math import ceil


from Params import args
import hypergraph_utils



class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):  #TODO:为什么倒数第二个必须要赋值
        #传入的就是 目标行为 的数据  train_data, self.item_num, self.trainMat, True??少了一个参数
        """
        TODO: 接下来要做的就是, 保证传入的就是 目标行为的数据: 目标行为处理的对, item_num:必须是行为数据集里item的数量(否则要是查不到就越界了)
        :param data:
        :param num_item:
        :param train_mat:
        :param num_ng:
        :param is_traing:
        """
        super(RecDataset, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        # self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  #test的时候没有必要采样  TODO: 这句话的语法不太懂
        dok_trainMat = self.train_mat.todok()  #将矩阵转为dok
        length = self.data.shape[0]  #第零维大小, 第零个数据大小??
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)  #相当于已经获得采样的数据,但是是否全是负还不一定

        for i in range(length):  #
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:  #就是说明原来的数据: 有0有1了
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
                self.neg_data[i] = iid  #加入负采样列表:   TODO:一个问题是:如果原始数据中负样本不够, 是不是就不能采到1V1的负样

    # def getMatrix(self):
    #     pass
    #
    # def getAdj(self):
    #     pass
    #
    # def sampleLargeGraph(self):
    #     """
    #     关于: 采样的问题: 只是在target上做采样
    #     """
    #
    #     def makeMask():
    #         pass
    #
    #     def updateBdgt():
    #         pass
    #
    #     def sample():
    #         pass
    #
    # def constructData(self):
    #     pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]

        if self.is_training:  #训练时: 需要正负样本对, 需要计算loss
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:  #测试时: 不需要负样本
            return user, item_i




