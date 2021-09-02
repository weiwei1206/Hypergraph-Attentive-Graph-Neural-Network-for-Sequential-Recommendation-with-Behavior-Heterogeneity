import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime

import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F

import hypergraph_utils
import DataHandler


import myModel
import myModel_NoGNN
import myModel_NoGNN_tmall
import myModel_NoGNN_mean
import myModel_RNN
import myModel_mean
import myModel_mean_tmall
import myModel_ori
import myModel_multi_head

from Params import args
from Utils.TimeLogger import log
import evaluate

if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


# modelTime = str(int(time.time()))[4:]  #TODO: 因为这个所以会过一会儿保存一下
now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')



class Model():
    def __init__(self):
        """
        1.准备dataset的数据: 包括训练集和测试集(但是, 以我现在的能力, 先训练集, 再测试集, 再多行为)
            (1)加载数据: 获得矩阵(H,G,E)并转为tensor, 获得itemNum, userNum
            (2)获得训练数据非零的最大最小值(按找师兄给的这个大矩阵, 确实要处理): 用以制作dataset, dataloader
        2.准备要运算的矩阵
        3.准备各种其它参数, 随机种子, 评价指标
        """

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'     #6, 8, 11, 15, 58
            
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_13' 
        #Tmall: 3,4,5,6,8,59
        #IJCAI_15: 5,6,8,10,13,53


        self.t_max = -1 # 处理后数据集的最大时间, 最小时间

        self.t_min = 0x7FFFFFFF
        self.time_number = -1 # 时间段数
        # self.matrixs  # 不加后面的看看会不会出错
        self.user_num = -1
        self.item_num = -1
        self.subgraphs = {}  #不知道字典可不可以以这种方式初始化
        self.behaviors = []#到底为什么要加self?  因为东西初始化的时候, 只在init里面.  如果想要:在这个类的其它函数中也使用, 必须要用它
        self.behaviors_data = {}
        # self.trainMat = csr_matrix()
        # self.train_loader
        # self.test_loader

        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        #
        self.curEpoch = 0
        # self.isLoadModel = args.isload

        if args.dataset == ('Tmall' or 'Tmall_LH'):
            # self.behaviors = ['buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']  #TODO: 暂时把pv去掉了, 因为数据还没处理好
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'JD':
            self.behaviors = ['review','browse', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'retailrocket':
            self.behaviors = ['view','cart', 'buy']
            
        # data_loader 到底组织在代码的哪个地方: 在model init的地方, 在train_model的地方调用
        #加载每个行为的数据   #因为这里的大小不确定: 所以, 是不是需要一个mask, 然后embedding的padding的时候使用
        #加载训练数据
        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:  # r是读取人工数据, rb是读取二进制文件
                data = pickle.load(fs)
                self.behaviors_data[i] = data  # TODO: 这样是不是对内存的要求比较高.  是先存到数据结构当中, 还是到时候再读一遍

                if data.get_shape()[0] > self.user_num:  # TODO: 先按最大的矩阵来, 如果效果不好是不是有pad
                    self.user_num = data.get_shape()[0]  # TODO: 但是, 这样就存在一个问题: 某个时段,会学出一张整表: 这张表中的某个user或者item是不存在的
                if data.get_shape()[1] > self.item_num:  # TODO: 配合运算的矩阵也变成相同大小, 是可以运算.
                    self.item_num = data.get_shape()[1]  # TODO: 本来没有的embedding也会初始化

                # value = data.data  #得到数据
                # row, col = data.nonzero()  # TODO: ????百度也没查到为什么会返回矩阵的行和列

                # """
                # 处理掉出问题的数据:   为什么把这一段注释掉??? 因为我不知道处理过后的行列各是多少?? 而且buy的数据好像没有出错
                # """
                # dataset_t_min=1511539200  #获得最早和最晚时间戳
                # dataset_t_max=1512316799  #13位时间戳的单位是毫秒, 10位时间戳的单位为秒
                #
                # new_row = []
                # new_col = []
                # new_value = []
                # for i in range(value.size):  #处理掉数据集中的错误
                #     cur_value = value[i]
                #     if cur_value > dataset_t_min and cur_value < dataset_t_max:
                #         new_row.append(row[i])
                #         new_col.append(col[i])
                #         new_value.append(cur_value)

                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                # original_matrix = sp.csr_matrix(new_value, (new_row, new_col), shape=(len(new_row), len(new_col)))  #csr矩阵的另一种初始化方式
                # ========>至此得到处理数据后的矩阵
                if self.behaviors[i]==args.target:
                    self.trainMat = data
                    self.trainLabel = 1*(self.trainMat != 0)  #TODO: 看看没用的数据到底是0还是, None
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  #TODO: 这里为什么要squeeze: 因为做了reduce_sum之后维度会变为1, 所以要把变为1维的那个维度消掉


            tmp_time_number = (self.t_max - self.t_min) / args.time_slot + 1  #TODO: 原来一直没算对, 现在看看是什么情况
            tmp_time_number = tmp_time_number.astype(int)
            # print("\n")
            # print("每次时间缝隙的数量:", tmp_time_number)
            # print("\n")
            # print("最大的时间:", self.t_max)
            # print("\n")
            # print("最小的时间:", self.t_min)
            # print("\n")
            # print("时间缝隙:", args.time_slot)
            # print("\n")

            if tmp_time_number > self.time_number:
                self.time_number = tmp_time_number  # 浮点数向上取整 TODO:做到了吗???

            print("打印看看有多少个时间段: ", self.time_number)
            print("\n")


        time = datetime.datetime.now()
        print("开始构建子图:  ", time)

        # 子图构建
        for i in range(0, len(self.behaviors)):
            beh_subgraphs = hypergraph_utils.subgraph_construction(self.behaviors_data[i], self.time_number, self.user_num, self.item_num, self.t_min)
            self.subgraphs[i] = beh_subgraphs  # TODO: 上面的代码段 要保证得到这个数据结构

        time = datetime.datetime.now()
        print("结束构建子图:  ", time)

        # 将子图由csr转为tensor
        # self.subgraphs = hypergraph_utils.sparse_matrix_to_torch_sparse_tensor(self.subgraphs, self.time_number, self.behaviors)  #因为这里是一个dict所以没有cuda  看看都有哪些变量需要移到.cuda()上
        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num)
        print("\n")


        #--------------------------------------------------------至此: 获得多个时间段下的矩阵------------------------------------------>>>>>

        train_u, train_v = self.trainMat.nonzero()
        # assert np.sum(self.trainMat.data ==0) == 0
        # log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset(train_data, self.item_num, self.trainMat, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

        #valid_data


        # test_data  TODO: 直接把传入的数据做成ui对就可以了
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        # testbatch = np.maximum(1, args.batch * args.sampNum // 100)  # TODO: 改变batch的大小----比train_batch大.  至少是1的大小(dataloader的参数)
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  #TODO: train的data_loader和test的data_loader不一样
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)  #如果test的时候, 这个batch出了什么错误


        # --------------------------------------------------------至此: 构建出dataset----------------------------------------------->>>>>


        # #准备好各个时间段的空矩阵, 以备之后赋值
        # for i in range(0, self.time_number):  # TODO: 确认这里不会到time_number, 不会使矩阵的数量多一个, 少一个
        #     tmp_mat = sp.dok_matrix(user_num, item_num)
        #     self.matrixs.appened(tmp_mat)
        #
        # for i in range(len(original_matrix)):
        #     # if mat_time > self.t_min and mat_time < self.t_max:
        #     matrixs[original_matrix.data[i]/agrs.time_slot][][]

    def prepareModel(self):
        self.modelName = self.getModelName()  #这句话只有在存储模型或者直接加载模型的时候才能用到
        self.setRandomSeed()
        self.gnn_layer = eval(args.gnn_layer)  #解析并执行字符串, 单引号和双引号解析为int, 三引号解析为str
        self.hidden_dim = args.hidden_dim

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            # self.model = myModel.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs) 
            self.model = myModel_ori.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            #myModel_multi_head
            # self.model = myModel_multi_head.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            # self.model = myModel_RNN.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            # self.model = myModel_mean.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            # self.model = myModel_mean_tmall.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)  
            # myModel_NoGNN_mean
            # self.model = myModel_NoGNN_mean.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            # self.model = myModel_NoGNN.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)   
            # myModel_NoGNN_tmall
            # self.model = myModel_NoGNN_tmall.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)
            # self.model = myModel.myModel_NoGNN_mean(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)


        self.opt = t.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay = 0)
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[60,75,90,105,130,150,180], gamma=0.98)

        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):  #user_embeding 和 item_embedding 的时候使用内积.  而且是和正负样本相关
        """
        正,负embedding分别和user做内积, 做两个内积的结果
        """
        pred_i = t.sum(t.mul(u,i), dim=1)  #*args.mult  #因为不知道是 极小还是极大  所以, 这里先往大变
        pred_j = t.sum(t.mul(u,j), dim=1)  #*args.mult
        return pred_i, pred_j

    def run(self):
        #TODO: 向师兄看齐(已完成: 这部分的架构应该和师兄一样)
        # 1.准备模型
        # 2.(加载和保存模型, 之后有机会再做)
        # 3.循环进行每个epoch的训练:
        #       (1)设置多久test一次
        #       (2)训练模型 + 打印
        #       (3)如果需要test模型: test + 打印
        #       (4)(如果需要保存模型.  保存)
        # 4.整体test模型 + 打印
        # 5.(整体保存模型)

        self.prepareModel()
        if args.isload == True:
            # self.loadModel(args.loadModelPath)
            print("----------------------pre test:")
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}")    #以f开头的字符串, {}中间的值会被替换掉

        log('Model Prepared')


        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        print("Test before train:")
        HR, NDCG = self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch+1):  
            test = (e % args.tstEpoch == 0)  
            self.curEpoch = e
            log("*****************开始第%d个epoch的训练 ************************"%e)  

            if args.isJustTest == False:
                epoch_loss = self.trainEpoch()
                self.train_loss.append(epoch_loss)  
                log("epoch %d/%d, epoch_loss=%.2f"% (e, args.epoch, epoch_loss))
                self.train_loss.append(epoch_loss)
                # self.scheduler.step()  #-----------------------------------------------需要注释掉的
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            

            if HR > self.best_HR:
                self.saveHistory()
                self.saveModel()
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
            
            
            if NDCG > self.best_NDCG:
                self.saveHistory()
                self.saveModel()
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch 
                cvWait = 0
                # print("--------------------------------------------------------------------------------------------------------------------------HR", self.best_HR)
                print("--------------------------------------------------------------------------------------------------------------------------best_NDCG", self.best_NDCG)
            

            if (HR<self.best_HR) and (NDCG<self.best_NDCG): 
                cvWait += 1


            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                # self.loadModel(self.modelName)
                self.saveHistory()
                self.saveModel()
                break


                # print(f"epoch{e}/{args.epoch}, test HR = {HR}, valid NDCG = {NDCG}")  TODO:深深地怀疑这个用法不能打印函数

                # print("epoch %d/%d, test HR = %.4f, valid NDCG = %.4f"%(e, \
                #                                                         args.epoch, \
                #                                                         HR, \
                #                                                         NDCG))  #TODO:这段打印报了很多错
                # log("epoch %d/%d, test HR = %.4f, valid NDCG = %.4f" % (e, args.epoch, HR, NDCG))


            # 调节学习率
            # self.adjust_learning_rate(self.opt, e)

            # 提前终止  TODO: 自己再搞一下
            # if HR > best_HR:
            #     best_HR = HR
            #     cvWait = 0
            #     best_epoch = self.curEpoch
            #     self.saveModel()
            # else:
            #     cvWait += 1
            #     log("cvWait = %d"%(cvWait))
            # 
            # self.saveHistory()
            #
            #
            
        # 和上面的test不同的是: 这个是最后的测试. 而上面的是在每个epoch进行的过程中 每三个一次, 免得自己等太久看不到结果
        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)
        # self.saveHistory()
        # self.saveModel()


        # log("epoch %d/%d, test HR = %.4f, valid NDCG = %.4f" % (e, args.epoch, HR, NDCG))

    def trainEpoch(self):
        """
        1.将groundtruth送进来
        2.通过model, 获得所有的embedding
        3.计算各种loss, 反向传播, 更新
        :return:
        """
        train_loader = self.train_loader

        # log("start negative sample...")
        train_loader.dataset.ng_sample()
        # log("finish negative sample...")

        epoch_loss = 0
        cnt = 0
        for user, item_i, item_j in train_loader:  #TODO:train_loader----理解成一个迭代器,每次返回要训练的数据.
            # time = datetime.datetime.now()
            # print("step_time_start:  ", time)

            # if cnt==1:
                # break

            """
            上面三个量的数据类型都是: tensor
            上面三个量的数据大小都是: 32, 即跟着参数batch走
            """
            # TODO: 即, 现在是将所有的train数据 或者 test数据都送进dataloader. 我们要做的是要将什么送进dataloader
            # print("\n")  #因为我这里要看看data_loader和batch_size的关系
            # print("看看user的数据类型", type(user))
            # print("看看user的大小", user.shape)
            # print("看看item_i的数据类型", type(item_i))
            # print("看看item_i的大小", item_i.shape)
            # print("看看item_j的数据类型", type(item_j))
            # print("看看item_j的大小", item_j.shape)
            # print("cnt: ", cnt)
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()

            # print("***********************************************************************")
            # print("***********************************************************************")
            # print("******************************cnt: %d ************************************" % cnt)
            # print("***********************************************************************")
            # print("***********************************************************************")
            # print(self.model.state_dict())

            user_embed, item_embed = self.model(self.subgraphs)

            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]

            pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()  #TODO:  因为loss为nan才设置的 bpr_loss只是正负样本对的预测结果, 不涉及ground_truth  TODO:可以用mean做
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5 * (bprloss + args.reg * regLoss) / args.batch
            epoch_loss = epoch_loss + bprloss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # time = datetime.datetime.now()
            # print("step_time_end: ", time)

            cnt+=1
            # if cnt%20 == 0:  #TODO:一共642916条数据, 每次取batch=32来, 一个epoch确实需要训练很长时间
            #     print("cnt: %d  step_loss: %f \n" % (cnt, loss.item()))  # python打印输出的语法: 如果要打印的话, 用%连接
            log('step %d, step_loss = %f'%(cnt, loss.item()), save=False, oneline=True)  #log打印不了
        log("finish train")
        return epoch_loss

    def testEpoch(self, data_loader, save=False):
        # load test dataset
        # HR, NDCG = self.validModel(self.test_loader)
        # log("test HR = %.4f, test NDCG = %.4f" % (HR, NDCG))
        # log("model name : %s" % (self.modelName))
        # epochHit, epochNdcg = [0] * 2
        epochHR, epochNDCG = [0]*2
        user_embed, item_embed = self.model(self. subgraphs)

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)  #TODO: 这里.cuda()会报错, 可能是函数返回的东西不是tensor. 但是感觉不是cuda会有问题, 之后再移动.
            userEmbed = user_embed[user_compute]  #[614400, 16], [147894, 16]
            # bug = [ (index, value)  for index, value in enumerate(item_compute)  if (value>=22734 or value<0)]
            # print("**************************bug:  ", bug)
            # print("**************************max(item_compute):  ", max(item_compute))
            # print("**************************min(item_compute):  ", min(item_compute))
            # print("**************************item_embed: ", len(item_embed))

            itemEmbed = item_embed[item_compute]
            # itemEmbed = item_embed[ \
            #     item_compute \
            #     ]  #[], []
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)  #原来是*点乘, 维度相同, 按一行合成一个数: 就是预测的分数

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)  #TODO: 是按照自己的想法写的, 非常有可能报错
            epochHR = epochHR + hit  #TODO: 看看这里到底是一个列表, 还是加法
            epochNDCG = epochNDCG + ndcg  #
            # print(f"Step {cnt}:  hit:{hit}, ndcg:{ndcg}")
            cnt += 1 #TODO: 可以看看像这种情况会在哪些地方报错
            tot += user.shape[0]


        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG
    # def validModel(self, data_loader, save=False):
    #     HR, NDCG = [], []
    #     user_embed, item_embed = self.model(self.subgraphs)
    #
    #     for user, item_i in data_loader:
    #         user = user.long().cuda()  #TODO: 问问环测师兄为什么这里要用long()
    #         item_i = item_i.long().cuda()
    #
    #         userEmbed = user_embed[user]
    #         testItemEmbed = item_embed[item_i]
    #         pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)  #还是所有的embedding一起算
    #
    #         #TODO: 按照每个batch计算 评价指标
    #         batch = int(user.cpu().numpy().size / 101)
    #         assert user.cpu().numpy().size % 101 == 0
    #         for i in range(batch):  # TODO: 问题来了, 为什么哪里都会有一个101??
    #             batch_scores = pred_i[i * 101: (i + 1) * 101].view(-1)
    #             _, indices = t.topk(batch_scores, self.args.top_k)
    #             tmp_item_i = item_i[i * 101: (i + 1) * 101]
    #             recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
    #             gt_item = tmp_item_i[0].item()
    #             HR.append(evaluate.hit(gt_item, recommends))
    #             NDCG.append(evaluate.ndcg(gt_item, recommends))
    #
    #     return np.mean(HR), np.mean(NDCG)
    def sampleTestBatch(self, batch_user_id, batch_item_id):
        #TODO:
        # 输入: 一个batch的user_id
        # 需求:
        # (1) [batch, 1]ground_truth------其实外层函数是有的, 直接传进来就好
        # (2) buy邻接矩阵------self.应该是有的, 直接用就好
        # (3) []
        # (4) []
        # [batch, item_num]可以负采样的------从矩阵中取出的一条, 有值为:false, 无值为:true
        # 输出: batch*100个user_id, batch*100个item_id, [batch, 1]ground_truth, [batch, 100]
        batch = len(batch_user_id)
        tmplen = (batch*100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()  #取到sub矩阵
        user_item1 = batch_item_id  #得到每个user对应的正例
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]  #得到每个user对应的正例
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])  #TODO:得到一个sub矩阵, 并且把负例挑出来------!!!这里按理来说应该得到所有负的id.  不定长,所有负样本的id做成一列
            pvec = self.labelP[negset]  #取到未交互的item的频数 
            pvec = pvec / np.sum(pvec)  #对所有取到的频数做归一化
            # random_neg_sam = np.random.choice(negset, 99, replace=False, p=pvec)  #在negset里面取, 取99个, 以pvec的概率
            
            random_neg_sam = np.random.permutation(negset)[:99]  #随机打乱, 取前99个: 如果报错你的包有问题, 很有可能是你自己的拼写错误
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item])))  #TODO: np.concatenate 这个接口一次性完成数组的拼接. 但是, 有一个很神奇的事情让我debug了半天: 第一个参数自己就要加括号
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100
# 参数一: batch*100即每个user重复100遍[batch*100] 参数二:随机采样的99个负例item+正例item[batch*100]
# 参数三: 正例对应的item_id[batch]                参数四:随机采样+负例[batch, 100]
    def calcRes(self, pred_i, user_item1, user_item100):  #[6144, 100] [6144] [6144, (ndarray:(100,))]
        #TODO:
        # 1.遍历: 每一个预测结果
        #       (1)
        #       (2)
        #       (3)
        #       (4)

        hit = 0
        ndcg = 0

        # print("等一下: 这个函数你进来了吗?")

        for j in range(pred_i.shape[0]):
            # predvals = list(zip(pred_i[j], user_item100[j]))  #100:100预测分数和id放在一起
            # predvals.sort(key=lambda x:x[0], reverse=True)  #按分数降序排列
            # shoot = list(map(lambda x:x[1], predvals[:args.shoot]))  #取预测分数的前shoot个的item_id

            _, shoot_index = t.topk(pred_i[j], args.shoot) 
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            # if user_item1[j] in shoot:  #如果正样本的id在前k个
            #     hit += 1  
            #     ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  

            if type(shoot)!=int and (user_item1[j] in shoot):  #如果正样本的id在前k个
                hit += 1  
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  
            elif type(shoot)==int and (user_item1[j] == shoot):
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
       

        return hit, ndcg  #int, float

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  #TODO: 再确认一下自己的参数名都是否正确
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_time_slot_" + str(args.time_slot) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    #TODO: 需要的东西主要从history中加载
    def saveHistory(self):  #获得训练到某一个epoch为止的各种loss和指标, 存储到某个文件中
        #要画图的东西存到history里面
        history = dict()
        history['loss'] = self.train_loss  #获得训练到某一个epoch为止的loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:  # r代表不转义 TODO: 这个路径还要再确认一下吧, 主要是这个ModelName的问题
            pickle.dump(history, fs)


    def saveModel(self):  #这个模型没什么, 就是保存函数. TODO: 主要思考这里保存什么函数
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.model,
            # 'reg': self.reg,
            'history': history,
        }
        t.save(params, savePath)

    def loadModel(self, loadPath):     #第二个参数: 
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1
        # self.lr = checkpoint['lr']
        # self.args.reg = checkpoint['reg']
        # # 恢复history
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))



if __name__ == '__main__':
    # TODO:   follow师兄的实验设置, 将模型向和大家一个水平调整
    #      -3.把embedding写成para和, 把embedding写成embedding是不一样
    #      -2.初始化只有一次: 所有weight初始的值只有一次
    #      -1.到底什么是一个模型需要训练的参数:  (1)写进去的weights    (2)embedding算不算???:  embedding到底是自己计算更新, 还是和模型一起会计算
    #      0.查看自己的参数有没有更新:  因为, 现在明显感觉, 自己的loss只是通过embedding计算的. 所以,即感觉参数好像没有更新???
    #      1.将run的过程, 调整成和师兄一样
    #      2.确定, 自己的dataloader到底要如何组织
    #      3.理解大图采样
    #      4.理解batch采样
    #      5.理解测评----最终目的, 知道模型要向着一个好的结果, 中间的计算过程是什么, 会有哪些影响因素
    #      6.查看师兄的代码, 看看有哪些巧妙的做法, 能够使得模型结果上升.
    print(args)
    my_model = Model()  #为了偷懒想将整个训练的模型移到GPU上, 看看这样做会不会报错: 不是pytorch框架里的东西是没有cuda()的
    my_model.run()
    # my_model.test()

