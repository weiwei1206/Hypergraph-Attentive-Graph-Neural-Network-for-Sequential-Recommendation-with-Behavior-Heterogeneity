import numpy as np
import scipy.sparse as sp
import torch

from Params import args


def subgraph_construction(data, time_number, user_num, item_num, t_min):
    """
    :param data:  整体的大矩阵
    :param time_number:  时间段的数量
    :return:

    1.遍历大的adj的每一个元素
    2.连接对应的矩阵: 元素的data - 最小时间/ 时间缝隙
    """

    subgraphs = {}  #{t: dok}

    # print("\n")
    # print("时间个数",time_number)
    # print("\n")

    for t in range(0, time_number):  #创建空的子图矩阵: 这里的两个number肯定是要取最大的, 方便之后一起处理
        subgraphs[t] = {}

    for t in range(0, time_number):  #创建空的子图矩阵: 这里的两个number肯定是要取最大的, 方便之后一起处理
        subgraphs[t]['H'] = sp.dok_matrix((item_num, user_num), dtype=np.int)

    """
    根据原始矩阵, 构建time_number个矩阵
    目标: 按理来说, 这个代码块完成之后, 会有time_number个矩阵
    """
    data_coo = data.tocoo()
    for i in range(0, len(data_coo.data)):
        tmp_t = (data_coo.data[i] - t_min)/args.time_slot
        # print("类型转换之前的tmp_t:", tmp_t)
        tmp_t = tmp_t.astype(int)
        # print("\n")

        # print("子图内部的打印: ")
        # print("当前数据应该填到哪个时间矩阵里:", tmp_t)
        # print("\n")
        # print("应该变成行的列:", data_coo.col[i])
        # print("\n")
        # print("应该变成列的行:", data_coo.row[i])
        # print("\n")
        # print("时间缝隙:", args.time_slot)
        # print("\n")
        subgraphs[tmp_t]['H'][data_coo.col[i], data_coo.row[i]] = 1  #TODO: 字典遍历的时候超出范围

    """
    构建H和G
    """
    for t in range(0, time_number):
        subgraphs[t]['G'], subgraphs[t]['U'] = generate_G_from_H(subgraphs[t]['H'])
        subgraphs[t]['H'] = None

    return subgraphs  # 一个t, 三个矩阵'H','G','E'


def generate_G_from_H(H):
    """
    :param H:
    :return:
    目标: 对于每个H矩阵, 返回G, U以运算item, user的embedding
    """

    n_edge = H.shape[1]  #H[99037, 147894]    n_edge[147894]
    W = sp.diags(np.ones(n_edge))  #W[147894, 147894]
    DV = np.array(H.sum(1))
    DV2 = sp.diags(np.power(DV+1e-8, -0.5).flatten())  #D[99037, 99037] 为什么这个地方会报: 除以0: 因为求逆运算是一个除法的过程
    DE = np.array(H.sum(0))
    DE2 = sp.diags(np.power(DE+1e-8, -0.5).flatten())  #B[147894, 147894]
    HT = H.T

    #BHD XP   TODO: 这里是对应元素做乘法
    U = DE2 * HT * DV2  #*矩阵乘法   [147894, 147894]*[147894, 99037]*[99037, 99037] :TODO:其实这里就是用item的embedding更新user的embedding

    #DHWBHD XP
    G = DV2 * H * W * DE2 * HT * DV2  # [99037, 99037]*[99037, 147894]*[147894, 147894]*[147894, 147894]*[147894, 99037]*[99037, 99037]

    # DV2_H_W = DV2 * H * W
    # DE2_HT_DV2 = DE2 * HT * DV2
    
    
    return matrix_to_tensor(G), matrix_to_tensor(U)

# def sparse_matrix_to_torch_sparse_tensor(subgraphs_original, time_number, behaviors):  #这个函数可以废了
#
#     subgraphs = {}
#
#     for beh in range(len(behaviors)):
#         subgraphs[beh] = {}
#         for t in range(0, time_number):
#             subgraphs[beh][t] = {}
#             subgraphs[beh][t]['H'] = matrix_to_tensor(subgraphs_original[beh][t]['H'])
#             subgraphs[beh][t]['G'] = matrix_to_tensor(subgraphs_original[beh][t]['G'])
#             subgraphs[beh][t]['U'] = matrix_to_tensor(subgraphs_original[beh][t]['U'])
#             print("打印看看到底有什么自己处理了半天到底是什么数据结构: \n")
#             print(type(subgraphs[beh][t]['G']))
#
#     return subgraphs

def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  #环测师兄原来是这么写的.astype(np.float32)
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #创建坐标张量 从numpy.ndarray创建一个张量  有时候astype是为了整个模型的需要
    values = torch.from_numpy(cur_matrix.data)  #创建数据张量
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #tensor数据类型转换的函数需要, 记一下