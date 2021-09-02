import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from Params import args


class dot_attention(nn.Module):

    def __init__(self):
            super(dot_attention, self).__init__()
            self.dropout = nn.Dropout(args.drop_rate)
            self.softmax = nn.Softmax(dim=2)   #-----------------??

    def forward(self, q, k, v, scale=None, attn_mask=None):  #[batch*head_num, beh, h_d]
        
            attention = torch.bmm(q, k.transpose(1, 2))  #-----------------  [-1, beh, dim]*[-1, dim, beh]
            if scale:
                attention = attention * scale        
            # if attn_mask:
                # attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
            
            # 计算softmax
            attention = self.softmax(attention)  #[-1, beh, beh]
            # 添加dropout
            attention = self.dropout(attention)        
            # 和v做点积。
            context = torch.bmm(attention, v)  #[-1,  beh, h_d]
            return context, attention


class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim=args.hidden_dim, num_heads=args.head_num, dropout=args.drop_rate):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads   # 每个头的维度
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention()

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm 归一化。

    def forward(self, key, value, query, attn_mask=None):  #[beh, N, d]=[len, batch_size, dim]
        
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 线性映射。
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 按照头进行分割
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)  #[batch*head_num, beh, h_d]
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 缩放点击注意力机制
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)  ##[-1, beh, h_d]  [-1, beh, beh]

        # 进行头合并 concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)  #[N, beh, dim]

        # 进行线性映射
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 添加残差层和正则化层。
        output = self.layer_norm(residual + output)
        output = output.transpose(0, 1)  #[beh, N, dim]

        return output, attention



class myModel(nn.Module):
    def __init__(self, userNum, itemNum, time_number, behavior, subgraphs):  #TODO: 这里的参数在调用处要解决
        super(myModel, self).__init__()  #自定义的类下面必须写这个
        """
        1.准备dataset的数据: 包括训练集和测试集(但是, 以我现在的能力, 先训练集, 再测试集, 再多行为)
        2.准备要运算的矩阵
        3.准备各种其它参数, 随机种子, 评价指标
        """

        """
        先说说自己能够想得到的初始化需要的参数:
        1. 时间段的个数
        2. 模型连接在一起的时候需要的参数: ???用什么机制连接???
        3. 各个时间段下的embedding
        4. 各个时间段下的embedding的初始化, 第一个自然是用一般的初始化, 而下一个用的是上一个做初始化
        5. 我知道了, 关于多个模型的训练: 应该是整体学出一个, embedding一趟更新, 而不是每个GNN做一次
        6. 关于"各个时间段下如何组织":  TODO: 要和良昊师兄讨论
        7. 每个时间段下的邻接矩阵送入 各个模型训练. 
        8. 对于dataset, 其实取一个就好: 因为还是最终的embedding查表
        9. 将多个embedding集合成一个embedding的任务是在模型里面完成的, 最终只是需要送出去embedding的表
        """

        self.userNum = userNum
        self.itemNum = itemNum
        self.time_number = time_number
        self.behavior = behavior
        self.subgraphs = subgraphs
        # self.static_embedding  #需要初始化的函数还是需要写出来的
        self.embedding_dict = self.init_embedding() #有没有embedding_dict再说 = nn.EmbeddingDict()
        self.weight_dict = self.init_weight()
        self.hgnns = self.init_hgnns() #nn.ModuleDict()  !!!!!如果想把某个"模型模块"或者"参数"注册到整个模块中, 则必须"直接是!!!"ModuleList()或者ModuleDict(). 不能是二阶的.
        # self.act = torch.nn.PReLU()
        self.act = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.layer_norm = torch.nn.LayerNorm(args.hidden_dim, eps=1e-8)
        # self.attention_dict = self.init_attention()
        self.self_attention_net = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim*2),
            nn.Dropout(args.drop_rate),
            nn.PReLU(),
            nn.Linear(args.hidden_dim*2, args.hidden_dim),
            nn.Dropout(args.drop_rate),
            nn.PReLU()
        )
        #init embedding
        """
        1.初始化: 一开始的embedding: 送入每个模型之后, 就可以得到不同的embedding
        2.将从每个模型得到的embedding存起来: (1)整体更新使用 (2)训练查表的时候使用
        3.初始化模型
        4.
        """

        # t.nn.Embedding(shape)
        # t.nn.Paramter(tensor)

        # modelDict = torch.nn.ModuleDict

        # for t in range(0, time_number):
        #     for beh in ...:
        #         modelDict[''] = HGNN(...)
        #     pass   #TODO: 模型的传入肯定是模型的输入
        # #整个大的模型传入: 按时间序列分好的数据

    def init_embedding(self):
 
        times_user_embedding = {}
        times_item_embedding = {}
        for t in range(0, self.time_number):
            times_user_embedding[t] = {}
            times_item_embedding[t] = {}
  
        embedding_dict = {  #nn.ParameterDict():必须是列表
    #         'static_user_embedding': static_user_embedding,
    #         'static_item_embedding': static_item_embedding,
            'times_user_embedding': times_user_embedding,
            'times_item_embedding': times_item_embedding,
        }

        return embedding_dict

    # def init_attention():
    #     attention_dict = nn.ParameterDict({
    #         'attention': 
    #         'self_attention':
    #         'multi_head_self_attention':
    #     })
    #     return attention_dict

    def init_weight(self):  #TODO: embedding和parameter有什么区别

        #TODO: 这个层面weight要做的事: 其实就是整个模型的weight:
        #     1.所有QKV: 所有的QKV只是用一个W:[d,d]
        #     2.K:  t-1个时段的
        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_k': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_v': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_d_d': nn.Parameter(initializer(torch.empty([args.hidden_dim, 1]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })

        # weight_dict = nn.ParameterDict({
        #     'qkv_weight': nn.Parameter(initializer(torch.empty(args.hidden_dim, args.hidden_dim))),
        # })
        # for t in range(0, (self.time_number-1)):
        #     self.weight_list[t] = nn.Parameter(initializer(torch.empty([len(self.behavior), args.hidden_dim, args.hidden_dim])))
        return weight_dict  #目前为止, 全局只有一个weight

    def init_hgnns(self):
        hgnns = nn.ModuleList()
        for t in range(0, self.time_number):
            # hgnns[t] = nn.ModuleDict()
            hgnns.append(nn.ModuleDict())
            for beh in self.behavior:
                hgnns[t][beh] = HGNN(self.userNum, self.itemNum)

        #降低一个模型的参数赋值给所有的模型
        weights = hgnns[0][self.behavior[0]].state_dict()
        for t in range(0, self.time_number):
            for beh in self.behavior:
                hgnns[t][beh].load_state_dict(weights)


        return hgnns

    def init_attention(self):
        """
        初始化attention的模型
        :return:
        """
        pass

    def forward(self, subgraphs):

        #希望得到: 每个时间段的各种行为的embedding
        for t in range(0, self.time_number):
            # all_embeddings[t] = {}
            for i, beh in enumerate(self.behavior):
                model = self.hgnns[t][beh]
                # all_embeddings[t][beh]['user'], all_embeddings[t][beh]['item'] = model(subgraphs[beh][t]['G']['u'], self.embedding_dict[t][beh]['user'], self.embedding_dict[t][beh]['item'])
                if t == 0:
                    self.embedding_dict['times_item_embedding'][t][beh], self.embedding_dict['times_user_embedding'][t][beh] = model(self.subgraphs[i][t]['G'], self.subgraphs[i][t]['U'] , model.item_embedding.weight, model.user_embedding.weight)  #如果是第一个时间段就把原始的embedding传进去
                else:
                    self.embedding_dict['times_item_embedding'][t][beh], self.embedding_dict['times_user_embedding'][t][beh] = model(self.subgraphs[i][t]['G'], self.subgraphs[i][t]['U'], self.embedding_dict['times_item_embedding'][t-1][beh], self.embedding_dict['times_user_embedding'][t-1][beh])

              

        #TODO: 处理多个时段的embedding的关系: 即, 每个时段embedding的更新
        #  1.如果是第一个时段, 只是自己
        #  2.如果是之后的时段:
        #       (0).通过HGNN得到的embedding: []
        #       (1).得到QKV:
        #               1)乘以W, 得到QKV: [beh, N, d].[d, d] == [beh, N, d]------问题:每个时段一个W, 还是多个时段共用一个W, 还有K:t-1------[beh,N,d][d,d]==>[beh*N,d][d,d]==>做完矩阵乘法后[beh*N, d]==>再将维度还原回来[beh,N,d]
        #               2)得到每个时段的Q: 将维度[beh, N, d]扩展为[beh, 1, N, d], 即在第二个维度扩展------想要[beh, beh]必须Q[beh, 1]------如果填充到[第一维]是广播机制, 那么填充到[第二维]是什么机制
        #       (2).K(参数):是每个时段各有一个, 还是大家共用一个K?
        #               1)正常来说每个时段一个[beh, N, d]: 不不, 我先选择一个, 可以吹memory机制, 多了会过拟合
        #               2)乘以W: [beh, N, d]*[beh, d, d] ==> [beh, N, d]
        #               3)扩展维度: [1, beh, N, d]------想要[beh, beh]必须Q[1,beh]即,相当于[beh,1][1,beh]专置相乘
        #       (3).得到每个时段的att:
        #               0)[beh, 1, N, d]*[1, beh, N, d]  ====  "这一步就相当于: Q*K^T专置相乘"------高维矩阵的运算法则不是: 前两个一样, 后面按矩阵乘法维度来嘛
        #               1)Att == [beh, beh, N, d] ==== 第一个beh:乘到V的每一行,  第二个beh: 乘到
        #               2)reduce_sum: 变成权重[beh, beh, N, 1], 即,将最后一d变为1------这个有按照原来的维度理解吗, 原来的维度自然而然的是1所以不用, 现在高维度是多了这一步
        #               3)score的softmax: [beh, beh(softmax), N, 1]------是不是要对哪个维度做self-attention,就对下个维度做softmax
        #       (4).att*V:
        #               0)V: [1, beh, N, d]------这里在第一维增加是应广播机制要求吗?
        #               1)[beh, beh, N, 1]*[1, beh, N, d]  ==> [beh, beh(reduce_sum), N, d]------按高维矩阵乘法来说后两维, 没法运算啊
        #               2)[beh, N, d]
        #       (5).到底哪里是做矩阵乘法哪里是做点乘
        # #  3.这一个:   上一个转化+这一个
        #       (6).最后一步： [1, ,d]*[beh, N, d]


        #TODO: 循环, 从1开始, 将每个时间段和上个时间段的embedding传入,  得到新的embedding,  将更新的embedding与t时刻的embedding相加

        #TODO: 直接将各种行为的embedding取均值


#——————————————————————————————————————————————————————————————————————————————————以下

        for t in range(0, self.time_number):
            if t==0:
                continue
            else:  #[3, 3, 147894, 1])  #self.multi_head_
                # user_z = self.act(torch.matmul(self.self_attention(self.weight_dict, self.embedding_dict['times_user_embedding'][t-1], self.embedding_dict['times_user_embedding'][t]), self.weight_dict['w_self_attention_user']))  #pytorch是个不能随随便便写自加的东西
                # item_z = self.act(torch.matmul(self.self_attention(self.weight_dict, self.embedding_dict['times_item_embedding'][t-1], self.embedding_dict['times_item_embedding'][t]), self.weight_dict['w_self_attention_item']))
                
                user_z = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['times_user_embedding'][t-1], self.embedding_dict['times_user_embedding'][t])
                item_z = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['times_item_embedding'][t-1], self.embedding_dict['times_item_embedding'][t])
                
                for i, beh in enumerate(self.behavior):
                    self.embedding_dict['times_user_embedding'][t][beh] = (self.embedding_dict['times_user_embedding'][t][beh] + user_z[i]) /2
                    self.embedding_dict['times_item_embedding'][t][beh] = (self.embedding_dict['times_item_embedding'][t][beh] + item_z[i]) /2

                    # self.embedding_dict['times_user_embedding'][t][beh] = self.act(torch.matmul((self.embedding_dict['times_user_embedding'][t][beh] + user_z[i]) /2, self.weight_dict['w_self_attention_user']))
                    # self.embedding_dict['times_item_embedding'][t][beh] = self.act(torch.matmul((self.embedding_dict['times_item_embedding'][t][beh] + item_z[i]) /2, self.weight_dict['w_self_attention_item']))

                    # self.embedding_dict['times_user_embedding'][t][beh] = self.act((self.embedding_dict['times_user_embedding'][t][beh] + user_z[i]) /2)
                    # self.embedding_dict['times_item_embedding'][t][beh] = self.act((self.embedding_dict['times_item_embedding'][t][beh] + item_z[i]) /2)

                    # self.embedding_dict['times_user_embedding'][t][beh] = (self.weight_dict['alpha'][0]*self.embedding_dict['times_user_embedding'][t][beh] + (1-self.weight_dict['alpha'][0])*user_z[i])  #1:[N, hidden_dim]  2:[N, hidden_dim]  3:[N, hidden-dim]
                    # self.embedding_dict['times_item_embedding'][t][beh] = (self.weight_dict['alpha'][1]*self.embedding_dict['times_item_embedding'][t][beh] + (1-self.weight_dict['alpha'][1])*item_z[i])

                #原来有过想法在最后一个时间段做attention
    #______________________________________________以上
        #这个要分别对user， item处理
        #[beh, N, d][d,1]==>[beh, N, 1]  TODO: 需要在参数表里准备[d,1]的参数
        #[beh, N, d][beh, N, 1] 
        #[beh(reduce_sum), N, 1]==>[N, 1]
        user_embedding = self.behavior_attention(self.embedding_dict['times_user_embedding'][self.time_number-1])
        item_embedding = self.behavior_attention(self.embedding_dict['times_item_embedding'][self.time_number-1])
        
        # ------------以下为原来的代码块：
        # user_embeddings = []
        # item_embeddings = []
        # for i, beh in enumerate(self.behavior):  #最后一个行为已经有nan的数据了

        #     user_embeddings.append(self.embedding_dict['times_user_embedding'][(self.time_number-1)][beh])# todo 多个时间embedding
        #     item_embeddings.append(self.embedding_dict['times_item_embedding'][(self.time_number-1)][beh])

        # user_embeddings = torch.stack(user_embeddings, dim=0)  #这里的数据类型: ParameterDict
        # item_embeddings = torch.stack(item_embeddings, dim=0)

        # user_embedding = torch.mean(user_embeddings, dim=0)
        # item_embedding = torch.mean(item_embeddings, dim=0)  #TODO: 我深深地怀疑自己做加和, 做内积的维度都错了, 到时候再改喽, 因为现在害怕维度错, 会有问题



            #将多个行为的embedding进行平均融合
        return user_embedding, item_embedding


        # TODO: 师兄的组织方式是: 将item的数据排序, 然后 "将同一个时段的一起处理"
        # TODO: 看看HyperRec那篇就知道, 小段数据如何和整体配合

    def para_dict_to_tenser(self, para_dict):  #这个类自己的函数: 在参数和调用的时候都要加 self
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])

        tensors = torch.stack(tensors, dim=0)

        return tensors

    def self_attention(self, trans_w, embedding_t_1, embedding_t):  #到底要不要写一个类, TODO: 问一下师兄或者大佬, 到底啥时候要写类, 啥时候一个函数就可以解决
        """
        就实现attention自己
        """
        # TODO:
        #       (1).得到QKV:
        #               1)乘以W, 得到QKV: [beh, N, d][d, d] == [beh, N, d]------问题:每个时段一个W, 还是多个时段共用一个W, 还有K:t-1------[beh,N,d][d,d]==>[beh*N,d][d,d]==>做完矩阵乘法后[beh*N, d]==>再将维度还原回来[beh,N,d]
        #               2)得到每个时段的Q: 将维度[beh, N, d]扩展为[beh, 1, N, d], 即在第二个维度扩展------想要[beh, beh]必须Q[beh, 1]------如果填充到[第一维]是广播机制, 那么填充到[第二维]是什么机制
        #       (2).K(参数):是每个时段各有一个, 还是大家共用一个K?
        #               1)正常来说每个时段一个[beh, N, d]: 不不, 我先选择一个, 可以吹memory机制, 多了会过拟合
        #               2)乘以W: [beh, N, d]*[beh, d, d] ==> [beh, N, d]
        #               3)扩展维度: [1, beh, N, d]------想要[beh, beh]必须Q[1,beh]即,相当于[beh,1][1,beh]专置相乘
        #       (3).得到每个时段的att:
        #               0)[beh, 1, N, d]*[1, beh, N, d]  ====  "这一步就相当于: Q*K^T专置相乘"------高维矩阵的运算法则不是: 前两个一样, 后面按矩阵乘法维度来嘛
        #               1)Att == [beh, beh, N, d] ==== 第一个beh:乘到V的每一行,  第二个beh: 乘到
        #               2)reduce_sum: 变成权重[beh, beh, N, 1], 即,将最后一d变为1------这个有按照原来的维度理解吗, 原来的维度自然而然的是1所以不用, 现在高维度是多了这一步
        #               3)score的softmax: [beh, beh(softmax), N, 1]------是不是要对哪个维度做self-attention,就对下个维度做softmax
        #       (4).att*V:
        #               0)V: [1, beh, N, d]------这里在第一维增加是应广播机制要求吗?
        #               1)[beh, beh, N, 1]*[1, beh, N, d]  ==> [beh, beh(reduce_sum), N, d]------按高维矩阵乘法来说后两维, 没法运算啊
        #               2)[beh, N, d]

        # def forward(self, trans_w, embedding_t_1, embedding_t):  #TODO: 搞清楚在哪个时段下是[beh][t], 哪个时段下是[t][beh]
                  # att*V
                  # 沿着第二个维度做reduce_sum
                  # 返回: 更新后的t时刻embedding

        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.hidden_dim/args.head_num

        # q = self.layer_norm(q)  
        # k = self.layer_norm(k)
        # v = self.layer_norm(v)

        Q = torch.matmul(q, trans_w['w_q'])  #TODO: 现在是一个parameterDict, 如何将Dict中的各个value的key取出来, 进行矩阵运算
        K = torch.matmul(k, trans_w['w_k'])
        V = torch.matmul(v, trans_w['w_v'])

        Q = torch.unsqueeze(Q, 1)  #[beh, 1, N, d]
        K = torch.unsqueeze(K, 0)  #[1, beh, N, d]
        V = torch.unsqueeze(V, 0)  #[1, beh, N, d]

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        # print("att 的类型:", type(att))
        # print("att 的大小1:", att.shape)  #[3,3,147894, 16]   要打印大小: .shape或者.size()
        att = torch.sum(att, dim=-1) #, keepdims=True)  #torch中的sum函数, 和tf中的reduce_sum函数是一样的@ #todo: one line implementation
        att = torch.unsqueeze(att, dim=-1)  #torch中的sum函数, 和tf中的reduce_sum函数是一样的
        # print("att 的大小2:", att.shape)  #[3,3,147894]
        att = F.softmax(att, dim=1)  #torch中的softmax函数: torch.nn.functional.softmax
        # print("att 的大小3:", att.shape)  #[3,3,147894]

        self.self_attention_para = nn.Parameter(att)

        Z = torch.mul(att, V)  #TODO: 只是不知道这里返回的是: [beh, 1, N, d]还是[beh, N, d]   到时候再看?
        # print("Z 的大小1:", Z.shape)  #[3,3,147894,16]
        Z = torch.sum(Z, dim=1)  #TODO: 只是不知道这里返回的是: [beh, 1, N, d]还是[beh, N, d]   到时候再看?
 

        return Z

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  #到底要不要写一个类, TODO: 问一下师兄或者大佬, 到底啥时候要写类, 啥时候一个函数就可以解决
       
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.hidden_dim/args.head_num

        mutil_head_attention = MultiHeadAttention().cuda()
        Z, att = mutil_head_attention(q, k, v)

        self.multi_head_self_attention_para = nn.Parameter(att)

        #-------------原来的代码-------------------------------------------------------------
        # q = self.layer_norm(q)  
        # k = self.layer_norm(k)
        # v = self.layer_norm(v)

        # Q = torch.matmul(q, trans_w['w_q'])  #TODO: 现在是一个parameterDict, 如何将Dict中的各个value的key取出来, 进行矩阵运算
        # K = torch.matmul(k, trans_w['w_k'])
        # V = torch.matmul(v, trans_w['w_v'])  #[beh, N, d]

        # Q = Q.view(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  #[beh, N, head_num, d]  [head_num, beh, N, d]
        # K = Q.view(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)
        # V = Q.view(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        # Q = torch.unsqueeze(Q, 2)  #[beh, 1, N, d] [head_num, beh, beh, N, d_h]
        # K = torch.unsqueeze(K, 1)  #[1, beh, N, d]
        # V = torch.unsqueeze(V, 1)  #[1, beh, N, d]

        # att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  #[head_num, beh, beh, N, d_h]
        # att = torch.sum(att, dim=-1) #[head_num, beh, beh, N]
        # att = torch.unsqueeze(att, dim=-1)  #[head_num, beh, beh, N, 1]
        # att = F.softmax(att, dim=2)  #[head_num, beh, beh*, N, 1]

        # self.multi_head_self_attention_para = nn.Parameter(att)

        # Z = torch.mul(att, V)  #[head_num, beh, beh, N, d_h]
        # Z = torch.sum(Z, dim=1)  #[head_num, beh, N, d_h]

        # Z_list = [value for value in Z]
        # Z = torch.cat(Z_list, -1)
        # Z = self.dropout(Z)
        # # Z = self.self_attention_net(Z)
        #-------------原来的代码-------------------------------------------------------------

        return Z

    def behavior_attention(self, embedding_input):  #TODO: query
        embedding = self.para_dict_to_tenser(embedding_input)  #[beh, N, d]  计算需要的量：因为， 参数输入的是字典， 这里转成高维矩阵 
        attention = torch.matmul(embedding, self.weight_dict['w_d_d'])  #[beh, N, d][d, 1]==>[beh, N, 1]*32   TODO:32   (8,16,32)  TODO: softmax
        attention = F.softmax(attention, dim=0)*2.5  #TODO:这里稍微变大一点就loss==nan
        self.attention_para = nn.Parameter(attention)

        Z = torch.mul(attention, embedding)  #[beh, N, 1][beh, N, d]==>[beh, N, d]
        Z = torch.sum(Z, dim=0)  #[beh, N, d]==>[N, d]

        return Z


# myModel里会用到HGNN
class HGNN(nn.Module):
    def __init__(self, userNum, itemNum):
        super(HGNN, self).__init__()  #注意区分下面初始化函数和上面初始化函数的不同
        # self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim
        self.sigmoid = torch.nn.Sigmoid()
        # self.embedding_dict = self.init_weight(userNum, itemNum, args.hidden_dim)  #TODO:肯定要初始化
        self.user_embedding, self.item_embedding = self.init_embedding()
        self.alpha, self.i_concatenation_w, self.u_concatenation_w = self.init_weight()
        # self.slope = args.slope  
        self.act = torch.nn.LeakyReLU(negative_slope=args.slope)  #TODO: slope再解决
        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):  #TODO:这里环测师兄-1, 我没有, 到时候再研究    ????为什么层数是2, 但是这个ModuleList的长度却是:7
            self.layers.append(HGNNLayer(args.hidden_dim, args.hidden_dim, weight=True, activation=self.act))  #原来这里是由参数定义实现的, 现在自己直接定义为隐藏层大小的权重

    def init_embedding(self):
        """
        作用: 这里的初始化初始的不是: transformation的W, 而是embedding
        """
        # initializer = nn.init.xavier_uniform_
        # embedding_dict = nn.ParameterDict({
        #     'user_emb': nn.Parameter(initializer(torch.empty(userNum, hidden_dim))),
        #     'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hidden_dim))),
        # })

        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        # self.user_embedding.weight = initializer(torch.empty(self.userNum, args.hidden_dim))
        # self.item_embedding.weight = initializer(torch.empty(self.itemNum, args.hidden_dim))
        # self.user_embedding = nn.Embedding.from_pretrained(initializer(torch.empty(userNum, hidden_dim)))
        # self.item_embedding = nn.Embedding.from_pretrained(initializer(torch.empty(itemNum, hidden_dim)))

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  #两个值, 分别是item, user的embedding
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        # init.xavier_uniform_(alpha)


        return alpha, i_concatenation_w, u_concatenation_w

    def forward(self, G, U, input_item_embedding, input_user_embedding):
        # all_user_embeddings = self.embedding_dict['user_emb']
        # all_item_embeddings = self.embedding_dict['item_emb']
        all_item_embeddings = []
        all_user_embeddings = []

        #--------------------------------alpha的版本---------------------------------------------------------
        self.alpha.data = self.sigmoid(self.alpha)  #TODO: 到时候检查一下, 看这里有没有学到东西
        item_embedding = self.alpha[0]*input_item_embedding + (1-self.alpha[0])*self.item_embedding.weight
        user_embedding = self.alpha[1]*input_user_embedding + (1-self.alpha[1])*self.user_embedding.weight
        #--------------------------------alpha的版本---------------------------------------------------------

        # item_embedding = args.gate_rate*input_item_embedding + (1-args.gate_rate)*self.item_embedding.weight
        # user_embedding = args.gate_rate*input_user_embedding + (1-args.gate_rate)*self.user_embedding.weight
        

        for i, layer in enumerate(self.layers):
            # if i == 0:
            #     self.embedding_dict['item_emb'].weight, self.embedding_dict['user_emb'].weight = layer(G, U, item_embedding_param, user_embedding_param)
            # else:
            item_embedding, user_embedding = layer(G, U, item_embedding, user_embedding)

            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)  #TODO:
            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)

            # if i==0:
            #     all_item_embeddings = norm_item_embeddings  #TODO: 这里的+运算研究一下
            #     all_user_embeddings = norm_user_embeddings
            # else:
            #     all_item_embeddings = torch.stack()
            #     all_user_embeddings = torch.stack()

            all_item_embeddings.append(norm_item_embeddings)
            all_user_embeddings.append(norm_user_embeddings)

        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embedding = torch.cat(all_user_embeddings, dim=1)

        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w)
        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w)


        # all_item_embeddings = torch.stack(all_item_embeddings, dim=0)  #TODO: torch.mean不能对list做, 但是torch.stack可以最list做
        # all_user_embeddings = torch.stack(all_user_embeddings, dim=0)  #TODO: 要取mean的话, 对stack后的tensor的某一维取均值即可
        # item_embedding = torch.mean(all_item_embeddings, dim=0)  #一个问题就是user取最后一层的进行计算就可以了
        # user_embedding = torch.mean(all_user_embeddings, dim=0)  #TODO: 1.看看到时候维度对不对  2.得到所有的embedding之后如何进行mean

        return item_embedding, user_embedding

# HGNN会用HGNNLayer
class HGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=True, activation=None):
        super(HGNNLayer, self).__init__()
        # self.weight = weight  #所以这个参数可以控制, 要不要训练

        # self.act = torch.nn.ELU()
        self.act = torch.nn.PReLU()
        # self.act = torch.nn.LeakyReLU(args.slope)
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

    def forward(self, G, U, item_embedding_para, user_embedding_para):

        item_embedding = torch.mm(item_embedding_para, self.i_w)
        item_embedding = torch.mm(G, item_embedding)
        item_embedding = self.act(item_embedding)


        user_embedding = torch.mm(item_embedding, self.u_w)
        user_embedding = torch.mm(U, user_embedding)  #左边sp， 右边dense

        # torhc.spmm 
        # torch.sparse.mm(sp, dense)   #TODO:
        user_embedding = self.act(user_embedding)  #TODO:

        return item_embedding, user_embedding




