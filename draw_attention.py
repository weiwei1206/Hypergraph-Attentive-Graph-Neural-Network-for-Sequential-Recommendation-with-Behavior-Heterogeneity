#自己代码的版本
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


loadPath = "D:\CODE\master_behavior_attention\Model\IJCAI_15\enhance__IJCAI_self_attention_behavior_IJCAI_15_2021_03_19__11_16_01_lr_0.001_reg_0.016_batch_size_4096_time_slot_5184000_gnn_layer_[16, 16].pth"
# loadPath = "D:\CODE\master_behavior_attention\Model\IJCAI_15\_self_attenntion_and_multi_attention_IJCAI_15_2021_03_16__17_32_55_lr_0.001_reg_0.01_batch_size_4096_time_slot_5184000_gnn_layer_[24, 24].pth"
#IJCAI四个头跑的最好的: D:\CODE\master_behavior_attention\Model\IJCAI_15\enhance__IJCAI_self_attention_behavior_IJCAI_15_2021_03_19__11_16_01_lr_0.001_reg_0.016_batch_size_4096_time_slot_5184000_gnn_layer_[16, 16].pth
#IJCAI一个头跑的最好的: D:\CODE\master_behavior_attention\Model\IJCAI_15\_self_attenntion_and_multi_attention_IJCAI_15_2021_03_16__17_32_55_lr_0.001_reg_0.01_batch_size_4096_time_slot_5184000_gnn_layer_[24, 24].pth
checkpoint = torch.load(loadPath)
model = checkpoint['model']
params = model.state_dict()

print(params['multi_head_self_attention_para'].shape)  #[4,4,4,35573,1]
# print(params['self_attention_para'].shape)  #[4,4,35573, 1]
print(params['attention_para'].shape)  #[4,4,35573,1]

multi_head_self_attention_ndarray = params['multi_head_self_attention_para'].squeeze().permute(0, 3, 1, 2, ).cpu().numpy()  #[head_num, beh, beh*, N, 1]
# self_attention_ndarray = params['self_attention_para'].squeeze().permute(2,0,1).cpu().numpy()  #[beh, beh, N, 1]
# attention_ndarray = params['attention_para'].squeeze().permute(1, 0, 2).cpu().numpy()  #[beh, N, 1]

# multi_head_self_attention_ndarray0 = multi_head_self_attention_ndarray[0]  #[35573, 4, 4]
# multi_head_self_attention_ndarray1 = multi_head_self_attention_ndarray[1]
# multi_head_self_attention_ndarray2 = multi_head_self_attention_ndarray[2]
# multi_head_self_attention_ndarray3 = multi_head_self_attention_ndarray[3]



#draw_attention
tmall_beh = ['pv','fav', 'cart', 'buy']
IJCAI_15_beh = ['click','fav', 'cart', 'buy']
att_index = ['att']


#没有multi_head
# df = pd.DataFrame(self_attention_ndarray[6], columns=IJCAI_15_beh, index=IJCAI_15_beh)
# # df = pd.DataFrame(attention_ndarray[0], columns=att_index, index=tmall_beh)
# fig = plt.figure()

# ax = fig.add_subplot(111)

# cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
# fig.colorbar(cax)


# tick_spacing = 1
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

# ax.set_xticklabels([''] + list(df.columns))
# ax.set_yticklabels([''] + list(df.index))
# plt.savefig('D:\CODE\master_behavior_attention\Picture\self_attentionself_attention6.jpg')
# # plt.savefig('D:\CODE\master_behavior_attention\Pictureattentionself_attention.jpg')
# # plt.show()
# # PlotMats(self_attention_ndarray = params['self_attention_para'].cpu().numpy(), , show=False, savePath='visualization/legend.pdf', vrange=[0, 1])


#有multi_head
str_pre = '22'
fig = plt.figure()
for i in range(4):
    self_attention_ndarray = multi_head_self_attention_ndarray[i]
    user_number = 19
    subplot_int = int(str_pre + str(i+1))

    df = pd.DataFrame(self_attention_ndarray[user_number], columns=IJCAI_15_beh, index=IJCAI_15_beh)
    # df = pd.DataFrame(attention_ndarray[0], columns=att_index, index=tmall_beh)

    ax = fig.add_subplot(subplot_int)
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    # fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

# fig.colorbar(cax)
plt.savefig(f'D:\CODE\master_behavior_attention\Picture\multi_head_self_attentionself_attention{user_number}.pdf')
# plt.savefig('D:\CODE\master_behavior_attention\Picture\self_attentionself_attention6.jpg')
# plt.savefig('D:\CODE\master_behavior_attention\Pictureattentionself_attention.jpg')
# plt.show()
# PlotMats(self_attention_ndarray = params['self_attention_para'].cpu().numpy(), , show=False, savePath='visualization/legend.pdf', vrange=[0, 1])


