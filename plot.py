import pickle
import matplotlib.pyplot as plt
import numpy as np

colors = [
    'red', 'cyan', 'blue', 'green', 'black', 'magenta', 'pink', 'purple',
    'chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon',
    'gold', 'darkred'
]

lines = ['-', '--', '-.', ':']



#Tmall
sets = [
   "enhance_self_attention_multi_4_self_attention_behavior_Tmall_2021_03_17__19_53_06_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "all_self_attention_behavior_Tmall_2021_03_17__23_15_25_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "all_self_attention_behavior_Tmall_2021_03_18__11_09_08_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "mean_self_attention_behavior_Tmall_2021_03_17__23_20_02_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "RNN_self_attention_behavior_Tmall_2021_03_17__23_26_13_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "mean_self_attention_behavior_Tmall_2021_03_18__12_27_59_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[12, 12]",
   "RNN_self_attention_behavior_Tmall_2021_03_17__23_32_58_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "Nobehavior_self_attention_behavior_Tmall_2021_03_17__23_38_37_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
   "no_behavior_self_attention_behavior_Tmall_2021_03_18__09_47_10_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16]",
]

names = [
 "ours3",
 "ours1",
 "ours",
 "mean2",
 "RNN",
 "mean",
 "No-GNN",
 "No-behavior",
 "No-behavior2",
]

smooth = 1
startLoc = 1
Length = 150

# dataset = "Epinions_time"
dataset = "Tmall"
# dataset = "IJCAI_15"


#获得序列的length
MIN_LENGTH = 10000
for j in range(len(sets)):
    val = sets[j]
    name = names[j]
    print('val', val)
    with open(r'./History/' + dataset + r'/' + val + '.his', 'rb') as fs:
        res = pickle.load(fs)
    temlength = len(res['HR'])
    print(temlength)
    if temlength<MIN_LENGTH:
        MIN_LENGTH = temlength 



for j in range(len(sets)):
    val = sets[j]
    name = names[j]
    print('val', val)
    with open(r'./History/' + dataset + r'/' + val + '.his', 'rb') as fs:
        res = pickle.load(fs)
    # rmse = res['step_rmse']
    # mae = res['step_mae']
    # for i in range(len(rmse)):
    # 	print("rmse %d: %.4f"%(i, rmse[i]))
    # for i in range(len(mae)):
    # 	print("mae %d: %.4f"%(i, mae[i]))

    # printBest(res)
    length = Length
    temy = [None] * 3
    temlength = len(res['HR'])
    temy[0] = np.array(res['loss'][startLoc:min(length, temlength)])
    temy[1] = np.array(res['HR'][startLoc:min(length, temlength)])
    temy[2] = np.array(res['NDCG'][startLoc:min(length, temlength)])
    for i in range(3):
        if len(temy[i]) < length - startLoc:
            temy[i] = np.array(
                list(temy[i]) + [temy[i][-1]] * (length - temlength))  #将最后一个值补全
    length -= 1
    y = [[], [], []]
    for i in range(int(length / smooth)):
        if i * smooth + smooth - 1 >= len(temy[0]):
            break
        for k in range(3):
            temsum = 0.0
            for l in range(smooth):
                temsum += temy[k][i * smooth + l]
            y[k].append(temsum / smooth)
    y = np.array(y)
    length = y.shape[1]
    x = np.zeros((3, length))


    for i in range(3):
        x[i] = np.array(list(range(length)))
    plt.figure(1)
    plt.subplot(131)
    plt.title('LOSS FOR TRAIN')
    plt.plot(x[0], y[0], color=colors[j], label=name)
    plt.legend()

    plt.subplot(132)
    plt.title('HR FOR TEST')
    plt.plot(x[1], y[1], color=colors[j], label=name)
    plt.legend()

    plt.subplot(133)
    plt.title('NDCG FOR TEST')
    plt.plot(x[2], y[2], color=colors[j], label=name)
    plt.legend()

plt.show()
