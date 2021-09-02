import numpy as np
from Params import *
import pickle
import scipy
from scipy.sparse import csr_matrix
# from DataHandler import LoadData

#D:\CODE\DATASET\Tmall
#D:\CODE\DATASET\IJCAI_15

trnMats = pickle.load(open("D:\CODE\DATASET\IJCAI_15/trn_buy", "rb"))   
tstInt = pickle.load(open("D:\CODE\DATASET\IJCAI_15/tst_int", "rb"))
trnLabel = 1*(trnMats != 0)
tstUsrs = [index for index, value in enumerate(tstInt) if value!=None]

# trnMat = np.sum(trnMats)[tstUsrs]    #不知师兄这句是何意, 是多行为吗
trnMat = trnLabel[tstUsrs]

trnnum = np.reshape(np.array(np.sum(trnMat, axis=1)), [-1])
trnnum = np.sort(trnnum)
summ = np.sum(trnnum)
acc = np.cumsum(trnnum)   #按行累加, 第二行加上第一行的数之和
div = summ / 5    #----------------------------------------------------------将总的交互数量进行5等分, 找到相应的分割线
cur = div


marks = [-1]
for i in range(trnnum.shape[0]):
	if acc[i] <= cur and (i==trnnum.shape[0]-1 or acc[i+1] > cur):   #找到边界:  如果当前值小于现在的边界, 之后的值大于现在的边界, 那么:  当前值的交互就是边界了.
		marks.append(trnnum[i])
		cur += div
print(marks)

for i in range(1, len(marks)):
	mark, premark = marks[i], marks[i-1]
	print(mark, np.sum((trnnum<=mark)*(trnnum>premark)))
print(np.min(trnnum), np.max(trnnum))

tstIndex = []
trnnums = np.sum(trnLabel, axis=1)    #还是和列表的维度对应上吧, 原来的是拆成二维, 应该是遍历不了
for i in range(1, len(marks)):
	mark, premark = marks[i], marks[i-1]
	filename = "D:\CODE\DATASET\IJCAI_15/" + "tst_int_" + str(mark)

	tmp_tstIndex = [index for index, value in enumerate(trnnums) if value<=premark or value>mark]
	tmp_tst = np.array(tstInt)
	tmp_tst[tmp_tstIndex] = None
	tmp_tst = tmp_tst.tolist()
	
	# tmp_tstIndex = np.where((trnnums < premark))  # and (trnnums > mark))    #选出不在范围的数
	# tmp_tst[tmp_tstIndex] = None     #打印一下看看值到底正不正确
	pickle.dump(tmp_tst ,open(filename, "wb"))   #

