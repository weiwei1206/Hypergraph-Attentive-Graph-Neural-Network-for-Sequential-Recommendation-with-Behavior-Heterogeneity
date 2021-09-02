# def run(self):
# 		self.prepareModel()
# 		log('Model Prepared')
# 		if args.load_model != None:
# 			self.loadModel()
# 			stloc = len(self.metrics['TrainLoss'])
# 		else:
# 			stloc = 0
# 			init = tf.global_variables_initializer()
# 			self.sess.run(init)
# 			log('Variables Inited')
# 		for ep in range(stloc, args.epoch):
# 			test = (ep % 3 == 0)
# 			reses = self.trainEpoch()
# 			log(self.makePrint('Train', ep, reses, test))
# 			if test:
# 				reses = self.testEpoch()
# 				log(self.makePrint('Test', ep, reses, test))
# 			if ep % 5 == 0:
# 				self.saveHistory()
# 			print()
# 		reses = self.testEpoch()
# 		log(self.makePrint('Test', args.epoch, reses, True))
# 		self.saveHistory()
#
#
# def sampleTestBatch(self, batchIds):  #TODO: 这个函数的作用:
# 		batch = len(batchIds)
# 		temTst = self.tstInt[batchIds]#return value
# 		temLabel = self.label[batchIds].toarray()  #TODO: self.label ?   batch user label information[32,num_item]
# 		temlen = (batch*100)
# 		uIntLoc = [None] * temlen
# 		iIntLoc = [None] * temlen
# 		tstLocs = [None] * (batch)#[bat, 100]
#
# 		cur = 0
# 		for i in range(batch):
# 			posloc = temTst[i]# one positive sample(item)
# 			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])# available negative samples(locations) :有值为false, 为0为true.  得到index
# 			rdmNegSet = np.random.permutation(negset)[:99]  #随机排序取前99个
# 			locset = np.concatenate((rdmNegSet, np.array([posloc])))#99+1
# 			tstLocs[i] = locset
#
# 			for j in range(100):
# 				uIntLoc[cur] = batchIds[i]  #重复100
# 				iIntLoc[cur] = locset[j]
# 				cur += 1
#
# 		return uIntLoc, iIntLoc, temTst, tstLocs
#
#
#
# def testEpoch(self):
# 		# self.atts = [None] * args.user
# 		epochHit, epochNdcg = [0] * 2
# 		ids = self.tstUsrs  #
# 		num = len(ids)
# 		testbatch = np.maximum(1, args.batch * args.sampNum // 100)  #TODO: 改变batch的大小----比train_batch大.  至少是1的大小(dataloader的参数)  leave-one-out
# 		steps = int(np.ceil(num / testbatch))  #
# 		for i in range(steps):  #这句话应该是dataloader做的
# 			st = i * testbatch
# 			ed = min((i+1) * testbatch, num)
# 			batchIds = ids[st: ed]
# 			uIntLoc, iIntLoc, temTst, tstLocs = self.sampleTestBatch(batchIds)  #TODO: 这四个变量之间有什么关系 uIntLoc, iIntLoc[bat*100], temTst[bat]positive sample, tstLocs[bat, 100] tested item locations
# 			preds = self.sess.run(self.pred, feed_dict={self.uids: uIntLoc, self.iids: iIntLoc}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
# 			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)  #TODO: 为什么要改变它的形状
# 			epochHit += hit
# 			epochNdcg += ndcg
# 			log('Step %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
# 		ret = dict()
# 		ret['HR'] = epochHit / num
# 		ret['NDCG'] = epochNdcg / num
# 		return ret
#
#
# def calcRes(self, preds, temTst, tstLocs):  #embedding预测的结果, user的item_i, 每个user需要测试的100个item
#     hit = 0
#     ndcg = 0
#     for j in range(preds.shape[0]):  #一个batch的大小:  传进来的当然是一个batch但是, 循环就表示对每一个测试样例进行处理 [batch, 100]
#         predvals = list(zip(preds[j], tstLocs[j]))  #把ground_truth和每个用户的测试集放在一起.  TODO: 这里是把预测的分数和id放在一起了吗?  100:分数, item
#         predvals.sort(key=lambda x: x[0], reverse=True)  #TODO: 按照第二个升序排列.
#         shoot = list(map(lambda x: x[1], predvals[:args.shoot]))  #TODO:  取到刚刚升序排列的列表, shoot是取前10个元素.   这整句话是什么意思
#         if temTst[j] in shoot: #如果这个item在刚刚排列的里面
#             hit += 1  #是对batch里面的每个user测试, 如果中了, 就把每个结果累加吗
#             ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
#     return hit, ndcg
#
#
#
#
