import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')

	#for this model
	parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  #------自己探索一下寻找到合适的大小
	parser.add_argument('--gnn_layer', default="[16,16]", type=str, help='gnn layers: number + dim')  #------gnn层的尝试
	parser.add_argument('--time_slot', default=60*60*24*360, type=float, help='length of time slots')  #自己加的时间缝隙, 先用30试试
	parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')  #------自己探索一下寻找到合适的大小
	parser.add_argument('--gate_rate', default=0.8, type=float, help='gating rate')  #------如果loss下降慢就调大, 如果不变或者反复横跳就往小调
	parser.add_argument('--point', default='topk_20', type=str, help='')
	parser.add_argument('--title', default='self_attention_behavior', type=str, help='title of model')


	#for train
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  #------如果loss下降慢就调大, 如果不变或者反复横跳就往小调
	parser.add_argument('--batch', default=4096, type=int, help='batch size')  #------batchsise的影响是什么: 一般多少合适
	parser.add_argument('--reg', default=1.45e-2, type=float, help='weight decay regularizer')  #TODO:找到这个参数在哪??
	parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')  #TODO: 再探索一下提前终止, 师兄有用提前终止吗, 还是需要代码
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')  #TODO: 探索一下这个参数在哪
	parser.add_argument('--shoot', default=50, type=int, help='K of top k')
	parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')  #------TODO: 找找看这个参数在哪
	parser.add_argument('--drop_rate', default=0.1, type=float, help='drop_rate')  #------TODO: 找找看, 这个参数在哪
	parser.add_argument('--seed', type=int, default=19)  #------TODO: 随机种子的影响
	parser.add_argument('--slope', type=float, default=0.1)  #
	parser.add_argument('--patience', type=int, default=300)

	#for save and read
	parser.add_argument('--path', default='/home/ww/Code/DATASET/work3_dataset/', type=str, help='data path')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--dataset', default='retailrocket', type=str, help='name of dataset')  #------换数据集, 看看和SOTA差多远
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #布尔值加引号...直接编译为True
	parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
	parser.add_argument('--loadModelPath', default='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/topk_20_self_attention_behavior_IJCAI_15_2021_07_29__17_13_23_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000_gnn_layer_[16,16].pth', type=str, help='loadModelPath')
	#Tmall用來做test的pth:    D:\CODE\master_behavior_attention\Model\Tmall\HR_NDCG_self_attention_behavior_Tmall_2021_03_18__13_27_30_lr_0.001_reg_0.01_batch_size_2048_time_slot_259200_gnn_layer_[16, 16].pth
	#IJCAI:  D:\CODE\master_behavior_attention\Model\IJCAI_15\enhance__IJCAI_self_attention_behavior_IJCAI_15_2021_03_19__11_16_01_lr_0.001_reg_0.016_batch_size_4096_time_slot_5184000_gnn_layer_[16, 16].pth
	#	    D:\CODE\master_behavior_attention\Model\IJCAI_15\_self_attenntion_and_multi_attention_IJCAI_15_2021_03_16__17_32_55_lr_0.001_reg_0.01_batch_size_4096_time_slot_5184000_gnn_layer_[24, 24].pth
	#特意为IJCAI测试训练的：  D:\CODE\master_behavior_attention\Model\IJCAI_15\enhance__IJCAI_self_attention_behavior_IJCAI_15_2021_03_19__19_06_13_lr_0.001_reg_0.016_batch_size_4096_time_slot_5184000_gnn_layer_[16, 16].pth
	#Tmall最好的模型： 
	#IJCAI最好的模型：D:\CODE\master_behavior_attention\Model\IJCAI_15\ours_again_self_attention_behavior_IJCAI_15_2021_03_23__09_50_14_lr_0.001_reg_0.016_batch_size_4096_time_slot_7776000_gnn_layer_[64,64].pth

	#use less
	# parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')  #test_batch大小: 我好像把这个弄大了
	# parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')  #------多头注意力机制, 可以试一下, 看看影响大吗
	# parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')  #TODO: 训练的sample还没有做, 应该试一下
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')  #-----全连接层吗: 还没有实现, 试一下
	parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')  #
	parser.add_argument('--graphSampleN', default=10000, type=int, help='use 25000 for training and 200000 for testing, empirically')  #这里注释是不是敲错一个0
	parser.add_argument('--divSize', default=1000, type=int, help='div size for smallTestEpoch')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')  #??
	parser.add_argument('--slot', default=0.5, type=float, help='length of time slots')  #什么意思时间戳除以5吗???

	
	return parser.parse_args()
args = parse_args()

# TODO: 这几句被注释掉了, 后面用到了. 问题是: args还可以按照后面的这个直接加???
# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734

# swap user and item
# tem = args.user
# args.user = args.item
# args.item = tem

# args.decay_step = args.trn_num
# args.decay_step = args.item//args.batch
args.decay_step = args.trnNum//args.batch
