#就是师兄一开始让画的稀疏度的图, 模型和许多baseline都放进去的那个
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

barVals = [103, 46, 52, 38]
names = ['0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
barName = '# Areas'

lines = {
	'SVM': [0.1738, 0.3996, 0.6976, 0.8942],
	'STGCN': [0.0226, 0.4558, 0.8159, 0.9317],
	'GMAN': [0.2318, 0.5361, 0.8269, 0.9353],
	'DeepCrime': [0.2271, 0.5811, 0.7930, 0.9296],
	'ST-MetaNet': [0.0653, 0.3146, 0.9509, 0.9999],
	'ST-SHN': [0.3815, 0.6204, 0.8272, 0.9353],
}
title = 'NYC_macro'
lineMetName = 'Macro-F1'     #在右侧竖着放下了
ylim = [-0.05, 1.05]      #y轴的边界




lineColors = ['#0e72cc', '#6ca30f', '#f59311', '#16afcc', '#555555', '#fa4343']
# lineMarkers = ['x', 'D', 's', '+', 'o']
lineMarkers = ['v', '^', '<', '>', 'D', 'o']

# barColor = '#80808080'
barColor = '#76500570'   #柱状图的颜色


matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in' 
x = [i*0.3 for i in range(len(names))]
width = 0.2
plt.bar(x, barVals, width=width, tick_label=names, fc=barColor)  #画柱状图需要凑齐"各种参数"

fig, ax1 = plt.subplots(1, figsize=(13/3*0.8, 2.9*0.8))
# fig, ax1 = plt.subplots(1, figsize=(30/3*0.8, 5*0.8))
ax1.bar(x, barVals, width=width, tick_label=names, fc=barColor)
ax1.set_ylabel(barName, color='#765005', fontweight='bold', labelpad=-8)#8)

ax2 = ax1.twinx()
i = 0
for lineName in lines:
	ax2.plot(x, lines[lineName], color=lineColors[i], marker=lineMarkers[i])
	i += 1
ax2.set_ylabel(lineMetName, fontweight='bold', fontsize=10, labelpad=1)
ax2.set_ylim(ylim[0], ylim[1])

plt.grid(axis='y', ls='--')
plt.grid(axis='x', ls='--')

plt.legend(list(lines.keys()), loc='lower right', ncol=2, framealpha=1, fancybox=False, handlelength=1.2, handleheight=1.2, handletextpad=0.4, labelspacing=0.4, columnspacing=0.2, fontsize=10, borderaxespad=-0.25, borderpad=0.2)

fig.tight_layout()    #师兄说, 这个是输出为pdf的时候页边距是多少
plt.subplots_adjust(top=0.958,
bottom=0.076,
left=0.106,
right=0.85,
hspace=0.2,
wspace=0.2)
plt.show()
plt.savefig('figures/sparsity_%s.pdf' % title)

