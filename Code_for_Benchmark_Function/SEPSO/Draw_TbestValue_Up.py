import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

'''本代码用于画论文4.2.1节的图'Fitness curves of the Higher-level SEPSO on 4 benchmark functions' '''

dir = 'runs/SEPSO_BF1_N10_T501_bsFalse 2023-07-12 16_08'
event_acc = EventAccumulator(dir)
event_acc.Reload()
events = event_acc.Scalars('LogTbest')
data = [event.value for event in events]
data = np.array(data)
plt.plot(np.arange(data.shape[0]), data, label='BF1')

dir = 'runs/SEPSO_BF2_N10_T501_bsFalse 2023-07-12 20_18'
event_acc = EventAccumulator(dir)
event_acc.Reload()
events = event_acc.Scalars('LogTbest')
data = [event.value for event in events]
data = np.array(data)
plt.plot(np.arange(data.shape[0]), data, label='BF2')

dir = 'runs/SEPSO_BF3_N10_T501_bsFalse 2023-07-13 01_19'
event_acc = EventAccumulator(dir)
event_acc.Reload()
events = event_acc.Scalars('LogTbest')
data = [event.value for event in events]
data = np.array(data)
plt.plot(np.arange(data.shape[0]), data, label='BF3')

dir = 'runs/SEPSO_BF4_N10_T501_bsFalse 2023-07-13 06_21'
event_acc = EventAccumulator(dir)
event_acc.Reload()
events = event_acc.Scalars('LogTbest')
data = [event.value for event in events]
data = np.array(data)
plt.plot(np.arange(data.shape[0]), data, label='BF4')


fsize = 16

# 添加标题和坐标轴标签
# plt.title('Plot with Labels')
plt.xticks(size=fsize)
plt.yticks(size=fsize)
plt.xlabel('Number of evolutions',fontsize=fsize)
plt.ylabel('Fitness (log)',fontsize=fsize)
plt.tight_layout()
plt.grid()


# 创建图例
plt.legend(fontsize=fsize-4)
plt.savefig('TbestValue_Up.pdf',bbox_inches="tight")

