from torch.utils.tensorboard import SummaryWriter
from SEPSO_Down_Path import SEPSO_Down_PATH
import numpy as np
import torch
import os

'''
对SEPSO_Up_Path.py保存下来的Tbest.pt参数逐一进行进行eval_times次评估(评估值都取log10)
运行结束后，得到：
    1.最优的Tbest参数, torch.tensor, shape=(48,)
    2.最优Tbest参数做eval_times次评估的曲线数组, np.array, shape=(eval_times, 5)
    3.Tbest.pt中各参数的Tbest_value的eval_times次曲线, np.array, shape=(eval_times, 500)
    4.Tbest.pt中各参数的Tbest_value的eval_times次均值曲线, 用tensorboard画, 
      用于验证Self-evolving是否有效, 也可确定best_params的在Tbest.pt中的索引
'''


def evaluate():
    ExpName = 'SEPSO_Path_N10_T501_E10 2023-07-19 17_14'
    writer = SummaryWriter(log_dir='runs/'+ExpName)
    BF = 'path'
    try: os.mkdir(f'BF{BF}')
    except: pass


    FileName = 'Tbest/' + ExpName + '.pt'
    Evolved_Tbest = torch.load(FileName)  # (500,48)
    max_evolution = Evolved_Tbest.shape[0] # 共保存了多少组参数，即进化了多少轮


    eval_times = 50 # Evolved_Tbest中每组参数评估多少次
    sepso_down = SEPSO_Down_PATH(SaveCurve=True)

    curves_temp = np.zeros((eval_times, sepso_down.Max_iterations)) # 当前参数的eval_times条曲线, (50,1400)
    Tbest_value_mean_best = np.inf

    Tbest_value_curves = np.zeros((eval_times, max_evolution)) # (50,500), Tbest.pt中各参数的Tbest_value的eval_times次曲线

    # 遍历保存的所有参数
    for e in range(max_evolution):
        params = Evolved_Tbest[e].clone()

        # 每个参数进行eval_times次评估
        for _ in range(eval_times):
            sepso_down.evaluate_params(params)
            curves_temp[_] = sepso_down.curve.cpu().clone().numpy()

        Tbest_value_curves[:, e] = curves_temp[:,-1]
        Tbest_value_mean = curves_temp[:,-1].mean()
        print(f'BF{BF}, Param:{e}, Tbest_value_mean:{Tbest_value_mean}')
        writer.add_scalar(f'Tbest_value_mean{eval_times}', Tbest_value_mean, e) #4.Tbest.pt中各参数的Tbest_value的eval_times次均值曲线

        if Tbest_value_mean<Tbest_value_mean_best:
            Tbest_value_mean_best = Tbest_value_mean
            curves_best = curves_temp.copy() #np.array
            Bestparams = params #torch.tensor

    np.save(f'BF{BF}/Tbest_value_curves.npy',Tbest_value_curves) # 3.Tbest.pt中各参数的Tbest_value的eval_times次曲线
    np.save(f'BF{BF}/SEPSO-R{eval_times}.npy',curves_best) #2.最优Tbest参数做eval_times次评估的曲线数组
    torch.save(Bestparams,f'BF{BF}/SEPSO_Bestparams.pt') #1.最优的Tbest参数



if __name__ == '__main__':
    evaluate()





