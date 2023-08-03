from torch.utils.tensorboard import SummaryWriter
from SEPSO_Down_Path import SEPSO_Down_PATH
from datetime import datetime
import os,shutil
import argparse
import torch
import time

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting for DPPSO_GPU'''
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=10, help='Total Number of Particles in one Group')
parser.add_argument('--Total_iterations', type=int, default=501, help='Total iterations') #多1好画图，否则索引到499不方便
parser.add_argument('--write', type=str2bool, default=True, help='Whether record fitness curve')
parser.add_argument('--save', type=str2bool, default=True, help='Whether save the Tbest')
parser.add_argument('--Bootstrap', type=str2bool, default=False, help='Wether Update SEPSO_Up parameters with Tbest')
parser.add_argument('--ET', type=int, default=10, help='Evaluation times of fitness function')
opt = parser.parse_args()

class SEPSO_UP():
    def __init__(self, opt):
        ''' SEPSO_UP是Self-evolving PSO的上层算法, 实际上是使用的DTPSO
            核心功能: 生成多组params(每组params是SEPSO_UP的一个粒子), 使用SEPSO_DOWN来评估该params, 使用SEPSO_UP来找到最优param
            注意:
                1.params在传入前需要clone以防止数据联动.
                2.SEPSO_UP和SEPSO_DOWN的self.G, self.D不一致; 当UP需要Bootstrap()时self.N必须一致
                3.SEPSO_UP的搜索空间并不相对于0对称, 并且粒子中的不同成分有不同的搜索空间'''

        self.Total_iterations = opt.Total_iterations
        self.dvc = torch.device('cpu')
        self.G, self.N, self.D = 8, opt.N, 48 # number of Groups, particles in goups, and particle dimension
        self.arange_idx = torch.arange(self.G, device=self.dvc)  # 索引常量
        self.fitness = torch.ones((self.G, self.N)) * torch.inf
        self.Bootstrap = opt.Bootstrap
        self.ET = opt.ET
        self.fitHoder = torch.zeros(self.ET)

        # 代表SEPSO_DOWN的w_init, w_end_rate(w_end = w_init*w_end_rate), v_limit_ratio, C1, C2, C3
        self.Max_range = torch.tensor([0.9, 0.9, 0.8, 2.0, 2.0, 2.0]).unsqueeze(-1).repeat(1,self.G).view(-1) #(self.D,)
        self.Min_range = torch.tensor([0.2, 0.1, 0.1, 1.0, 1.0, 1.0]).unsqueeze(-1).repeat(1,self.G).view(-1) #(self.D,)

        # R Matrix, (4,G,N,1)
        self.Random = torch.ones((4,self.G,self.N,1), device=self.dvc) #更新速度时在装载随机数

        # 生成本次实验名字
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        self.Exp_Name = f'SEPSO_Path_N{opt.N}_T{opt.Total_iterations}_E{opt.ET}' + timenow

        self.save = opt.save
        if opt.save:
            try: os.mkdir('Tbest')
            except: pass
            self.Tbest_all = torch.zeros((self.Total_iterations, self.D))

        self.write = opt.write
        if opt.write:
            writepath = 'runs/'+self.Exp_Name
            if os.path.exists(writepath): shutil.rmtree(writepath)
            self.writer = SummaryWriter(log_dir=writepath)


    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def reset(self, params):
        '''Reset the parameters and the particles of the DTPSO'''
        # Inertia Initialization for 8 Groups
        self.w_init = params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (params[0:self.G]*params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)  # 0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3
        self.w_delta = (self.w_init - self.w_end)/self.Total_iterations # (G,1,1)


        # Velocity Constraint Initialization for 8 Groups
        v_limit_ratio = params[(2*self.G):(3*self.G)] #(G,)
        Haf_range = ((self.Max_range - self.Min_range)/2).expand((self.G, 1, self.D)) # (G,1,D), 搜索区间长度的一半,相当于DTPSO的Search_range[1]
        self.v_max = v_limit_ratio[:, None, None] * Haf_range #(G,1,D) + (G,1,1)*(G,1,D) = (G,1,D)
        self.v_min = -self.v_max #(G,1,D)


        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = params[(3*self.G):(4*self.G)]
        self.Hyper[2] = params[(4*self.G):(5*self.G)]
        self.Hyper[3] = params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

        # L Matrix, (4,G,N,D)
        self.Locate = torch.zeros((4,self.G,self.N,self.D), device=self.dvc)
        self.Locate[1:4] = self._uniform_random(low=self.Min_range, high=self.Max_range, shape=(self.G,self.N,self.D)) #[0,X,X,X]

        # K Matrix, (4,G,N,D)
        self.Kinmtc = torch.zeros((4,self.G,self.N,self.D), device=self.dvc) #[V,Pbest,Gbest,Tbest]
        self.Kinmtc[0] = self._uniform_random(low=-self.Max_range, high=self.Max_range, shape=(self.G,self.N,self.D))
        self.Kinmtc[0].clip_(self.v_min, self.v_max)  # 限制速度范围

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf

    def _bootstrap(self):
        '''Set the parameters of SEPSO_Up with its Tbest'''
        params = self.Kinmtc[3,0,0].clone()

        # Inertia Initialization for 8 Groups
        self.w_init = params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (params[0:self.G]*params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)  # 0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3
        self.w_delta = (self.w_init - self.w_end)/self.Total_iterations # (G,1,1)


        # Velocity Initialization for 8 Groups
        v_limit_ratio = params[(2*self.G):(3*self.G)] #(G,)
        Haf_range = ((self.Max_range - self.Min_range)/2).expand((self.G, 1, self.D)) # (G,1,D), 搜索区间长度的一半,相当于DTPSO的Search_range[1]
        self.v_max = v_limit_ratio[:, None, None] * Haf_range #(G,1,D) + (G,1,1)*(G,1,D) = (G,1,D)
        self.v_min = -self.v_max #(G,1,D)


        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = params[(3*self.G):(4*self.G)]
        self.Hyper[2] = params[(4*self.G):(5*self.G)]
        self.Hyper[3] = params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

    def iterate(self):
        sepso_down = SEPSO_Down_PATH()
        '''双for循环遍历计算SEPSO_UP所有粒子的Fitness'''
        for i in range(self.Total_iterations):
            t0 = time.time()
            for Gid in range(self.G):
                for Nid in range(self.N):
                    for _ in range(self.ET):
                        self.fitHoder[_] = sepso_down.evaluate_params( self.Locate[1,Gid,Nid].clone() )
                    self.fitness[Gid, Nid] = self.fitHoder.mean()



            ''' Step 2: 更新Pbest, Gbest, Tbest 的 value 和 particle '''
            # Pbest
            P_replace = (self.fitness < self.Pbest_value) # (G,N)
            self.Pbest_value[P_replace] = self.fitness[P_replace] # 更新Pbest_value
            self.Kinmtc[1, P_replace] = self.Locate[1, P_replace] # 更新Pbest_particles

            # Gbest
            values, indices = self.fitness.min(dim=-1)
            G_replace = (values < self.Gbest_value) # (G,)
            self.Gbest_value[G_replace] = values[G_replace] # 更新Gbest_value
            self.Kinmtc[2, G_replace] = (self.Locate[2, self.arange_idx, indices][G_replace].unsqueeze(1))

            # Tbest
            min_fitness = self.fitness.min()
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value
                flat_idx = self.fitness.argmin()
                self.Kinmtc[3] = (self.Locate[3, flat_idx//self.N, flat_idx % self.N]).clone() #这里必须clone, 否则数据联动

            ''' Step 3: 更新速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Min_range, self.Max_range) # 限制位置范围

            '''Print, Write, and Save'''
            time_per_iteration = round(time.time()-t0,1)
            print(f'Iteration:{i},  logMinvalue:{round(self.Tbest_value.item(),1)}, '
                  f'IterationTime:{time_per_iteration}s,   RemainTime:{round((self.Total_iterations-i)*time_per_iteration/3600,1)}h')

            if self.write:
                self.writer.add_scalar('LogTbest', self.Tbest_value, global_step=i)
                self.writer.add_scalar('MeanFit', self.fitness.mean(), global_step=i)

            if self.save:
                self.Tbest_all[i] = self.Kinmtc[3,0,0].clone()
                if (i+1) % 10 == 0:
                    Saved_name = 'Tbest/' + self.Exp_Name + '.pt'
                    torch.save(self.Tbest_all[0:i], Saved_name)

            if self.Bootstrap: self._bootstrap() # Update SEPSO_Up's parameters with Tbest



if __name__ == '__main__':
    params = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9, # w_init
                           0.5, 0.57, 0.125, 0.75, 0.5, 0.555, 0.25, 0.333, # w_end_ratio
                           0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3, # v_limit_ratio
                           2,1,2,2,2,2,1,1, # C1
                           1,1,2,2,1,1,2,2, # C2
                           1,2,1,1,2,2,2,2], dtype=torch.float) # C3
    sepso_up = SEPSO_UP(opt)
    sepso_up.reset(params)
    sepso_up.iterate()
