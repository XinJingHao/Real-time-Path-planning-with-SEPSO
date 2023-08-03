import numpy as np
import argparse
import torch
import time
import os

BF_name = ['Sphere', 'Rosenbrock', 'Rastrigin', 'Griewank']

def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as file:
        for key in dictionary.keys():
            value = dictionary[key]
            file.write(f"{key}: {value}\n")


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

'''Hyperparameter Setting for DTPSO'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='Running devices')
parser.add_argument('--compile', type=str2bool, default=False, help='Whether use torch.compile')
parser.add_argument('--N', type=int, default=10, help='Total Number of Particles in one Group')
parser.add_argument('--D', type=int, default=30, help='Dimension of the Particle')
parser.add_argument('--BF', type=int, default=1, help='Benchmark Function from 1~4')
parser.add_argument('--Total_iterations', type=int, default=1400, help='Total iterations')
parser.add_argument('--Repeat', type=int, default=50, help='Total times of repeated experiment')
parser.add_argument('--Record', type=str2bool, default=True, help='Whether record fitness curve')
opt = parser.parse_args()

try: os.mkdir(f'BF{opt.BF}')
except: pass

class DTPSO():
    def __init__(self, opt):
        self.curve = np.ones(opt.Total_iterations) # 用于保存fitness曲线
        self.Total_iterations = opt.Total_iterations
        self.dvc = torch.device(opt.dvc)
        self.G, self.N, self.D = 8, opt.N, opt.D # number of Groups, particles in goups, and particle dimension
        self.BF = opt.BF
        self.Search_range = torch.tensor([-600.,600.], device=self.dvc)# search space of the Particles
        self.arange_idx = torch.arange(self.G, device=self.dvc) # 索引常量

        # Inertia Initialization for 8 Groups
        self.w_init = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9], device=self.dvc).unsqueeze(-1).unsqueeze(-1)
        self.w_end = torch.tensor([0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3], device=self.dvc).unsqueeze(-1).unsqueeze(-1)
        self.w_delta = (self.w_init - self.w_end)/self.Total_iterations # (G,1,1)

        # Velocity Initialization for 8 Groups
        self.v_limit_ratio = torch.tensor([0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3],device=self.dvc).unsqueeze(-1).unsqueeze(-1) # (G,1,1)
        self.v_min = (self.v_limit_ratio * self.Search_range[0]) # (G,1,1)
        self.v_max = (self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio # (G,1,1)

        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.tensor([[1,1,1,1,1,1,1,1.],
                                   [2,1,2,2,2,2,1,1.],
                                   [1,1,2,2,1,1,2,2.],
                                   [1,2,1,1,2,2,2,2.]], device=self.dvc).unsqueeze(-1).unsqueeze(-1)
        self.Hyper[0] = self.w_init

        # R Matrix, (4,G,N,1)
        self.Random = torch.ones((4,self.G,self.N,1), device=self.dvc) #更新速度时在装载随机数

        # L Matrix, (4,G,N,D)
        self.Locate = torch.zeros((4,self.G,self.N,self.D), device=self.dvc)
        self.Locate[1:4] = self.uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]

        # K Matrix, (4,G,N,D)
        self.Kinmtc = torch.zeros((4,self.G,self.N,self.D), device=self.dvc) #[V,Pbest,Gbest,Tbest]
        self.Kinmtc[0] = self.uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) * self.v_init_ratio

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf

        if opt.compile: self.iterate = torch.compile(self.iterate)


    def uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low


    def iterate(self):
        for i in range(self.Total_iterations):
            ''' Step 1: 计算Fitness (G,N)'''

            if self.BF == 1: # Sphere function
                fitness = self.Locate[1].pow(2).sum(dim=-1)

            elif self.BF == 2: # Rosenbrock function
                fitness = (100*(self.Locate[1,:,:,1:self.D] - self.Locate[1,:,:,0:(self.D-1)].pow(2)).pow(2) +
                           (self.Locate[1,:,:,0:(self.D-1)]-1).pow(2)).sum(dim=-1)

            elif self.BF == 3:  # Rastrigin Function
                fitness = (self.Locate[1].pow(2) - 10*torch.cos(2*torch.pi*self.Locate[1])).sum(dim=-1) + self.D*10

            elif self.BF == 4:  # Griewank function
                fitness = 1 + self.Locate[1].pow(2).sum(dim=-1)/4000 - \
                          torch.prod(torch.cos(self.Locate[1]/((torch.arange(1,self.D+1,device=self.dvc).expand(self.G, self.N, self.D)).pow(0.5))), dim=-1)


            ''' Step 2: 更新Pbest, Gbest, Tbest 的 value 和 particle '''
            # Pbest
            P_replace = (fitness < self.Pbest_value) # (G,N)
            self.Pbest_value[P_replace] = fitness[P_replace] # 更新Pbest_value
            self.Kinmtc[1, P_replace] = self.Locate[1, P_replace] # 更新Pbest_particles

            # Gbest
            values, indices = fitness.min(dim=-1)
            G_replace = (values < self.Gbest_value) # (G,)
            self.Gbest_value[G_replace] = values[G_replace] # 更新Gbest_value
            self.Kinmtc[2, G_replace] = (self.Locate[2, self.arange_idx, indices][G_replace].unsqueeze(1))

            # Tbest
            min_fitness = fitness.min()
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value
                flat_idx = fitness.argmin()
                self.Kinmtc[3] = (self.Locate[3, flat_idx//self.N, flat_idx % self.N]).clone() #这里必须clone, 否则数据联动

            ''' Step 3: 更新速度 '''
            self.Hyper[0] -= self.w_delta # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围

            '''打印并保存曲线'''
            if ((i+1)%500 == 0) and (not opt.Record): print('Iteration:{},  logMinvalue:{}'.format(i+1, torch.log(self.Tbest_value)))
            if opt.Record: self.curve[i] = torch.log(self.Tbest_value)



if __name__ == '__main__':
    if opt.Record: curves = np.zeros((opt.Repeat,opt.Total_iterations)) #(50,1400)
    torch.set_default_dtype(torch.float64)
    total_time = 0
    for _ in range(opt.Repeat):
        print('第{}轮实验:'.format(_))
        dtpso = DTPSO(opt)

        t0 = time.time()
        dtpso.iterate()
        total_time += (time.time() - t0)

        if opt.Record: curves[_] = dtpso.curve
    if opt.Record: np.save('BF{}/DTPSO-50.npy'.format(opt.BF), curves)
    print('实验次数:{}, 总时间:{}, 单轮时间:{}, Compile:{}, 运行设备:{}'.format(opt.Repeat, total_time, total_time/opt.Repeat, opt.compile, opt.dvc))



    # 保存实验配置
    Setups = {'BF{}'.format(opt.BF) : BF_name[opt.BF-1],
              'G,N,D' : '{},{},{}'.format(8, opt.N, opt.D),
              'Search Range' : [-600.,600.],
              'device' : opt.dvc,
              'compile' : opt.compile,
              'Total iterations' : opt.Total_iterations,
              'Repeat' : opt.Repeat,
              'Total time' : total_time,
              'Single time' : total_time/opt.Repeat}

    Exp_name = 'BF{}/DTPSO_R{}_{}'.format(opt.BF, opt.Repeat, opt.dvc)
    if opt.compile: Exp_name += 'Compile.txt'
    else: Exp_name += '.txt'
    save_dict_to_txt(Setups, Exp_name)
