import argparse
import time
from collections import deque
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import torch
import numpy as np


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
parser.add_argument('--dvc', type=str, default='cpu', help='Running devices')
parser.add_argument('--compile', type=str2bool, default=False, help='Whether use torch.compile')
parser.add_argument('--N', type=int, default=170, help='Total Number of Particles in one Group')
parser.add_argument('--D', type=int, default=20, help='Dimension of the Particle, NP=D/2')
parser.add_argument('--Max_iterations', type=int, default=100, help='Maximal iterations')
parser.add_argument('--render', type=str2bool, default=True, help='render the search progress')
parser.add_argument('--TrucWindow', type=int, default=20, help='Truncation window lenth: the lager,the longer,the better')
opt = parser.parse_args()

device = torch.device(opt.dvc)

class DTPSO_Path_Plan():
    def __init__(self, opt):
        self.Max_iterations = opt.Max_iterations
        self.dvc = device
        self.G, self.N, self.D = 8, opt.N, opt.D # number of Groups, particles in goups, and particle dimension
        self.Search_range = torch.tensor([5.,360.], device=self.dvc)# search space of the Particles
        self.arange_idx = torch.arange(self.G, device=self.dvc) # 索引常量

        # Map Initialization
        self.window_size = 366
        self.Obs_Segments = torch.load('Generate_Obstacle_Segments/Obstacle_Segments.pt').to(self.dvc) #(M,2,2) or (4*O,2,2)
        self.O = self.Obs_Segments.shape[0] // 4 # 障碍物数量
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,4,2,2) #注意Grouped_Obs_Segments 和 Obs_Segments 是联动的

        # Path Initialization
        self.NP = int(opt.D/2) # 每个粒子所代表的路径的端点数量
        self.S = self.NP-1 # 每个粒子所代表的路径的线段数量
        self.P = self.G*self.N*self.S # 所有粒子所代表的路径的线段总数量
        self.x_start, self.y_start = opt.start, opt.start # 起点坐标
        self.x_target, self.y_target = opt.target, opt.target # 终点坐标
        self.rd_area = 0.5*(opt.target-opt.start)/(self.NP-1) # 按先验知识初始化粒子时的随机范围

        # R Matrix, (4,G,N,1)
        self.Random = torch.ones((4,self.G,self.N,1), device=self.dvc) #更新速度时在装载随机数

        self.render = opt.render
        if self.render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((self.window_size, self.window_size))

            # 障碍物图层
            self.map_pyg = pygame.Surface((self.window_size, self.window_size))
            self.map_pyg.fill((255, 255, 255))
            for _ in range(self.Grouped_Obs_Segments.shape[0]):
                obs_color = (50, 50, 50) if _ < (self.O - 2) else (225, 100, 0)
                pygame.draw.polygon(self.map_pyg, obs_color, self.Grouped_Obs_Segments[_, :, 0, :].cpu().int().numpy())
            self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low


    def Reset(self, Priori_X, ratio):
        self.t0 = time.time()
        # Auto-Truncation Mechanism
        self.Obs_Free = False
        self.Tbest_values_deque = deque(maxlen=opt.TrucWindow)

        # Inertia Initialization for 8 Groups
        self.w_init = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9], device=self.dvc).unsqueeze(-1).unsqueeze(-1)
        self.w_end = torch.tensor([0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3], device=self.dvc).unsqueeze(-1).unsqueeze(-1)
        self.w_delta = (self.w_init - self.w_end)/self.Max_iterations # (G,1,1)

        # Velocity Initialization for 8 Groups
        self.v_limit_ratio = torch.tensor([0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3],device=self.dvc).unsqueeze(-1).unsqueeze(-1) # (G,1,1)
        self.v_min = -(self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_max = (self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio # (G,1,1)

        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.tensor([[1,1,1,1,1,1,1,1.],
                                   [2,1,2,2,2,2,1,1.],
                                   [1,1,2,2,1,1,2,2.],
                                   [1,2,1,1,2,2,2,2.]], device=self.dvc).unsqueeze(-1).unsqueeze(-1)

        # L Matrix, (4,G,N,D)
        self.Locate = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)
        self._position_init(Priori_X, ratio)

        # K Matrix, (4,G,N,D)
        self.Kinmtc = torch.zeros((4,self.G,self.N,self.D), device=self.dvc) #[V,Pbest,Gbest,Tbest]
        self.Kinmtc[0] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) * self.v_init_ratio

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf


    def _position_init(self, Priori_X, ratio):
        '''每个Group中前RN个粒子根据在Priori_X附近初始化, 其它随机初始化
            X 是 torch.tensor shape=(D,); 0<ratio<1'''

        # 粒子位置随机初始化
        self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]

        # 每个Group中前RN个粒子在Priori_X附近初始化
        RN = int(ratio*self.N)
        self.Locate[1:4, :, 0:RN] = Priori_X + self._uniform_random(-self.rd_area, self.rd_area, (RN,self.D))  # (3,G,RN,D)

        # 限制位置至合法范围
        self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1])

        # 固定首位点
        self.Locate[1:4, :, :, 0] = self.x_start
        self.Locate[1:4, :, :, self.NP] = self.y_start
        self.Locate[1:4, :, :, self.NP - 1] = self.x_target
        self.Locate[1:4, :, :, 2 * self.NP - 1] = self.y_target

    def _Cross_product_for_VectorSet(self, V_PM, V_PP):
        '''计算 向量集V_PM 和  向量集V_PP 的交叉积 (x0*y1-x1*y0)
            V_PM = torch.tensor((p, m, 2, 2))
            V_PP = torch.tensor((p,2))
            Output = torch.tensor((p, m, 2))'''
        return V_PM[:, :, :, 0] * V_PP[:, 1, None, None] - V_PM[:, :, :, 1] * V_PP[:, 0, None, None]

    def _Is_Seg_Ingersection_PtoM(self, P, M):
        '''利用[交叉积-跨立实验]判断 线段集P 与 线段集M 的相交情况
            P = torch.tensor((p,2,2))
            M = torch.tensor((m,2,2))
            Output = torch.tensor((p,m)), dtype=torch.bool'''

        V_PP = P[:, 1] - P[:, 0]  # (p, 2),自身向量
        V_PM = M - P[:, None, None, 0]  # (p, m, 2, 2), 自身起点与其他线段端点构成的向量
        Flag1 = self._Cross_product_for_VectorSet(V_PM, V_PP).prod(dim=-1) < 0  # (p, m)

        V_MM = M[:, 1] - M[:, 0]  # (m, 2)
        V_MP = P - M[:, None, None, 0]  # (m, p, 2, 2)
        Flag2 = self._Cross_product_for_VectorSet(V_MP, V_MM).prod(dim=-1) < 0  # (m, p)

        Intersection = Flag1 * Flag2.T

        return Intersection

    def _get_Obs_Penalty(self):
        # Step 1: 将粒子群转化为线段，并展平为(G*N*S,2,2)
        particles = self.Locate[1].clone()  # (G,N,D)
        start_points = torch.stack((particles[:,:,0:(self.NP-1)], particles[:,:,self.NP:(2*self.NP-1)]), dim=-1) #(G,N,S,2), S条线段的起点坐标
        end_points = torch.stack((particles[:,:,1:self.NP], particles[:,:,(self.NP+1):2*self.NP]), dim=-1) #(G,N,S,2), S条线段的终点坐标
        Segments = torch.stack((start_points, end_points),dim=-2) #(G,N,S,2,2), (G,N,S)条线段的端点坐标
        flatted_Segments = Segments.reshape((self.P,2,2)) # (G*N*S,2,2), 将所有粒子展平为G*N*S条线段

        Intersect_Matrix = self._Is_Seg_Ingersection_PtoM(flatted_Segments, self.Obs_Segments) # (P,M)

        return Intersect_Matrix.sum(dim=-1).reshape((self.G,self.N,self.S)).sum(dim=-1) #(G,N)


    def iterate(self):
        for i in range(self.Max_iterations):
            if self.Obs_Free and (np.std(self.Tbest_values_deque)<5):
                print(f'Averate Iteration Time:{(time.time()-self.t0)/(i+1)}')
                break # Auto-Truncation
            # 目前还有bug: 当Tbest是有障碍的，但存在无障碍的路径并且陷入局部陷阱时，会以Tbest有障碍而退出

            ''' Step 1: 计算Fitness (G,N)'''
            # fitness = lenth + Obs_penalty
            lenth = torch.sqrt((self.Locate[1,:,:,0:(self.NP-1)] - self.Locate[1,:,:,1:self.NP]).pow(2) +
                                 (self.Locate[1,:,:,self.NP:(2*self.NP-1)] - self.Locate[1,:,:,(self.NP+1):(2*self.NP)]).pow(2)).sum(dim=-1)
            Obs_penalty = self._get_Obs_Penalty()
            fitness = lenth + 30*(Obs_penalty)**4


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
            flat_idx = fitness.argmin()
            idx0, idx1 = flat_idx//self.N, flat_idx%self.N
            if Obs_penalty[idx0, idx1] == 0: self.Obs_Free = True # Tbest's path is obstacle free
            else: self.Obs_Free = False
            min_fitness = fitness[idx0, idx1]
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value
                flat_idx = fitness.argmin()
                self.Kinmtc[3] = (self.Locate[3, flat_idx//self.N, flat_idx % self.N]).clone() #这里必须clone, 否则数据联动
            self.Tbest_values_deque.append(self.Tbest_value.item())


            ''' Step 3: 更新速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围
            # 固定首位点
            self.Locate[1:4, :, :, 0] = self.x_start
            self.Locate[1:4, :, :, self.NP] = self.y_start
            self.Locate[1:4, :, :, self.NP - 1] = self.x_target
            self.Locate[1:4, :, :, 2 * self.NP - 1] = self.y_target

            '''打印并保存曲线'''
            print(f'Iteration:{i+1},  logMinvalue:{torch.log10(self.Tbest_value)}, TbestValueStd:{np.std(self.Tbest_values_deque)}')

            if self.render:
                self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())
                for p in range(self.NP - 1):
                    pygame.draw.line(
                        self.canvas,
                        (255, 0, 0),
                        (self.Kinmtc[3,0,0,p].item(), self.Kinmtc[3,0,0,p + self.NP].item()),
                        (self.Kinmtc[3,0,0,p + 1].item(), self.Kinmtc[3,0,0,p + self.NP + 1].item()),
                        width=4)

                self.window.blit(self.canvas, self.map_pyg.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(50)




if __name__ == '__main__':
    opt.start = 20 # 起点X,Y坐标
    opt.target = 350 # 终点X,Y坐标
    dtpso = DTPSO_Path_Plan(opt)

    # 创建先验初始化位置
    Priori_X = (opt.start + (opt.target - opt.start)*torch.arange(dtpso.NP,device=device)/(dtpso.NP-1))
    Priori_X = torch.cat((Priori_X,Priori_X))

    t0 = time.time()
    dtpso.Reset(Priori_X, ratio=0.1)
    dtpso.iterate()
    print('Time per planning:',time.time()-t0)

    torch.save(dtpso.Kinmtc[3,0,0], 'Tbest.pt')

    import draw_Tbest # 画Tbest


