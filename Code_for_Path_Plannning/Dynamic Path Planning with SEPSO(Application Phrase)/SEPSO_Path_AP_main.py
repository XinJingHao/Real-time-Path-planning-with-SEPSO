import argparse
from collections import deque
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import torch
import numpy as np
import time


# Seed Everything
seed_value = 1
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
parser.add_argument('--N', type=int, default=170, help='Total Number of Particles in one Group')
parser.add_argument('--D', type=int, default=20, help='Dimension of the Particle, NP=D/2')
parser.add_argument('--Max_iterations', type=int, default=50, help='Maximal iterations')
parser.add_argument('--render', type=str2bool, default=True, help='render the search progress')
parser.add_argument('--TrucWindow', type=int, default=20, help='Truncation window lenth: the lager,the longer,the better')
parser.add_argument('--dynamic_map', type=str2bool, default=True, help='dynamic or stastic map')
parser.add_argument('--SEPSO', type=str2bool, default=True, help='Whether use self-evolved params')
parser.add_argument('--AT', type=str2bool, default=True, help='Whether use Auto Truncation')
parser.add_argument('--PI', type=str2bool, default=True, help='Whether use Priori Initialization')
parser.add_argument('--FPS', type=int, default=0, help='Frames Per Second when rendering, 0 for fastest')
opt = parser.parse_args()

device = torch.device(opt.dvc)

class DTPSO_Path_Plan():
    def __init__(self, opt):
        self.Max_iterations = opt.Max_iterations
        self.dvc = device
        self.G, self.N, self.D = 8, opt.N, opt.D # number of Groups, particles in goups, and particle dimension
        self.Search_range = [5., 360.]  # search space of the Particles
        self.arange_idx = torch.arange(self.G, device=self.dvc) # 索引常量
        self.TrucWindow = opt.TrucWindow
        self.FPS = opt.FPS

        # Obstacle Initialization
        self.dynamic_map = opt.dynamic_map
        self.window_size = 366
        self.Obs_Segments = torch.load('Generate_Obstacle_Segments/Obstacle_Segments.pt').to(self.dvc) #(M,2,2) or (4*O,2,2)
        self.O = self.Obs_Segments.shape[0] // 4 # 障碍物数量
        self.Grouped_Obs_Segments = self.Obs_Segments.reshape(self.O,4,2,2) #注意Grouped_Obs_Segments 和 Obs_Segments 是联动的
        self.Obs_V = torch.randint(-5,5,(self.O,1,1,2), device=self.dvc) #每个障碍物的x方向速度和y方向速度
        self.static_obs = 2
        self.Obs_V[-self.static_obs:]*=0 # 最后两个障碍物不动
        self.Obs_X_limit = torch.tensor([46,320], device=self.dvc)
        self.Obs_Y_limit = torch.tensor([0, 366], device=self.dvc)
        self.Start_V = -3
        self.Target_V = 8

        # Path Initialization
        self.NP = int(opt.D/2) # 每个粒子所代表的路径的端点数量
        self.S = self.NP-1 # 每个粒子所代表的路径的线段数量
        self.P = self.G*self.N*self.S # 所有粒子所代表的路径的线段总数量
        self.x_start, self.y_start = opt.start, opt.start # 起点坐标
        self.x_target, self.y_target = opt.target, opt.target # 终点坐标
        self.rd_area = 0.5*(self.Search_range[1]-self.Search_range[0])/(self.S) # 按先验知识初始化粒子时的随机范围

        # Matrix Initialization
        self.Random = torch.zeros((4,self.G,self.N,1), device=self.dvc)
        self.Kinmtc = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)  # [V,Pbest,Gbest,Tbest]
        self.Locate = torch.zeros((4, self.G, self.N, self.D), device=self.dvc) # [0,X,X,X]

        self.render = opt.render
        if self.render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((self.window_size, self.window_size))

            # 障碍物图层
            self.map_pyg = pygame.Surface((self.window_size, self.window_size))

        self.AT = opt.AT
        self.PI = opt.PI
        self.ExpName = 'SEPSO_' if opt.SEPSO else 'DTPSO_'
        self.ExpName += 'AT_' if opt.AT else 'noAT_'
        self.ExpName += 'PI_' if opt.PI else 'noPI_'

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def Reset(self, Priori_X, ratio, params=None):
        params = params.to(self.dvc)
        '''重置算法超参数, 初始化粒子群的动力学特征'''
        # Inertia Initialization for 8 Groups
        self.w_init = params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (params[0:self.G]*params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)
        self.w_delta = (self.w_init - self.w_end)/self.Max_iterations # (G,1,1)

        # Velocity Initialization for 8 Groups
        self.v_limit_ratio = params[(2*self.G):(3*self.G)].unsqueeze(-1).unsqueeze(-1) # (G,1,1)
        self.v_min = -(self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_max = (self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio # (G,1,1)

        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = params[(3*self.G):(4*self.G)]
        self.Hyper[2] = params[(4*self.G):(5*self.G)]
        self.Hyper[3] = params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

        self.ReLocate(Priori_X, ratio) # 根据Priori_X初始化粒子群的位置


    def ReLocate(self, Priori_X, ratio):
        '''初始化粒子群动力学特征 '''
        # L Matrix, (4,G,N,D), 根据先验知识Priori_X(可以取上次迭代的Tbest)来初始化粒子群
        self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]
        # 每个Group中前RN个粒子在Priori_X附近初始化
        RN = int(ratio*self.N)
        self.Locate[1:4, :, 0:RN] = Priori_X + self._uniform_random(-self.rd_area, self.rd_area, (RN,self.D)) # (3,G,RN,D)
        # 限制位置至合法范围
        self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1])
        self._fix_apex()

        # K Matrix, (4,G,N,D)
        self.Kinmtc[0] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) * self.v_init_ratio

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf

        # Auto-Truncation Mechanism
        self.Tbest_Free = False
        self.Tbest_values_deque = deque(maxlen=self.TrucWindow)


    def _fix_apex(self):
        '''固定路径的首末端点'''
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
        return Flag1 * Flag2.T

    def _get_Obs_Penalty(self):
        # 将粒子群转化为线段，并展平为(G*N*S,2,2)
        particles = self.Locate[1].clone()  # (G,N,D)
        start_points = torch.stack((particles[:,:,0:(self.NP-1)], particles[:,:,self.NP:(2*self.NP-1)]), dim=-1) #(G,N,S,2), S条线段的起点坐标
        end_points = torch.stack((particles[:,:,1:self.NP], particles[:,:,(self.NP+1):2*self.NP]), dim=-1) #(G,N,S,2), S条线段的终点坐标
        Segments = torch.stack((start_points, end_points),dim=-2) #(G,N,S,2,2), (G,N,S)条线段的端点坐标
        flatted_Segments = Segments.reshape((self.P,2,2)) # (G*N*S,2,2), 将所有粒子展平为G*N*S条线段

        # 将展平后的粒子群线段 与 地图中的障碍物边界线段 进行跨立实验，得到交叉矩阵
        Intersect_Matrix = self._Is_Seg_Ingersection_PtoM(flatted_Segments, self.Obs_Segments) # (P,M)

        return Intersect_Matrix.sum(dim=-1).reshape((self.G,self.N,self.S)).sum(dim=-1) #(G,N)

    def _get_fitness(self):
        # fitness = lenth + Obs_penalty
        self.lenth = torch.sqrt((self.Locate[1,:,:,0:(self.NP-1)] - self.Locate[1,:,:,1:self.NP]).pow(2) +
                             (self.Locate[1,:,:,self.NP:(2*self.NP-1)] - self.Locate[1,:,:,(self.NP+1):(2*self.NP)]).pow(2)).sum(dim=-1)
        self.Obs_penalty = self._get_Obs_Penalty()
        return self.lenth + 30 * (self.Obs_penalty) ** 4

    def iterate(self):
        for i in range(self.Max_iterations):
            if self.AT and (i>0.2*self.TrucWindow) and self.Tbest_Free and (np.std(self.Tbest_values_deque)<10):
                return self.Tbest_value.item(), (i+1) # Path Lenth, Iterations per Planning

            ''' Step 1: 计算Fitness (G,N)'''
            fitness = self._get_fitness()

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
            idx_g, idx_n = flat_idx//self.N, flat_idx%self.N
            min_fitness = fitness[idx_g, idx_n]
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness # 更新Tbest_value
                self.Kinmtc[3] = (self.Locate[3, idx_g, idx_n]).clone() #这里必须clone, 否则数据联动
                # 查看Tbest所代表的路径是否collision free:
                if self.Obs_penalty[idx_g, idx_n] == 0: self.Tbest_Free = True
                else: self.Tbest_Free = False
            self.Tbest_values_deque.append(self.Tbest_value.item())


            ''' Step 3: 更新速度 '''
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围
            self._fix_apex()

        return self.Tbest_value.item(), (i + 1)  # Path Lenth, Iterations per Planning

    def _update_map(self):
        ''' 更新起点、终点、障碍物'''
        # 起点运动:
        self.y_start += self.Start_V
        if self.y_start > self.Search_range[1]:
            self.y_start = self.Search_range[1]
            self.Start_V *= -1
        if self.y_start < self.Search_range[0]:
            self.y_start = self.Search_range[0]
            self.Start_V *= -1

        # 终点运动:
        self.y_target += self.Target_V
        if self.y_target > self.Search_range[1]:
            self.y_target = self.Search_range[1]
            self.Target_V *= -1
        if self.y_target < self.Search_range[0]:
            self.y_target = self.Search_range[0]
            self.Target_V *= -1

        # 障碍物运动
        self.Grouped_Obs_Segments += self.Obs_V
        Flag_Vx = ((self.Grouped_Obs_Segments[:, :, :, 0] < self.Obs_X_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 0] > self.Obs_X_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vx, :, :, 0] *= -1
        Flag_Vy = ((self.Grouped_Obs_Segments[:, :, :, 1] < self.Obs_Y_limit[0]) |
                   (self.Grouped_Obs_Segments[:, :, :, 1] > self.Obs_Y_limit[1])).any(dim=-1).any(dim=-1)
        self.Obs_V[Flag_Vy, :, :, 1] *= -1

    def _render_frame(self):
        # 画障碍物
        self.map_pyg.fill((255, 255, 255))
        for _ in range(self.Grouped_Obs_Segments.shape[0]):
            obs_color = (50, 50, 50) if _ < (self.O-self.static_obs) else (225, 100, 0)
            pygame.draw.polygon(self.map_pyg, obs_color, self.Grouped_Obs_Segments[_,:,0,:].cpu().int().numpy())
        self.canvas.blit(self.map_pyg, self.map_pyg.get_rect())

        # 画路径
        for p in range(self.NP - 1):
            pygame.draw.line(
                self.canvas,
                (0, 0, 255),
                (self.Kinmtc[3, 0, 0, p].item(), self.Kinmtc[3, 0, 0, p + self.NP].item()),
                (self.Kinmtc[3, 0, 0, p + 1].item(), self.Kinmtc[3, 0, 0, p + self.NP + 1].item()),
                width=4)

        # 画起点、终点
        pygame.draw.circle(self.canvas, (255, 0, 0), (self.x_start, self.y_start), 5)  # 起点
        pygame.draw.circle(self.canvas, (0, 255, 0), (self.x_target, self.y_target), 5)  # 终点

        self.window.blit(self.canvas, self.map_pyg.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.FPS)

    def DynamiclyPlan(self, params):
        # 创建先验初始化位置
        Priori_X = (self.x_start + (self.x_target - self.x_start) * torch.arange(self.NP, device=self.dvc) / (self.NP - 1))
        Priori_X = torch.cat((Priori_X, Priori_X))
        self.Reset(Priori_X.clone(), ratio=0.1, params=params)

        counter, PL, IPP, TPP = 0, 0, 0, 0
        while True:
            counter += 1
            t0 = time.time()
            pl, ipp = self.iterate() # Path Lenth, Iterations per Planning
            PL += pl
            IPP += ipp
            TPP += (time.time()-t0) # Time per Planning

            if counter % 100 == 0:
                print(f'PL:{round(PL/counter,1)}, TPP:{round(TPP/counter,5)}, IPP:{round(IPP/counter,1)}, Exp:{self.ExpName}{counter}')


            if self.render: self._render_frame()
            if self.dynamic_map: self._update_map()

            if self.PI: self.ReLocate(self.Kinmtc[3,0,0].clone(), ratio=0.25) # 这里ratio越大，每次Iteration越快，但也更容易陷入局部最优
            else: self.Reset(Priori_X.clone(), ratio=0.25, params=params)



if __name__ == '__main__':
    opt.start = 20 # 起点X,Y坐标
    opt.target = 350 # 终点X,Y坐标
    dtpso = DTPSO_Path_Plan(opt)

    if opt.SEPSO:
        # Envolved params:
        params = torch.load('SEPSO_Bestparams_E10.pt')
    else:
        # Rudimentary params:
        params = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9,
                               0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3,
                               0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3,
                               2, 1, 2, 2, 2, 2, 1, 1.,
                               1, 1, 2, 2, 1, 1, 2, 2.,
                               1, 2, 1, 1, 2, 2, 2, 2.])


    dtpso.DynamiclyPlan(params)



