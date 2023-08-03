import torch

class SEPSO_Down():
    def __init__(self, BF, SaveCurve=False): # params在传入前需要clone,
        ''' SEPSO_Down是Self-evolving PSO的底层算法, 实际上是使用的DTPSO
            核心功能（self.evaluate_params）: 输入一组DTPSO的超参数params, 让DTPSO迭代, 并用最终的Tbest_value作为该组params的fitness
            params[0:8] = w_init; params[8:16] = w_end; params[16:24] = v_limit_ratio;
            params[24:32] = Hyper[1]; params[32:40] = Hyper[2]; params[40:48] = Hyper[3]
            注意, params在传入前需要clone以防止数据联动. '''
        self.BF = BF
        self.Total_iterations = 1400
        self.dvc = torch.device('cpu')
        self.G, self.N, self.D = 8, 10, 30 # number of Groups, particles in goups, and particle dimension
        self.Search_range = torch.tensor([-600.,600.], device=self.dvc)# search space of the Particles
        self.arange_idx = torch.arange(self.G, device=self.dvc) # 索引常量

        # R Matrix, (4,G,N,1)
        self.Random = torch.ones((4,self.G,self.N,1), device=self.dvc) #更新速度时在装载随机数

        self.SaveCurve = SaveCurve #是否保存Tbest_value曲线
        if self.SaveCurve: self.curve = torch.zeros(1400, device=self.dvc)

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low)*torch.rand(shape, device=self.dvc) + low

    def reset(self, params):
        params = params.to(self.dvc)
        '''Reset the parameters and the particles of the DTPSO'''
        # Inertia Initialization for 8 Groups
        self.w_init = params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (params[0:self.G]*params[self.G:(2*self.G)]).unsqueeze(-1).unsqueeze(-1)  # 0.2, 0.4, 0.1, 0.6, 0.1, 0.5, 0.1, 0.3
        self.w_delta = (self.w_init - self.w_end)/self.Total_iterations # (G,1,1)

        # Velocity Constraint Initialization for 8 Groups
        self.v_limit_ratio = params[(2*self.G):(3*self.G)].unsqueeze(-1).unsqueeze(-1) # (G,1,1)
        self.v_min = (self.v_limit_ratio * self.Search_range[0]) # (G,1,1)
        self.v_max = (self.v_limit_ratio * self.Search_range[1]) # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio # (G,1,1)

        # H Matrix, (4,G=8,1,1)
        self.Hyper = torch.ones((4,self.G), device=self.dvc)
        self.Hyper[1] = params[(3*self.G):(4*self.G)]
        self.Hyper[2] = params[(4*self.G):(5*self.G)]
        self.Hyper[3] = params[(5*self.G):(6*self.G)]
        self.Hyper.unsqueeze_(-1).unsqueeze_(-1)

        # L Matrix, (4,G,N,D)
        self.Locate = torch.zeros((4,self.G,self.N,self.D), device=self.dvc)
        self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) #[0,X,X,X]

        # K Matrix, (4,G,N,D)
        self.Kinmtc = torch.zeros((4,self.G,self.N,self.D), device=self.dvc) #[V,Pbest,Gbest,Tbest]
        self.Kinmtc[0] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G,self.N,self.D)) * self.v_init_ratio

        # Best Value initialization
        self.Pbest_value = torch.ones((self.G,self.N), device=self.dvc) * torch.inf
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * torch.inf
        self.Tbest_value = torch.inf


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
            self.Hyper[0] = self.w_init - self.w_delta*i  # 惯性因子衰减
            self.Random[1:4] = torch.rand((3,self.G,self.N,1), device=self.dvc) #装载随机数
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0) #(G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max) # 限制速度范围

            ''' Step 4: 更新位置 '''
            self.Locate[1:4] += self.Kinmtc[0] # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1]) # 限制位置范围

            if self.SaveCurve: self.curve[i] = torch.log10(self.Tbest_value)

            '''打印'''
            # if ((i+1)%100 == 0): print('Iteration:{},  logMinvalue:{}'.format(i+1, torch.log10(self.Tbest_value)))

    def evaluate_params(self, params):
        '''传入DTPSO的超参数, 迭代后, 返回该超参数找到的Tbest'''
        self.reset(params)
        self.iterate()
        return self.Tbest_value.item()

# 用于验证SEPSO_Down.py是否可以运行
# params = torch.tensor([0.4, 0.7, 0.8, 0.8, 0.2, 0.9, 0.4, 0.9, # w_init
#                        0.5, 0.57, 0.125, 0.75, 0.5, 0.555, 0.25, 0.333, # w_end_ratio
#                        0.2, 0.1, 0.6, 0.4, 0.3, 0.5, 0.8, 0.3, # v_limit_ratio
#                        2,1,2,2,2,2,1,1, # C1
#                        1,1,2,2,1,1,2,2, # C2
#                        1,2,1,1,2,2,2,2], dtype=torch.float) # C3
# DTPSO = SEPSO_Down(BF=1)
# print(DTPSO.evaluate_params(params))