import matplotlib.pyplot as plt
import numpy as np


x_limit = 1400 # 最大显示区间

# For G8,N10,D30,Repeat50,Total_iterations1400
runtime = np.array([[3.97502, 0.64748, 0.37395], #BF1: PSO, DPPSO, DTPSO
                    [5.98614, 0.92159, 0.44433], #BF2: PSO, DPPSO, DTPSO
                    [6.20295, 0.95855, 0.42341], #BF3: PSO, DPPSO, DTPSO
                    [6.91236, 1.08262, 0.45757]]) #BF4: PSO, DPPSO, DTPSO




plt.figure(figsize=(25, 5))
fsize = 20
name = ['Sphere Function','Rosenbrock Function','Rastrigin Function','Griewank Function']
for i in range(1,5):
    plt.subplot(1, 4, i)

    # PSO
    X=np.load("BF{}/PSO-50.npy".format(i))
    X = np.log10(np.exp(X))  # ln 转 log10
    X=X[:,0:x_limit] # (50,1400)
    T=X.shape[1] # 1400

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T,1)
    tt = runtime[i-1,0]*t*1000/x_limit
    plt.plot(tt,mean,color="sandybrown",label='PSO')
    plt.fill_between(tt,max,min, facecolor="peachpuff",alpha=0.8)
    plt.xticks(np.arange(int(tt[-1]))[::50])  # 设置x轴显示间隔为50


    # DPPSO
    X=np.load("BF{}/DPPSO-50.npy".format(i))
    X = np.log10(np.exp(X))  # ln 转 log10
    X=X[:,0:x_limit*8]
    N=8#粒子维度
    T=X.shape[1]#迭代次数
    X=X[:,np.arange(0,T,step=N)]

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T/N,1)
    tt = runtime[i-1,1]*t*1000/x_limit
    plt.plot(tt,mean,color="dodgerblue",label='DPPSO')
    plt.fill_between(tt,max,min, facecolor="royalblue",alpha=0.2)
    plt.xticks(np.arange(int(tt[-1]))[::50])  # 设置x轴显示间隔为50


    # DTPSO
    X=np.load("BF{}/DTPSO-50.npy".format(i))
    X = np.log10(np.exp(X))  # ln 转 log10
    X=X[:,0:x_limit]
    T=X.shape[1]

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T,1)
    tt = runtime[i-1,2]*t*1000/x_limit
    plt.plot(tt,mean,color="fuchsia",label='DTPSO')
    plt.fill_between(tt,max,min, facecolor="fuchsia",alpha=0.1)
    plt.xticks(np.arange(int(tt[-1]))[::50])  # 设置x轴显示间隔为50


    plt.xlim(0, runtime[i-1,2]*t[-1]*1000/x_limit)
    plt.xticks(size = fsize)
    plt.yticks(size = fsize)
    plt.xlabel("Walltime (ms)",fontsize=fsize)
    plt.ylabel("Fitness (log)",fontsize=fsize)
    plt.title(name[i-1],fontsize=fsize)
    plt.tight_layout()
    plt.grid()




plt.legend(fontsize=fsize-3)
plt.savefig('BF_all_time.pdf',bbox_inches="tight")
# plt.show()


