import matplotlib.pyplot as plt
import numpy as np

'''
注意！！！
文件夹BF1~BF4中的
SEPSO取的是log10（更容易看出差距）
PSO, DPPSO, DTPSO都是取的ln, 因此画之前需要转化一下
'''

x_limit = 1400 # DTPSO的最大显示区间
plt.figure(figsize=(25, 5))
fsize = 20
name = ['Sphere Function','Rosenbrock Function','Rastrigin Function','Griewank Function']


for i in range(1,5):
    plt.subplot(1, 4, i)

    # PSO
    X=np.load("BF{}/PSO-50.npy".format(i))
    X = np.log10(np.exp(X)) #ln 转 log10
    X=X[:,0:x_limit] # (50,1400)
    T=X.shape[1] # 1400

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T,1)
    tt = t/1000
    plt.plot(tt,mean,color="sandybrown",label='PSO')
    plt.fill_between(tt,max,min, facecolor="peachpuff",alpha=0.8)


    # DPPSO
    X=np.load("BF{}/DPPSO-50.npy".format(i))
    X = np.log10(np.exp(X)) #ln 转 log10
    X=X[:,0:x_limit*8]
    N=8#粒子维度
    T=X.shape[1]#迭代次数
    X=X[:,np.arange(0,T,step=N)]

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T/N,1)
    tt = t/1000
    plt.plot(tt,mean,color="dodgerblue",label='DPPSO')
    plt.fill_between(tt,max,min, facecolor="royalblue",alpha=0.2)


    # DTPSO
    X=np.load("BF{}/DTPSO-50.npy".format(i))
    X = np.log10(np.exp(X)) #ln 转 log10
    X=X[:,0:x_limit]
    T=X.shape[1]

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T,1)
    tt = t/1000
    plt.plot(tt,mean,color="fuchsia",label='DTPSO')
    plt.fill_between(tt,max,min, facecolor="fuchsia",alpha=0.1)


    # SEPSO
    X=np.load("BF{}/SEPSO-R50.npy".format(i))
    X=X[:,0:x_limit]
    T=X.shape[1]

    max=np.max(X,axis=0)
    min=np.min(X,axis=0)
    mean=np.mean(X,axis=0)

    t=np.arange(0,T,1)
    tt = t/1000
    plt.plot(tt,mean,color="black",label='SEPSO')
    plt.fill_between(tt,max,min, facecolor="black",alpha=0.15)



    plt.xticks(size = fsize)
    plt.yticks(size = fsize)
    plt.xlabel("Iterations (K steps)",fontsize=fsize)
    plt.ylabel("Fitness (log)",fontsize=fsize)
    plt.title(name[i-1],fontsize=fsize)
    plt.tight_layout()
    plt.grid()




plt.legend(fontsize=fsize-3)
plt.savefig('BF_all_iteration.pdf',bbox_inches="tight")
# plt.show()



