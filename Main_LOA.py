import numpy as np
from torch import randperm


def rand():
    return np.random.random((1, 1))

def fit(inp):
    f_val=np.random.random((len(inp),1))
    return f_val
def liongen(xmin,xmax,siz):
    lion = xmin + ((xmax-xmin)*np.random.random((1,siz)))
    return lion
def mutation (cub, rate, xmin, xmax):
    # print(cub)
    len1=len(cub[0,:])# (cub[0,:].shape[1])
    point = np.max([1,int(np.round(rate*float(len1)))])
    new_cub=[]
    for j in range(0,len(cub[:,1])):#= 1:numel(cub[:,1].numel())
        new_cub.append(cub[j, :])
        for i in range(0,point):#= 1:point
            if i==0:
                new_cub=np.asarray(new_cub)
            mut_point = int(np.asarray(np.round((((len(cub[0,:]))-1)*np.random.random((1,1))))))
            new_cub[j,mut_point]= xmin[0,mut_point]+(xmax[0,mut_point]-xmin[0,mut_point])*float(np.random.random((1,1)))

    new_cub = np.round(new_cub)
    return new_cub
def lion(init_lion,xmin,xmax,max_eval):
    xmin = xmin[0,:].reshape(1, xmin[0,:].shape[0])
    xmax = xmax[0,:].reshape(1, xmax[0,:].shape[0])

    Mature_age = 3
    max_strength = 3
    gmax = 10
    mutation_rate = 0.15
    Max_age = 3
    eval = 0
    siz = init_lion.shape[1]
    init_fit = fit(init_lion)
    init_val = np.argsort(init_fit.T)
    init_val_1=init_val[0]
    male_in = init_lion[init_val_1[0],:]
    female_in = init_lion[init_val_1[1],:]

    ########Nomadic lion Generation##########
    nomadic_pool = 0*np.ones([2,siz],dtype = float)
    nomadic_fit = 0*np.ones([2,1],dtype = float)#zeros(1, 2)
    nomadic_pool[0,:] = liongen(xmin, xmax, siz)
    SD=nomadic_pool[0,:].reshape(1,nomadic_pool.shape[1])
    nomadic_fit[0, 0]= fit(SD)
    # nomadic_fit[0, 0]=fval
    eval = eval + 1
    mat = 0
    gen = 0
    final = []
    plot_eval = []
    trial = 0
    flag = 0
    #########  Initial Lion Generation  ######
    plot_eval = [plot_eval,1]
    male_in=male_in.reshape(1,male_in.shape[0])
    female_in=female_in.reshape(1,female_in.shape[0])
    male_fit = fit(male_in)
    female_fit = fit(female_in)
    trial_fit = male_fit
    eval = eval + 2
    ########   Iterative Process  ##############
    for mx in range(0,max_eval):
        ########  Age   Declaration  ############
        if (trial_fit == male_fit) or (trial_fit < male_fit):
            trial = trial + 1
        else:
            trial = 0
            trial_fit = male_fit
        ######### Fertility Evaluation  ##########
        if mat > max_strength:
            f = 0
            g = 0
            while (f == 0) and  (g < gmax):
                g = g + 1
                new_female = female_in
                point1 = np.asarray(randperm(siz)).T
                point= point1[0]
                val_1=xmax[0,point]
                val_2=(xmin[0,point])
                try:
                    val_3=( female_in[0, point] + (-0.05 + 0.1 * rand()) * (male_in[0, point] - rand() * female_in[0, point]))
                except:
                    try:
                        male_in = male_in.reshape(1, male_in.shape[0])
                        val_3 = (female_in[0, point] + (-0.05 + 0.1 * rand()) * (male_in[0, point] - rand() * female_in[0, point]))
                    except:
                        female_in = female_in.reshape(1, female_in.shape[0])
                        val_3 = (female_in[0, point] + (-0.05 + 0.1 * rand()) * (male_in[0, point] - rand() * female_in[0, point]))

                val_4=np.max([(val_2),int(val_3)])
                val_5=np.min([(val_1),(val_4)])
                new_female[0, point] = val_5#np.min([xmax[0,point], np.max(float(xmin[0,point]),( female_in[0, point] + (-0.05 + 0.1 * rand()) * (male_in[0, point] - rand() * female_in[0, point])))])
                new_fit = fit(new_female)
                eval = eval + len(new_fit)
                if new_fit < female_fit:
                    f = 1
                    female_in = new_female
                    female_fit = new_fit
                    mat = 0
        ######   Lion mating  ##########
        ###### Crossover   ############
        cubpool = 0*np.ones([8,siz],dtype = float)#zeros(8, siz)
        FV=(cubpool.shape[0]) / 2
        for i in range(0,int(FV)):
            vect = np.asarray(randperm(siz))#, randi(siz - 1))
            vect = vect.reshape(1, vect.shape[0])
            cubpool[i,:] = female_in
            for j in range(0,len(vect)):
                try:
                    cubpool[i, vect[0,j]] = male_in[0,vect[0,j]]
                except:
                    male_in = male_in.reshape(1, male_in.shape[0])
                    cubpool[i, vect[0, j]] = male_in[0, vect[0, j]]


        ###########  Mutation    ############
        for i in range(0,int(FV)) :
            cub_vect = cubpool[i,:].reshape(1, cubpool[i,:].shape[0])
            nc = mutation(cub_vect, mutation_rate, xmin, xmax)
            cubpool[4 + i,:]=nc
        mat = mat + 1
        cub_age = 0
        ##########    Finding male and female cubs    ################
        fit1 = fit(cubpool)
        idx1 = np.argsort(fit1.T)
        idx = idx1[0]
        eval = eval + len(fit1)
        male_cub = cubpool[idx[0],:]
        female_cub = cubpool[idx[1],:]
        cubpool_malefit = float(fit1[idx[0]])
        cubpool_femalefit =float( fit1[idx[1]])
        ##########Territorial Defense##############
        flag1 = 0
        while flag1 == 0:
            cub_age = cub_age + 1
            male_rate = 0.15
            female_rate = 0.15
            try:
                male_cub = male_cub.reshape(1, male_cub.shape[0])
            except:
                male_cub=male_cub
            s_male = mutation(male_cub, male_rate, xmin, xmax)
            s_malefit = fit(s_male)
            eval = eval + len(s_malefit)
            if cubpool_malefit > s_malefit:
                male_cub = s_male
                cubpool_malefit = s_malefit
            try:
                female_cub = female_cub.reshape(1, female_cub.shape[0])
            except:
                female_cub=female_cub
            s_female = mutation(female_cub, female_rate, xmin, xmax)
            s_femalefit = fit(s_female)
            eval = eval + fit(s_femalefit)
            if cubpool_malefit > s_femalefit:
                female_cub = s_female
                cubpool_femalefit = s_femalefit
            ########   Entrance of Nomadic Lion   ###########
            ####### Nomadic  Lion Generation########
            if trial > Max_age: #% Checking laggardness rate
                nomadic_lion = mutation(male_in, 1 - mutation_rate, xmin, xmax)
            else:
                nomadic_lion = liongen(xmin, xmax, siz)
            #########   New Survival Fight  #######
            nomadic_pool[1,:] = nomadic_lion
            Nom_pool = nomadic_pool[1,:].reshape(1, nomadic_pool[1,:].shape[0])
            nomadic_fit[1] = fit(Nom_pool)
            eval = eval + 1
            idx2 = np.argsort(nomadic_fit.T)
            ind = idx2[0]
            ind=ind[0]
            # [elig_fit ind] = min(nomadic_fit)
            elig_fit=nomadic_fit[ind]
            if (elig_fit < male_fit) and (elig_fit < cubpool_malefit) and (elig_fit < cubpool_femalefit):
                male_in = nomadic_pool[ind,:]
                male_fit = elig_fit
                cub_age = 0
                flag1 = 1
                if ind == 1:
                    nomadic_pool[0,:]=nomadic_pool[1,:]
                    nomadic_fit[0] = nomadic_fit[1]
            else:
                nomadic_distance1 = np.sqrt((1 / siz) *np.sum((nomadic_pool[0,:] - male_in)** 2))
                nomadic_distance2 = np.sqrt((1 / siz) * np.sum((nomadic_pool[1,:] - male_in)** 2))
                distance = [nomadic_distance1, nomadic_distance2]
                eval1 =np.exp(nomadic_distance1 / np.max(distance)) / (nomadic_fit[0] / np.max(nomadic_fit))
                if eval1 < np.exp(1):
                    nomadic_pool[0,:]=nomadic_pool[1,:]
                    nomadic_fit[0] = nomadic_fit[1]
                    # Nom_pool = nomadic_pool[1, :].reshape(1, nomadic_pool[1, :].shape[0])
            nom_pool= nomadic_pool[0,:].reshape(1, nomadic_pool[0,:].shape[0])
            nomadic = mutation(nom_pool, 0.15, xmin, xmax)
            tmp_fit = fit( nomadic)
            eval = eval + len(tmp_fit)
            if tmp_fit < nomadic_fit[0]:
                nomadic_pool[0,:]=nomadic
                nomadic_fit[0] = tmp_fit
            ##########   Survival Fight   ###########
            ########### Territorial Takeover  ######
            if cub_age > Mature_age:
                gen = gen + 1
                flag1 = 1
                if male_fit > cubpool_malefit:
                    male_in = male_cub
                    male_fit = cubpool_malefit
                old_female = female_in
                if female_fit > cubpool_femalefit:
                    female_in = female_cub
                    female_fit = cubpool_femalefit
                if np.sum(female_in) != np.sum(old_female):
                    mat = 0
        final.append(male_fit)
        gbest = male_in
    final=np.sort(np.asarray(final))[::-1]
    best = min(final)
    # plot_eval = [plot_eval ,eval]
    return [best,final,gbest]



##############################################
######   Lion Optimization Algorithm   ################
#The Lion's Algorithm: a new nature-inspired search algorithm#
#######B. R. Rajakumar##########################
##############################################
M,N,max_eval=3,10,1000
init_lion=np.random.random((M,N))
xmin,xmax=0*np.ones([1,N],dtype = int),1*np.ones([1,N],dtype = int)
[best,final,gbest]=lion(init_lion,xmin,xmax,max_eval)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
fig = plt.figure()
ax = plt.axes()
ax.plot(range(0,1000), final, color='g')
plt.title("Fitness Curve(LOA)")
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.show()





