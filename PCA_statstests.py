import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import ttest_ind, f_oneway
import PCA_preprocessing
from scipy.spatial.distance import mahalanobis

def projectdata(PC, data_mat):
    projected_data = np.matmul((np.transpose(data_mat)), PC)
    return projected_data

def PCA_analysis(l_data_PCA):
    k = 3  # nb of dimensions
    l_projected_data = []
    var = []
    data_mat = l_data_PCA.copy()
    for i in range(l_data_PCA.shape[0]):
        data_mat[i,:] = (l_data_PCA[i,:] - np.mean(l_data_PCA[i,:]))/(np.std(l_data_PCA[i,:])) #StandardScaler().fit_transform(l_data_PCA) # l_data_PCA
    #data_mat = l_data_PCA.copy()
    print(data_mat.shape)
    cov_mat = np.cov(data_mat)
    #print(cov_mat)
    eigval, eigvec = np.linalg.eig(cov_mat)
    # plt.plot(eigval)
    # plt.show()
    eigval_copy = np.copy(eigval)
    eigvec_copy = np.copy(eigvec)
    idx = eigval.argsort()[::-1] # ??
    eigvec_sorted = eigvec[:, idx]
    eigval_sorted = eigval[idx]
    PC = eigvec_sorted[:, :]
    projected_data = projectdata(PC, data_mat)
    eigval_sorted_normalised = eigval_sorted/np.sum(eigval_sorted)
    l_projected_data.append(projected_data)
    var.append(eigval_sorted_normalised)
    print("Dimension of projected data:")
    print(projected_data.shape)
    return projected_data, eigval_sorted_normalised, PC, eigvec_copy, eigval_copy

def circularshift_persess(neural):
    st_time = -3.25
    ed_time = 3.25
    binsize = 0.05 # sec
    trialnums = neural.shape[0]
    timepointnum = neural.shape[1]
    neural_sh = np.zeros(neural.shape)
    for i in range(trialnums):
        rng = (random.random())
        rng_shift = ((rng*(ed_time -(st_time)) + st_time)) # -5 to 5 sec rng
        rng_shift_timepoints = int((rng_shift - st_time)/binsize)
        if rng_shift_timepoints == 0:
            neur = neural[i,:,:]
        else:
            neur = np.concatenate((neural[i,rng_shift_timepoints:,:],neural[i,:rng_shift_timepoints,:]), axis=0)

        neural_sh[i,:,:] = neur
        
    return neural_sh

def binshuffle_15min(l_data):
    np.random.seed(42)
    l_data_sh = []
    
    for i in range(len(l_data)):
        neural = np.array(l_data[i])
        summ = np.sum(neural)
        timep = l_data[i].shape[0]
        freq = 30000
        t_ignore = int(30*0.05*30000)
        timeb_total = int(timep*freq*0.05) - t_ignore
        y = np.zeros((timeb_total, neural.shape[-1])).astype(int)
        for i in range(neural.shape[-1]):
            summ = np.sum(neural[:,i])
            t = (timeb_total*1)
            y[:,i] = np.random.binomial(n=1, p=summ/t, size=(timeb_total)).astype(int)
        new= np.zeros((neural.shape[0], neural.shape[-1]))
        c=int(30000*0.05)
        #print(new.shape)
        for i in range(new.shape[0]-1):
            new[i,:] =np.sum(y[i*c:(i+1)*c,:], axis=0)
        
        l_data_sh.append(new)
        del new
    return l_data_sh


def binshuffle_250ms_allunits(l_data):
    np.random.seed(42)
    l_data_new = [None]*len(l_data)
    for i in range(len(l_data)):
        l_data_new[i]=np.zeros(l_data[i].shape)
        for j in range(l_data[i].shape[0]):
            l_data_new[i][j,:] = np.sum(np.array(l_data[i][j:j+5,:]), axis=0)
    
    l_data_sh = []
    for i in range(len(l_data)):
        neural = np.array(l_data_new[i])
        summ = np.sum(neural)
        timep = l_data_new[i].shape[0]
        freq = 30000
        #t_ignore = int(30*0.05*30000)
        timeb_total = int(timep*freq*0.05) #- t_ignore
        y = np.zeros((timeb_total, neural.shape[-1]))
        for i in range(neural.shape[-1]):
            summ = np.sum(neural[:,i])
            t = (timeb_total)
            y[:,i] = np.random.binomial(n=1, p=summ/t, size=(timeb_total))
        new= np.zeros((neural.shape[0]-30+1, neural.shape[-1]))
        c=int(30000*0.05)
        #print(new.shape)
        for i in range(new.shape[0]):
            new[i,:] =np.sum(y[i*c:(i+1)*c,:], axis=0)
        l_data_sh.append(new)

    return l_data_sh


def binshuffle_15min_allsh(l_data):
    np.random.seed(42)
    l_data_sh = []
    for i in range(len(l_data)):
        neural = np.array(l_data[i])
        summ = np.sum(neural)
        timep = l_data[i].shape[0]
        freq = 30000
        t_ignore = int(30*0.05*30000)
        timeb_total = int(timep*freq*0.05) - t_ignore
        y = np.zeros((timeb_total, neural.shape[-1]))
        for i in range(neural.shape[-1]):
            #summ = np.sum(neural[:,i])
            t = (timeb_total*neural.shape[-1])
            y[:,i] = np.random.binomial(n=1, p=summ/t, size=(timeb_total))
        new= np.zeros((neural.shape[0]-30+1, neural.shape[-1]))
        c=int(30000*0.05)
        #print(new.shape)
        for i in range(new.shape[0]):
            new[i,:] =np.sum(y[i*c:(i+1)*c,:], axis=0)
        l_data_sh.append(new)
        del new
    return l_data_sh

def binshuffle(neural):
    np.random.seed(100)
    neural = np.array(neural)
    sum = np.sum(neural)
    timep = neural.shape[0]
    freq  = 30000
    timeb_total =int(timep*freq*0.05)
    y = np.zeros((timeb_total, neural.shape[-1]))
    for i in range(neural.shape[-1]):
        sum = np.sum(neural[:,i])
        t = (timeb_total*neural.shape[-1])
        y[:,i] = np.random.binomial(n=1, p=sum/t, size=(timeb_total))
    new= np.zeros(neural.shape)
    c=int(30000*0.05)
    #print(new.shape)
    for i in range(neural.shape[0]):
        new[i,:] =np.sum(y[i*c:(i+1)*c,:], axis=0)

    return new


def shuffletrialstypesandnums(allmice_dividedtrials, notignoredsessions, seed):
    random.seed(seed)
    newtrialnum_trialtype = [None]*len(allmice_dividedtrials)
    for i in range(0,5,1):
        newtrialnum_trialtype[i] = [None]*len(allmice_dividedtrials[i])
        for j in range(len(notignoredsessions)):
            newtrialnum_trialtype[i][j]=[]
            newtrialnum_trialtype[i][j] = np.zeros((allmice_dividedtrials[i][notignoredsessions[j]].shape[0], 2))
            for k in range(len(newtrialnum_trialtype[i][j])):
                newtrialnum_trialtype[i][j][k,0] = np.random.choice([0,2,4]) #int(random.randint(0,len(allmice_dividedtrials)-1))
                #print(newtrialnum_trialtype[i][j][k,0])
                #print(allmice_dividedtrials[int(newtrialnum_trialtype[i][j][k,0])][notignoredsessions[j]].shape[0])
                newtrialnum_trialtype[i][j][k,1] = int(random.randint(0,-1+int(allmice_dividedtrials[int(newtrialnum_trialtype[i][j][k,0])][notignoredsessions[j]].shape[0])))
    
    return newtrialnum_trialtype

'''
def metrics_generator_perdataset(l_proj_5trials_sh, allmice_neural_avgtrials_sh_zscored, bigarr_sh, unitspermouse, unitspermouse_cumsum,metric_type="length"):


    l_proj_5trials_1mouse_sh = [None]*len(unitspermouse)
    var_5trials1mouse_sh = [None]*len(unitspermouse)
    bigarr_excludeonemouse_sh = [None]*len(unitspermouse)
    for j in range(len(unitspermouse)):
        startidxunit = unitspermouse_cumsum[j]
        endidxunit = unitspermouse_cumsum[j]
        bigarr_statstests_sh = np.array([])
        for k in range(len(unitspermouse)):
            if k != j:
                bigarr_onemouse_sh = np.array(allmice_neural_avgtrials_sh_zscored[0][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:])
                for i in range(1, len(allmice_neural_avgtrials_sh_zscored)):
                    bigarr_onemouse_sh = np.hstack((bigarr_onemouse_sh,allmice_neural_avgtrials_sh_zscored[i][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:]))
                if bigarr_statstests_sh.size == 0:
                    bigarr_statstests_sh = bigarr_onemouse_sh
                else: 
                    bigarr_statstests_sh = np.vstack((bigarr_statstests_sh, bigarr_onemouse_sh))

        bigarr_excludeonemouse_sh[j] = bigarr_statstests_sh

    
    # Li type analysis :

    vectorsTrials_sh = [None]*5 # for each trial type
    l_traj_stats_sh = []
    l_traj_sh,_,_,eigvec_sh, eigval_sh = PCA_analysis(l_data_PCA=bigarr_sh)

    for i in range(len(bigarr_excludeonemouse_sh)):
        deleteeigenvalsvecs_sh =np.arange(unitspermouse_cumsum[i],unitspermouse_cumsum[i+1])#np.arange(eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1]),eigvec.shape[0]) #
        eigvec_copy_sh = np.copy(eigvec_sh)
        eigenval_copy_sh = np.copy(eigval_sh)
        eigenvec_stats_sh = np.delete(eigvec_copy_sh, deleteeigenvalsvecs_sh, axis=0) #eigvec_copy[:eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1]),:eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1])]#np.delete(eigvec_copy, deleteeigenvalsvecs, axis=0)
        #print(eigenvec_stats.shape)
        #eigenvec_stats = # np.delete(eigenvec_stats, deleteeigenvalsvecs, axis=1)
        eigval_stats_sh = np.delete(eigenval_copy_sh, deleteeigenvalsvecs_sh)
        idx_argsorteigval_sh = np.argsort(eigval_stats_sh)[::-1]
        eigenvec_stats_sh = eigenvec_stats_sh[:,idx_argsorteigval_sh]
        #print(eigenvec_stats_sh.shape)
        #print(bigarr_excludeonemouse_sh[i].shape)
        l_traj_bigsinglesess_sh = projectdata(data_mat=bigarr_excludeonemouse_sh[i], PC=eigenvec_stats_sh)[:,:3] #PCA_analysis(bigarr_excludeonemouse[i]) #
        l_traj_stats_sh.append(l_traj_bigsinglesess_sh)

    metrics_sh,_,_ = obtain_metrics(l_traj_stats=l_traj_stats_sh, finalunits_entryexit= allmice_neural_avgtrials_zscored, unitspermouse=unitspermouse, metric_type=metric_type)

    return metrics_sh, l_traj_sh,bigarr_excludeonemouse_sh, eigvec_sh, eigval_sh, l_traj_stats_sh
'''

def unitspermouse_excludeonesessgenerator(l_data, ignoreunits,excludetrials,ignoresessions, allmice_dividedtrials, allmice_neural_avgtrials_zscored):

    unitspermouse = []
    unitspermouse_cumsum = [0]
    ignoresessions = [] #[2,8,9,10]


    c=0

    for i in range(len(l_data)):
        if i in ignoresessions:
            continue
        else:
            k=0
            
            for j in range(len(ignoreunits)):
                if ignoreunits[j] >= unitspermouse_cumsum[c] and ignoreunits[j] < unitspermouse_cumsum[c]+allmice_dividedtrials[0][i].shape[-1]:
                    k+=1 
            
            unitspermouse.append(allmice_dividedtrials[0][i].shape[-1]-k)
            unitspermouse_cumsum.append(unitspermouse_cumsum[c]+unitspermouse[c])
            c+=1


    # makes the exclude one mouse list of data for stats tests 

    bigarr_excludeonemouse = [None]*len(unitspermouse)
    bigarr_onlyone = [None]*len(unitspermouse)
    for j in range(len(unitspermouse)):
        startidxunit = unitspermouse_cumsum[j]
        endidxunit = unitspermouse_cumsum[j]
        
        bigarr_statstests = np.array([])
        for k in range(len(unitspermouse)):
            if k != j:
                bigarr_onemouse = np.array(allmice_neural_avgtrials_zscored[0][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:])
                for i in range(1, len(allmice_neural_avgtrials_zscored)):
                    bigarr_onemouse = np.hstack((bigarr_onemouse, allmice_neural_avgtrials_zscored[i][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:]))
                if bigarr_statstests.size == 0:
                    bigarr_statstests = bigarr_onemouse
                else: 
                    bigarr_statstests = np.vstack((bigarr_statstests, bigarr_onemouse))
            else:
                bigarr_onemouse = np.array(allmice_neural_avgtrials_zscored[0][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:])
                for i in range(1, len(allmice_neural_avgtrials_zscored)):
                    bigarr_onemouse = np.hstack((bigarr_onemouse, allmice_neural_avgtrials_zscored[i][unitspermouse_cumsum[k]:unitspermouse_cumsum[k+1],:]))
                bigarr_onlyone[j] = bigarr_onemouse

        bigarr_statstests = np.delete(bigarr_statstests, excludetrials, axis=1)
        bigarr_excludeonemouse[j] = bigarr_statstests

    return unitspermouse, unitspermouse_cumsum, bigarr_excludeonemouse


def project_excludeonemouse(bigarr, bigarr_excludeonemouse,unitspermouse_cumsum, finalunits_entryexit, type_trial="real"):
    # Li type analysis :
    vectorsTrials = [None]*5 # for each trial type
    l_traj_stats = []
    l_traj_5trials_newPCAfunct, var_5trials_newPCAfunct ,pc_original_newPCAfunct,eigvec, eigval = PCA_analysis(l_data_PCA=bigarr)

    for i in range(len(bigarr_excludeonemouse)):
        deleteeigenvalsvecs =np.arange(unitspermouse_cumsum[i],unitspermouse_cumsum[i+1])#np.arange(eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1]),eigvec.shape[0]) #
        eigvec_copy = np.copy(eigvec)
        eigenval_copy = np.copy(eigval)
        eigenvec_stats =np.delete(eigvec_copy, deleteeigenvalsvecs, axis=0) #eigvec_copy[:eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1]),:eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1])]#np.delete(eigvec_copy, deleteeigenvalsvecs, axis=0)
        #print(eigenvec_stats.shape)
        #eigenvec_stats = # np.delete(eigenvec_stats, deleteeigenvalsvecs, axis=1)
        eigval_stats = np.delete(eigenval_copy, deleteeigenvalsvecs)
        idx_argsorteigval = np.argsort(eigval_stats)[::-1]
        eigenvec_stats = eigenvec_stats[:,idx_argsorteigval]
        print(eigenvec_stats.shape)
        print(bigarr_excludeonemouse[i].shape)
        l_traj_bigsinglesess = projectdata(data_mat=bigarr_excludeonemouse[i], PC=eigenvec_stats)[:,:] #PCA_analysis(bigarr_excludeonemouse[i]) #PCA_analysis(bigarr_onlyone[i])[0] #
        l_traj_stats.append(l_traj_bigsinglesess)

        c=0
        len_trial = (finalunits_entryexit[0].shape[1]) # time
        c1 = len_trial # 121
        colours = ['red', 'green', 'orange', 'blue', "black"]
        leg = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]

        #fig = plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')

        for k in range(len(finalunits_entryexit)):
            sig = 3
            #print(c)
            if k == 1 or k == 3:
                continue
            
            x =l_traj_stats[i][c:c1,0] #gaussian_filter1d(l_traj_stats[i][c:c1,0], sigma=sig)
            y =l_traj_stats[i][c:c1,1] #gaussian_filter1d(l_traj_stats[i][c:c1,1], sigma=sig)
            z =l_traj_stats[i][c:c1,2] #gaussian_filter1d(l_traj_stats[i][c:c1,2], sigma=sig)
            mid = int(x.size/2)
            plt.plot(x,y,z, label = leg[k], c = colours[k])
            ax.scatter3D(x[0], y[0], z[0], c='k',s=10,marker="o")
            ax.scatter3D(x[mid], y[mid], z[mid], c='k',s=10,marker="v")
            #ax.set_xlim([-7,7])
            #ax.set_ylim([-7,7])
            #ax.set_zlim([-7,7])
            c=c1
            c1 += len_trial

        plt.legend()
        plt.title('PCA, 1 sess exclusion')
        plt.savefig('RESULTS/NOT-INJECTED/PCA_finalpicturesNew/Stats_PCA_figures/PCA_allmice_allunits_EPM_definedtrials_entryexit_'+str(i)+type_trial+'.png')
        plt.show()
    return l_traj_stats,eigvec, eigval

def mahalanobis_manual(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()



def mahalanobis_distance(l_traj_stats, trialnums = 3, pc=3, datatype="real"):
   
    len_trial = int((l_traj_stats[0].shape[0]/trialnums))
    dist_metrics = np.zeros((len(l_traj_stats), trialnums))
    mid = int(len_trial/2)
    baseline_t = int(0/0.05)
    for i in range(len(l_traj_stats)):
        for j in range(trialnums):
            traj = l_traj_stats[i][j*len_trial+mid:(j+1)*len_trial-5, :pc]
            ref = l_traj_stats[i][j*len_trial+5:(j)*len_trial+mid+baseline_t, :pc]#
            # ref =(l_traj_stats[i][0*len_trial:(0)*len_trial+mid, :pc] + l_traj_stats[i][1*len_trial:(1)*len_trial+mid, :pc] + l_traj_stats[i][2*len_trial:(2)*len_trial+mid, :pc])/3

            # if j == 2:
            #     ref = l_traj_stats[i][1*len_trial+5:(1)*len_trial+mid, :pc]
            #print(ref.shape)
            act = traj[:, :pc]
            #matrix_cov = np.vstack((ref, act)).T
            #print(matrix_cov.shape)
            ##inv_cov = np.linalg.inv(np.cov(matrix_cov))
            #print(inv_cov.shape)

            # mah_inst = mahalanobis_manual(x=act, data=ref)
            # mah_inst_zscore = (mahalanobis_manual(x=act, data=ref) - np.mean(ref))/np.std(ref)
            
            # dist_metrics[i,j] = np.mean(mah_inst_zscore)
            dist_metrics[i,j] = np.mean(mahalanobis_manual(x=act, data=ref))

        # ref = np.linalg.norm(l_traj_stats[i][0*len_trial:(1)*len_trial, :pc], axis=1)
        # act = np.linalg.norm(l_traj_stats[i][1*len_trial:(2)*len_trial, :pc], axis=1)
        # matrix_cov = np.vstack((ref, act)).T
        # inv_cov = np.linalg.inv(np.cov(matrix_cov))
        # #print(inv_cov)
        # dist_metrics[i,0] = mahalanobis(u=act, v=ref, VI=inv_cov)
        # ref = np.linalg.norm(l_traj_stats[i][0*len_trial:(1)*len_trial, :pc], axis=1)
        # act = np.linalg.norm(l_traj_stats[i][2*len_trial:(3)*len_trial, :pc], axis=1)
        # matrix_cov = np.vstack((ref, act)).T
        # inv_cov = np.linalg.inv(np.cov(matrix_cov))
        # dist_metrics[i,1] = mahalanobis(u=act, v=ref, VI=inv_cov)
        # ref = np.linalg.norm(l_traj_stats[i][2*len_trial:(3)*len_trial, :pc], axis=1)
        # act = np.linalg.norm(l_traj_stats[i][1*len_trial:(2)*len_trial, :pc], axis=1)
        # matrix_cov = np.vstack((ref, act)).T
        # inv_cov = np.linalg.inv(np.cov(matrix_cov))
        # dist_metrics[i,2] = mahalanobis(u=act, v=ref, VI=inv_cov)


    means = np.mean(dist_metrics, axis=0)
    stdd = np.std(dist_metrics, axis=0)
    xlabels =["OA Entry", "Headdips", "CA Entry"] #["OA Entry", "Headdips", "OA Exit", "CA Entry", "CA Exit"]
    print(means)
    colours = ['red', 'darkorange','dimgrey']#['red', 'orange', 'lightcoral', 'dimgrey', 'darkgrey'] #['red', 'orange','green', 'blue', "black"]
    
    order_plot = [0,2,1]
    x_pos = np.arange(len(order_plot))
    plt.bar(x=x_pos, height=means[order_plot],yerr=stdd[order_plot], align='center', alpha=0.5, ecolor='black', capsize=10, color=colours)
    plt.xticks(x_pos,xlabels)
    
    plt.ylabel('Mahalanobis Distance (A.U.)')
    
    # plt.ylim([0,200])
    cc=0
    for i in order_plot:
        #print(cc)
        plt.scatter(cc*np.ones(len(dist_metrics[:,i])),dist_metrics[:,i],c='k')
        cc+=1
    #plt.ylim([0,600])
    plt.savefig("RESULTS/NOT-INJECTED/PCA_finalpicturesNew/MahalanobisMetric_"+datatype+".svg")
    plt.show()
    return dist_metrics.T



def plot_PCA(l_traj, num_trialtypes, len_trial, saveas):
    c1 = len_trial # 121
    colours = ['red', 'lightcoral', 'dimgrey', 'darkgrey', 'orange'] #['red', 'orange','green', 'blue', "black"]
    leg = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]

    fig = plt.figure(figsize=(10,10),facecolor='white')
    ax = plt.axes(projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    for i in range(num_trialtypes):
        sig = 3
        print(c)
        if i ==5:
            continue
        x = gaussian_filter1d(l_traj[c:c1,0], sigma=sig)
        y = gaussian_filter1d(l_traj[c:c1,1], sigma=sig)
        z = gaussian_filter1d(l_traj[c:c1,2], sigma=sig)
        mid = int(x.size/2)
        plt.plot(x,y,z, label = leg[i], c = colours[i])
        ax.scatter3D(x[0], y[0], z[0], c='k',s=5,marker="o")
        ax.scatter3D(x[mid], y[mid], z[mid], c='k',s=5,marker="*")
        ax.scatter3D(x[-1], y[-1], z[-1], c='k',s=5,marker="*")
        ax.set_ylim([-5,5])
        ax.set_zlim([-5,5])
        ax.set_xlim([-5,5])
        #ax.set_facecolor('white')
        #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        c=c1
        c1 += len_trial

    plt.legend()
    plt.title('PCA - Shuffled')
    if saveas != False:
        plt.savefig(saveas) 
    plt.show()

def plot_metrics(metrics, saveas):
    means_metrics = []
    sem_metrics = []

    for i in range(len(metrics)):
        metrics[i] = np.array(metrics[i])
        arr = metrics[i][:,:1].squeeze()
        #print(arr)
        means_metrics.append(np.mean(arr))
        sem_metrics.append(sem(arr))
    means_metrics = np.array(means_metrics)
    sem_metrics = np.array(sem_metrics)
    colours = ['red', 'orange', 'lightcoral', 'dimgrey', 'darkgrey']
    #print(colours)
    xlabels = ["OA Entry","Headdips", "OA Exit", "CA Entry", "CA Exit"]
    order_plot = [0,4,1,2,3]
    x_pos = np.arange(len(xlabels))
    plt.bar(x=x_pos, height=means_metrics[order_plot],yerr=sem_metrics[order_plot], align='center', alpha=0.5, ecolor='black', capsize=10, color=colours)
    plt.xticks(x_pos,xlabels)
    plt.ylabel('Trajectory Lengths (A.U.)')
    cc=0
    for i in order_plot:
        #print(cc)
        plt.scatter(cc*np.ones(len(metrics[i][:,:1])), metrics[i][:,:1],c='k')
        cc+=1
    #plt.ylim([0,30])
    #plt.savefig("PCA_finalpictures/PCA_EPM_Definedtrials_entryexit_sh_statistics.svg")
    if saveas != False:
        plt.savefig(saveas) 
    #plt.scatter(np.ones(len(unitactivitypermouseSQ)), euclideandistanceQ, c='k')
    plt.show()
    #plt.scatter(np.ones(len(unitactivitypermouseSQ)), euclideandistanceQ, c='k')
    

def ttests(metrics):
    _,pval = ttest_ind(metrics[0][:,:1].squeeze(), metrics[-1][:,:1].squeeze())
    print("OA entry and headdips: "+str(pval))
    _,pval = ttest_ind(metrics[0][:,:1].squeeze(), metrics[1][:,:1].squeeze())
    print("OA entry and OA Exit: "+str(pval))
    _,pval = ttest_ind(metrics[0][:,:1].squeeze(), metrics[2][:,:1].squeeze())
    print("OA entry and CA entry: "+str(pval))
    _,pval = ttest_ind(metrics[0][:,:1].squeeze(), metrics[3][:,:1].squeeze())
    print("OA entry and CA exit: "+str(pval))
    _,pval = ttest_ind(metrics[2][:,:1].squeeze(), metrics[3][:,:1].squeeze())
    print("CA entry and CA exit: "+str(pval))
    _,pval = ttest_ind(metrics[2][:,:1].squeeze(), metrics[-1][:,:1].squeeze())
    print("CA entry and headdips: "+str(pval))
    _,pval = ttest_ind(metrics[1][:,:1].squeeze(), metrics[2][:,:1].squeeze())
    print("OA exit and CA entry: "+str(pval))
    _,pval = ttest_ind(metrics[2][:,:1].squeeze(), metrics[4][:,:1].squeeze())
    print("OA exit and CA exit: "+str(pval))
    _,pval = ttest_ind(metrics[1][:,:1].squeeze(), metrics[-1][:,:1].squeeze())
    print("OA exit and headdips: "+str(pval))
    _,pval = ttest_ind(metrics[3][:,:1].squeeze(), metrics[-1][:,:1].squeeze())
    print("CA exit and headdips: "+str(pval))



def obtain_metrics(finalunits_entryexit, l_traj_stats, unitspermouse, pc_90perc=3,metric_type="length", plot_stats=True, mid_select=False):
    # two types of metric_type : "length", "Euclidean"
    angles = [None]*3#len(finalunits_entryexit)
    distances = [None]*3#len(finalunits_entryexit)
    metrics = [None]*3#len(finalunits_entryexit)
    metrics_new = [None]*3#len(finalunits_entryexit)
    len_trial = (finalunits_entryexit[0].shape[1]) # time

    origin = [None]*len(unitspermouse)

    for i in range(len(unitspermouse)):
        startingpoints = np.zeros(pc_90perc)
        c=0
        c1 = len_trial
        for j in range(3):#len(finalunits_entryexit)): # 5 trials
            
            startingpoints += l_traj_stats[i][c:c+1,:pc_90perc].squeeze()#l_proj_normalised[i][c:c+1,:].squeeze()
            c += len_trial 
        origin[i] = startingpoints/len(finalunits_entryexit)

    c=0
    sig = 3
   
    for i in range(3):#len(finalunits_entryexit)): # 5 trials
        #print(c1)
        metrics_new[i] = [None]*3
        for k in range(3):
            metrics_new[i][k] = []

        for j in range(len(unitspermouse)):
            # print(i)
            if j==0:
                distances[i] = [None]*len(unitspermouse)
                angles[i] = [None]*len(unitspermouse)
                metrics[i]= [None]*len(unitspermouse)
                
            metrics[i][j] = [None]*4#np.zeros(3) # one for euclidean dist, one for angle, one for vector

            x = (np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,0], sigma=sig))))**2 # (np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,0], sigma=sig))))**2
            #x = (np.diff(l_traj_stats[j][c:c1,0]))**2
            mid = int(x.shape[0]/2)
            for ii in range(1,pc_90perc):
                x += np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,ii], sigma=sig)))**2
                #x+= (np.diff(l_traj_stats[j][c:c1,ii]))**2
            if mid_select:
                start_m = mid
            else:
                start_m = 0
                
            sum_1traj = np.sum(x[start_m:]**0.5)#np.sum((x**2 + y**2 + z**2)**0.5)
            metrics[i][j][0] = (sum_1traj)
            metrics_new[i][0].append(sum_1traj)

            #vec1 = l_traj_stats[i][c:c1,:].squeeze()
            #vec2 =[0,0,1] #origin[i]
            #vector = np.mean(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))) #np.mean(l_proj_normalised[i][c:c1,:].squeeze() - origin[i])/np.linalg.norm((l_proj_normalised[i][c:c1,:].squeeze() - origin[i]), axis = 0)
            x = (np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,0], sigma=sig))))**2 
            y = np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,1], sigma=sig)))**2
            z = np.abs(np.diff(gaussian_filter1d(l_traj_stats[j][c:c1,2], sigma=sig)))**2
            vec1 = np.array([np.mean(x), np.mean(y), np.mean(z)]) #l_traj_stats_sh[i][c:c1,:].squeeze()
            vec2 = [1,1,1]#origin_sh[i]
            vector = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) #np.mean(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))) #np.mean(l_proj_normalised[i][c:c1,:].squeeze() - origin[i])/np.linalg.norm((l_proj_normalised[i][c:c1,:].squeeze() - origin[i]), axis = 0)
            
            #metrics[i][j][1] =np.array(3) # vector # angle
            metrics_new[i][1].append(vector)

            # vector itself - take max point
            maxdist = 0
            maxdistvec=[]
            #for k in range(l_traj_stats[j][c:c1,0].shape[0]):
            #    dist = ((x[k]-origin[i][0])**2 + (y[k]-origin[i][1])**2 + (z[k]-origin[i][2])**2)**0.5
            #    if dist>maxdist:
            #        maxdist = dist
            #        maxdistvec = np.array([x[k], y[k], z[k]])

            #metrics[i][j][2] = maxdistvec
            #vectorr =  np.array([np.max(np.abs(gaussian_filter1d(l_traj_stats[j][c:c1,0], sigma=sig))), np.max(np.abs(gaussian_filter1d(l_traj_stats[j][c:c1,1], sigma=sig))), np.max(np.abs(gaussian_filter1d(l_traj_stats[j][c:c1,2], sigma=sig)))]) # np.array([np.mean(x), np.mean(y), np.mean(z)])
            sig =3
            vectorr = (np.abs(gaussian_filter1d(l_traj_stats[j][c:c1,0] - l_traj_stats[j][c,0], sigma=sig))) #(l_traj_stats[j][c:c1,0] - l_traj_stats[j][c,0])**2# origin[j][0])**2 #
            vecc = np.max(gaussian_filter1d(l_traj_stats[j][c:c1,0] - l_traj_stats[j][c,0], sigma=sig))
            #print(pc_90perc)
            half = int((c1-c)/2)
            for ii in range(1,pc_90perc):
                vectorr += (np.abs(gaussian_filter1d(l_traj_stats[j][c:c1,ii]- l_traj_stats[j][c,ii], sigma=sig))) #np.vstack((vectorr,l_traj_stats[j][c:c1,ii]))# 
                vecc =np.vstack((vecc, np.max(gaussian_filter1d(l_traj_stats[j][c:c1,ii] - l_traj_stats[j][c,ii], sigma=sig))))#  
                #vectorr += (l_traj_stats[j][c:c1,ii] - l_traj_stats[j][c,ii])**2#- origin[j][ii])**2 #-l_traj_stats[j][c:c1,ii]
            vectorr = (vectorr)**0.5
            vec_normalised = vectorr#/(np.linalg.norm(vectorr))
            #print(vecc)
            metrics_new[i][2].append(vecc)
            metrics[i][j][1:] = vecc[:3].squeeze()
            half = int(vectorr.shape[0]/2)
            #print(l_traj_stats[j].shape)
            #print(l_traj_stats[j][c+0,:])
            if metric_type == "Euclidean":
                #lenn = #np.linalg.norm((vectorr) - np.zeros(pc_90perc))
                #print(lenn)
                argmaxx = np.argmax(vectorr)
                #print(argmaxx)
                metrics[i][j][0] = np.linalg.norm((l_traj_stats[j][argmaxx,:pc_90perc] - l_traj_stats[j][0,:pc_90perc]))#origin[j]))#

            if metric_type == "Spread":
                vec = l_traj_stats[j][c:c1,:]
                dist = 0
                mid = int(vec.shape[0]/2)
                for jj in range(mid, vec.shape[0]):
                    y = np.linalg.norm(np.abs(vec[mid:,:] - vec[jj,:]), axis=1)
                    print(y)
                    maxdist = np.max(abs(y))
                    
                    if maxdist>dist:
                        dist = maxdist
                metrics[i][j][0]=dist
            



        c = c1
        c1 = c1+ len_trial

    #visualise distances and angles
    if plot_stats:
        means_metrics = []
        sem_metrics = []
        for i in range(len(metrics)):
            metrics[i] = np.array(metrics[i])
            arr = metrics[i][:,:1].squeeze()
            means_metrics.append(np.mean(arr))
            sem_metrics.append(np.std(arr))
        means_metrics = np.array(means_metrics)
        sem_metrics = np.array(sem_metrics)
        xlabels =["OA Entry", "Headdips", "CA Entry"] #["OA Entry", "Headdips", "OA Exit", "CA Entry", "CA Exit"]
        
        colours = ['red', 'orange','dimgrey']#['red', 'orange', 'lightcoral', 'dimgrey', 'darkgrey'] #['red', 'orange','green', 'blue', "black"]
        
        order_plot = [0,2,1]#,2,3]
        x_pos = np.arange(len(order_plot))
        plt.bar(x=x_pos, height=means_metrics[order_plot],yerr=sem_metrics[order_plot], align='center', alpha=0.5, ecolor='black', capsize=10, color=colours)
        plt.xticks(x_pos,xlabels)
        if metric_type == "length":
            plt.ylabel('Trajectory Lengths (A.U.)')
        else:
            plt.ylabel("Euclidean Distance (A.U.)")
        #plt.ylim([0,200])
        cc=0
        for i in order_plot:
            #print(cc)
            plt.scatter(cc*np.ones(len(metrics[i][:,:1])), metrics[i][:,:1],c='k')
            cc+=1
        #plt.ylim([0,9])
        #plt.savefig("PCA_finalpictures/PCA_EPM_Definedtrials_entryexit_statistics_euclidean.svg")
        #plt.scatter(np.ones(len(unitactivitypermouseSQ)), euclideandistanceQ, c='k')
        #plt.ylim([450,600])
        #plt.show()
        

        
    return metrics, metrics_new, origin


def newshufflepcagenerateperdataset_reduced(l_mouse_name, allmice_dividedtrials, l_data,l_data_25, l_beh_lowdim_cont,notignoredsessions = [0,1,2,3,4,5,6,7,8,9,10,11]):

    behaviour_trials_sh,neural_trials_sh, neural_trials_sh_25, _,_, beh_timepoints_sh  = PCA_preprocessing.get_raw_beh(l_data=l_data, l_beh_lowdim_cont=l_beh_lowdim_cont, l_data_25=l_data_25)

    _, allmice_neural_avgtrials_newsh, allmice_neural_avgtrials_newsh_25, allmice_dividedtrials_sh,allmice_dividedtrials_sh_25, _,_ = PCA_preprocessing.get_preprocessed_trials(l_mouse_name=l_mouse_name, behaviour_trials=behaviour_trials_sh, neural_trials=neural_trials_sh,beh_timepoints=beh_timepoints_sh,trialtype='entryexit',neural_trials_25=neural_trials_sh_25)

    # 50ms sliding window
    allmice_behavior_avgtrials_sliding_newsh = [None]*len(allmice_dividedtrials)
    len_trial = allmice_neural_avgtrials_newsh[0].shape[-1]-5
    len_trial_25 = allmice_neural_avgtrials_newsh_25[0].shape[-1]-10

    for i in range(len(allmice_neural_avgtrials_newsh_25)):
        allmice_neural_avgtrials_newsh_25[i] = allmice_neural_avgtrials_newsh_25[i][:,:len_trial_25]
    

    allmice_neural_avgtrials_newshuffle_zscored,ignoreunit = PCA_preprocessing.z_score_orderedhm(allmice_neural_avgtrials_newsh_25.copy(), window_bin=0.025)

    bigarr_newshuffle = (np.array(allmice_neural_avgtrials_newshuffle_zscored[0]))
    for i in range(1, len(allmice_neural_avgtrials_newshuffle_zscored)):
        #np.random.shuffle(np.array(allmice_neural_avgtrials_newshuffle_zscored[i][:,70]))
        bigarr_newshuffle = np.hstack((bigarr_newshuffle,allmice_neural_avgtrials_newshuffle_zscored[i]))
    excludetrials = np.concatenate((np.arange(len_trial_25,len_trial_25*2), np.arange(len_trial_25*3, len_trial_25*4)))
    bigarr_newshuffle =(np.delete(bigarr_newshuffle, excludetrials, axis=1))
    
    # np.random.shuffle(bigarr_newshuffle)

    #finalunits_entryexit_newshuffle = PCA_preprocessing.plotheatmap_orderOAentry(ehm_newshuffle, ihm_newshuffle, rhm_newshuffle, allmice_neural_avgtrials_newshuffle_zscored, trialtitle[0], trialsubtypes_entryexit)
    l_proj_5trials_newshuffle, var_5trials_newshuffle ,_ = PCA_preprocessing.PCA_analysis(l_data_PCA=(bigarr_newshuffle))
    
    #np.savez("shuffledPCAbootstrap500.npz", l_proj_5trials_newshuffle)

    return l_proj_5trials_newshuffle[:,:], allmice_neural_avgtrials_newshuffle_zscored, allmice_dividedtrials_sh_25, bigarr_newshuffle, ignoreunit, var_5trials_newshuffle

def newshufflepcagenerateperdataset(seed, l_mouse_name,finalunits_entryexit,allmice_dividedtrials,allmice_dividedtrials_beh, l_data,l_data_25, l_beh_lowdim_cont,notignoredsessions = [0,1,2,3,4,5,6,7,8,9,10,11]):
    np.random.seed(seed)
    print(seed)
    # allmice_dividedtrials, allmice_dividedtrials_beh
    
    trialnums = [None]*len(finalunits_entryexit)

    for i in range(len(trialnums)):
        trialnums[i] = [None]*len(allmice_dividedtrials[i])
        for j in range(len(allmice_dividedtrials[i])):
            trialnums[i][j] = allmice_dividedtrials[i][j].shape[0]

    # now use this trialnums data to generate random numbers from 0 to size(l_data)-3sec and select snippets
    windowlength = finalunits_entryexit[0].shape[1]+5
    allmice_dividedtrials_newshuffle = [None]*len(finalunits_entryexit)
    allmice_dividedtrials_newshuffle_25 = [None]*len(finalunits_entryexit)
    allmice_dividedtrialsbeh_newshuffle = [None]*len(finalunits_entryexit)
    
    randomidx = [None]*len(trialnums)

    # trialtype_trialnum_matrix = shuffletrialstypesandnums(allmice_dividedtrials, notignoredsessions, seed)

    ###########################################################################################################################
    # print(trialtype_trialnum_matrix)
    
    # for i in range(len(trialnums)):
    #     allmice_dividedtrials_newshuffle[i] = [None]*len(notignoredsessions)
    #     allmice_dividedtrialsbeh_newshuffle[i] = [None]*len(notignoredsessions)
    #     randomidx[i] = [None]*len(trialnums[i])
        
        
    #     for j in range(len(notignoredsessions)):
    #         #print(l_data[j].shape)
            
    #         neural_originaldata = l_data[notignoredsessions[j]]
    #         beh_originaldata = l_beh_lowdim_cont[notignoredsessions[j]]
    #         allmice_dividedtrials_newshuffle[i][j] = np.zeros((trialnums[i][notignoredsessions[j]],allmice_dividedtrials[i][notignoredsessions[j]].shape[1],allmice_dividedtrials[i][notignoredsessions[j]].shape[-1]))
    #         allmice_dividedtrialsbeh_newshuffle[i][j] = np.zeros((trialnums[i][notignoredsessions[j]],allmice_dividedtrials[i][notignoredsessions[j]].shape[1],2))
    #         # randomidx[i][j] = [None]*trialnums[i][notignoredsessions[j]]

            # for k in range(trialnums[i][notignoredsessions[j]]):
                
            #     randomidx[i][j][k] = random.randint(0, len(neural_originaldata)-windowlength)
            #     # if i==1 or i==3:
            #     #     randomidx[i][j][k] = random.randint(0, len(neural_originaldata)-windowlength)
            #     #     allmice_dividedtrials_newshuffle[i][j][k,:,:] = neural_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength]
            #     #     continue

            #     trialidxsh = int(trialtype_trialnum_matrix[i][j][k][0])
            #     trialnumidxsh =int(trialtype_trialnum_matrix[i][j][k][1])
            #     fr = np.sum(allmice_dividedtrials[trialidxsh][j][trialnumidxsh,:,:])

            #     allmice_dividedtrials_newshuffle[i][j][k,:,:] = np.random.poisson(lam=fr, size=[136,allmice_dividedtrials[trialidxsh][j][trialnumidxsh,:,:].shape[-1]]).astype(np.float64) #(neural_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength,:]) # neural_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength,:] #
                
            #     #(allmice_dividedtrials[trialidxsh][notignoredsessions[j]][trialnumidxsh,:,:])#
            #     # np.random.poisson(lam=fr, size=timepointnum[0]).astype(np.float64) ##neural_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength]# 
            #     #np.random.shuffle(allmice_dividedtrials_newshuffle[i][j][k,:,:])
            #     #allmice_dividedtrials_newshuffle[i][j][k,:,:] = neural_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength]
            #     #allmice_dividedtrialsbeh_newshuffle[i][j][k,:,:] = beh_originaldata[randomidx[i][j][k]:randomidx[i][j][k]+windowlength]

    #            # print(allmice_dividedtrials_newshuffle[i][j].shape)
    #         # allmice_dividedtrials_newshuffle[i][j] = circularshift_persess(neural = allmice_dividedtrials[i][notignoredsessions[j]])

    # # averaging these
    # allmice_neural_avgtrials_newsh = [None]*len(allmice_dividedtrials_newshuffle)

    # for i in range(len(trialnums)):
    #    #print(i)
    #     for j in range(len(notignoredsessions)):
    #         #print(allmice_dividedtrials_newshuffle[i][j])
    #         avg = np.mean(allmice_dividedtrials_newshuffle[i][j], axis=0) # time, units
    #         if j == 0:
    #             allmice_neural_avgtrials_newsh[i] = avg.T
    #         else:
    #             allmice_neural_avgtrials_newsh[i] = np.vstack((allmice_neural_avgtrials_newsh[i], avg.T))
    # # ######################################################################################################################################################


    behaviour_trials_sh,neural_trials_sh, neural_trials_sh_25, _,_, beh_timepoints_sh  = PCA_preprocessing.get_raw_beh(l_data=l_data, l_beh_lowdim_cont=l_beh_lowdim_cont, l_data_25=l_data_25)
    _, allmice_neural_avgtrials_newsh, allmice_neural_avgtrials_newsh_25, allmice_dividedtrials_sh,allmice_dividedtrials_sh_25, _,_ = PCA_preprocessing.get_preprocessed_trials(l_mouse_name=l_mouse_name, behaviour_trials=behaviour_trials_sh, neural_trials=neural_trials_sh,beh_timepoints=beh_timepoints_sh,trialtype='entryexit',neural_trials_25=neural_trials_sh_25)

    # 50ms sliding window
    allmice_behavior_avgtrials_sliding_newsh = [None]*len(allmice_dividedtrials)
    len_trial = allmice_neural_avgtrials_newsh[0].shape[-1]-5
    len_trial_25 = allmice_neural_avgtrials_newsh_25[0].shape[-1]-10
    # window = 5

    # for i in range(len(allmice_dividedtrials)):
    #     allmice_behavior_avgtrials_sliding_newsh[i]= np.zeros((allmice_neural_avgtrials_newsh[i].shape[0],len_trial))
    #     for j in range(len_trial):
    #         allmice_behavior_avgtrials_sliding_newsh[i][:,j] = np.sum(allmice_neural_avgtrials_newsh[i][:,j:j+window], axis=1)

    allmice_neural_avgtrials_newshuffle_zscored,ignoreunit = PCA_preprocessing.z_score_orderedhm(allmice_neural_avgtrials_newsh_25.copy(), window_bin=0.025)

    # ehm_newshuffle = [None]*5
    # ihm_newshuffle = [None]*5
    # rhm_newshuffle = [None]*5
    # trialtype = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]
    # for i in range(len(ehm_newshuffle)):
    #     ehm_newshuffle[i], ihm_newshuffle[i], rhm_newshuffle[i],_,_,_ = PCA_preprocessing.ehm_ihm_rhm_pertrial(allmice_neural_avgtrials_newshuffle_zscored[i], trialtype[i])

    # # now z-score these and then bigarr these
    # trialtitle = ["Entry-Exit", "Arm to Arm"]
    # trialsubtypes_entryexit = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]
    # trialsubtypes_8t = ["OA - OA", "CA - CA", "OA - CA", "CA - OA","Headdips"]

    # finalunits_entryexit_newshuffle = PCA_preprocessing.plotheatmap(ehm_newshuffle, ihm_newshuffle, rhm_newshuffle, trialtitle[0], trialsubtypes_entryexit)

    # PCA on this 
    # np.random.shuffle(np.array(allmice_neural_avgtrials_newshuffle_zscored[0][:,70:]))
    bigarr_newshuffle = (np.array(allmice_neural_avgtrials_newshuffle_zscored[0]))
    for i in range(1, len(allmice_neural_avgtrials_newshuffle_zscored)):
        #np.random.shuffle(np.array(allmice_neural_avgtrials_newshuffle_zscored[i][:,70]))
        bigarr_newshuffle = np.hstack((bigarr_newshuffle,allmice_neural_avgtrials_newshuffle_zscored[i]))
    excludetrials = np.concatenate((np.arange(len_trial,len_trial*2), np.arange(len_trial*3, len_trial*4)))
    bigarr_newshuffle =(np.delete(bigarr_newshuffle, excludetrials, axis=1))
    
    # np.random.shuffle(bigarr_newshuffle)

    #finalunits_entryexit_newshuffle = PCA_preprocessing.plotheatmap_orderOAentry(ehm_newshuffle, ihm_newshuffle, rhm_newshuffle, allmice_neural_avgtrials_newshuffle_zscored, trialtitle[0], trialsubtypes_entryexit)
    l_proj_5trials_newshuffle, var_5trials_newshuffle ,_ = PCA_preprocessing.PCA_analysis(l_data_PCA=(bigarr_newshuffle))
    
    #np.savez("shuffledPCAbootstrap500.npz", l_proj_5trials_newshuffle)

    return l_proj_5trials_newshuffle[:,:], allmice_neural_avgtrials_newshuffle_zscored, allmice_dividedtrials_sh_25, bigarr_newshuffle, ignoreunit, var_5trials_newshuffle

