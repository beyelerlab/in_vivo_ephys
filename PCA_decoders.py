from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn
import numpy as np
from scipy.ndimage import gaussian_filter1d
import PCA_preprocessing
import PCA_statstests
from scipy.stats import sem
import matplotlib.pyplot as plt

def decoder(l_traj_stats, num_trials =3, decodertype = "svm" ,type ="single"):

    inc = int(len(l_traj_stats[0])/num_trials)
    if type == "multi":
        num_datasets = len(l_traj_stats)
        num_sess = len(l_traj_stats[0])
        inc = int(len(l_traj_stats[0][0])/num_trials)
    else:
        num_datasets = 1
        num_sess = len(l_traj_stats)
    labels = np.zeros((len(l_traj_stats)*num_trials*num_datasets))

    trajectories = np.zeros((len(l_traj_stats)*num_trials*num_datasets, inc, 3))
    c = 0
    
    #print(inc)
    for k in range(num_datasets):
        for i in range(num_sess):
            for j in range(num_trials):
                if num_datasets>1:
                    trajectories[c,:,:] = l_traj_stats[k][i][inc*j:inc*(j+1),:3]
                else:
                    trajectories[c,:,:] = l_traj_stats[i][inc*j:inc*(j+1),:3]
                labels[c] = j
                c+=1
       
    train_X, train_Y, test_X, test_Y,adjusted_r_squared, auc, ind_accuracy, confusion_matrix = [],[],[],[],[], [],[],[]
    ind_accuracy_OA,ind_accuracy_CA,ind_accuracy_h= [],[],[]
    # train-test split : 3/4th of dataset to train, 1/4th to test
    for i in range((trajectories).shape[1]):
        
        train_x, test_x, train_y, test_y = train_test_split(trajectories[:,i,:].squeeze(), labels)
        #print(test_y)
        if decodertype =="linear":
            reg = sklearn.linear_model.Lasso(alpha=0.1) #linear_model.LinearRegression() # svm.LinearSVC()#svm.SVC() #
            reg.fit(train_x, train_y)
            yhat = reg.predict(test_x)
        #y_pred_proba = reg.predict_proba(test_x)[::,1]
        if decodertype == "svm":
            #print('svm')
            reg = svm.LinearSVC(max_iter=50000)#svm.SVC(probability=True) #linear_model.Lasso(alpha=0.1) #linear_model.LinearRegression() # #
            reg.fit(train_x, train_y)
            yhat = reg.predict(test_x)
            #y_pred_proba = reg.predict_proba(test_x)[::,1]
        SS_Residual = sum((test_y-yhat)**2)       
        SS_Total = sum((test_y-np.mean(test_y))**2)    
        #print(SS_Total) 
        #r_squared =np.mean(np.corrcoef(yhat, test_y)) #1-(float(SS_Residual))/SS_Total #
        #print(yhat)
        confusion_matrix.append(sklearn.metrics.confusion_matrix(test_y, yhat))
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["OA_entry","CA_entry","Headdips"])
        #cm_display.plot()
        #plt.show()  
        r_squared = np.trace(confusion_matrix[i])/np.sum(confusion_matrix[i])
        #adjusted_r_squared.append(1 - (1-r_squared)*(len(test_Y)-1)/(len(test_Y)-test_x[0].size-1))
        adjusted_r_squared.append(r_squared)
        if len(np.unique(train_y)) == 3:
            ind_accuracy_OA.append(confusion_matrix[i][0,0]/np.sum(confusion_matrix[i][0,:]))
            ind_accuracy_CA.append(confusion_matrix[i][1,1]/np.sum(confusion_matrix[i][1,:]))
            ind_accuracy_h.append(confusion_matrix[i][-1,-1]/np.sum(confusion_matrix[i][-1,:]))
        # ind_accuracy.append([ind_accuracy_OA, ind_accuracy_CA, ind_accuracy_h])
        #auc.append(sklearn.metrics.roc_auc_score(test_Y, y_pred_proba, multi_class='ovr'))
        #print(reg.coef_)
    return gaussian_filter1d(adjusted_r_squared, sigma=2),ind_accuracy_OA,ind_accuracy_CA,ind_accuracy_h



def dataset_generator_decoder(allmice_dividedtrials, unitspermouse_cumsum,eigvec, eigval, units):
    
    allmice_neural_sliding = [None]*len(allmice_dividedtrials)
    if allmice_dividedtrials[0][0].shape[1]>200:
        len_trial = allmice_dividedtrials[0][0].shape[1]-10
        window = 10
        window_bin = 0.025
    else:
        len_trial = allmice_dividedtrials[0][0].shape[1]-5
        window = 5
        window_bin = 0.05
    removeunitspertrial = [None]*len(allmice_dividedtrials)
    trials_final = [None]*len(allmice_dividedtrials)
    

    for i in range(len(allmice_dividedtrials)):
        allmice_neural_sliding[i]=[None]*len(allmice_dividedtrials[i])
        removeunitspertrial[i] = [None]*len(allmice_dividedtrials[i])
        trials_final[i] = [None]*len(allmice_dividedtrials[i])

        for k in range(len(allmice_dividedtrials[0])):
            trials_final[i][k]=[]
            removeunitspertrial[i][k]=[]
            allmice_neural_sliding[i][k] = np.zeros((allmice_dividedtrials[i][k].shape[0],len_trial,allmice_dividedtrials[i][k].shape[-1]))
            for j in range(len_trial):
                allmice_neural_sliding[i][k][:,j,:] = np.sum(allmice_dividedtrials[i][k][:,j:j+window,:], axis=1)
            
            for j in range(allmice_neural_sliding[i][k].shape[0]):
                zsc,rem = PCA_preprocessing.z_score(array=allmice_neural_sliding[i][k][j,:,:].T.copy(), ignoreunits=[], window_bin=window_bin)
                
                removeunitspertrial[i][k].append(rem)
                if len(rem) != 0:
                    zsc = np.delete(zsc, rem, axis=0)
                if np.sum(1*np.isnan(zsc)) == 0 and np.sum(1*np.isinf(zsc)) == 0:
                    #trials_final[i][k].append(zsc)
                    deleteeigenvalsvecs =np.arange(unitspermouse_cumsum[k],unitspermouse_cumsum[k+1])#np.arange(eigvec.shape[0]-(-unitspermouse_cumsum[i]+unitspermouse_cumsum[i+1]),eigvec.shape[0]) #
                    if len(rem) != 0:
                        deleteeigenvalsvecs= np.delete(deleteeigenvalsvecs, rem)
                    eigvec_copy = np.copy(eigvec)
                    eigenval_copy = np.copy(eigval)
                    # print(len(deleteeigenvalsvecs))
                    eigenvec_stats = eigvec_copy[deleteeigenvalsvecs,:]
                    eigval_stats =eigenval_copy[deleteeigenvalsvecs]
                    idx_argsorteigval = np.argsort(eigval_stats)[::-1]
                    eigenvec_stats = eigenvec_stats[:,idx_argsorteigval]
                    l_traj_bigsinglesess = PCA_statstests.projectdata(data_mat=zsc, PC=eigenvec_stats)[:,:units] #PCA_analysis(bigarr_excludeonemouse[i]) #PCA_analysis(bigarr_onlyone[i])[0] #
                    
                    if (l_traj_bigsinglesess.shape)[-1] == 0:
                        continue
                    trials_final[i][k].append(l_traj_bigsinglesess)
            trials_final[i][k]= np.array(trials_final[i][k])
    return trials_final



def dataset_permute_bootstrapper(trials_final, dataset_num =100, trialtype_select=np.array([0,2,4])):
    np.random.seed(42)
    datasets = [None]*dataset_num
    mintrialnum = 0
    # trialtype_select = np.array([0,2,4])
    for i in trialtype_select:
        
        for j in range(len(trials_final[i])):
            trials_final[i][j] = np.array(trials_final[i][j])
            if i==trialtype_select[0] and j==0:
                mintrialnum = trials_final[i][j].shape[0]
            if mintrialnum>trials_final[i][j].shape[0]:
                mintrialnum = trials_final[i][j].shape[0]
    # print(mintrialnum)
    # print()
    mintrialnum = mintrialnum-1
    for i in range(dataset_num):
        datasets[i]=[]
        for j in range(len(trials_final[0])):# per mouse
            
            idx_trialtype = []
            for k in trialtype_select: # per trialtype - choose trials, concatenate to x and append to dataset
                idx_trialtype.append(np.random.permutation(np.arange(0,trials_final[k][j].shape[0]))[:mintrialnum])
            
            for tr in range(mintrialnum):
                x =[]
                c=0
                for k in trialtype_select:
                    if k==trialtype_select[0]:
                        x = np.array(trials_final[k][j][idx_trialtype[c][tr],:,:])
                    else:
                        
                        x = np.concatenate((x, trials_final[k][j][idx_trialtype[c][tr],:,:]), axis=0)
                    c+=1
                   
                datasets[i].append(x)
    return datasets

def decoder_3c2(trials_final_real,trials_final_sh, trialtype_select, title, dataset_num, l_traj_stats):
    np.random.seed(50)
    datasets_real_1 = dataset_permute_bootstrapper(dataset_num=10,trials_final=trials_final_real, trialtype_select=trialtype_select)
    datasets_sh_1 = dataset_permute_bootstrapper(dataset_num=10,trials_final=trials_final_sh, trialtype_select=trialtype_select)
    decodertype = "svm"
    adjusted_r_squared, adjusted_r_squared_sh=[], []
    # acc_OA, acc_CA, acc_hd, acc_OA_sh, acc_CA_sh, acc_hd_sh = [], [], [], [], [], []
    for i in range(dataset_num):
        print(i)
        x,oa,ca, hd = decoder(l_traj_stats=datasets_real_1[i], decodertype=decodertype,num_trials=2)
        adjusted_r_squared.append(x)
        # acc_OA.append(oa)
        # acc_CA.append(ca)
        # acc_hd.append(hd)

        x,oa,ca, hd = decoder(l_traj_stats=datasets_sh_1[i], decodertype=decodertype, num_trials=2)
        adjusted_r_squared_sh.append(x)
        # acc_OA_sh.append(oa)
        # acc_CA_sh.append(ca)
        # acc_hd_sh.append(hd)

    fig, ax = plt.subplots()
    mean_real_adj = np.mean(adjusted_r_squared, axis=0)
    mean_sh_adj = np.mean(adjusted_r_squared_sh, axis=0)
    stdd_real_adj = sem(adjusted_r_squared, axis=0) # change to sem 
    stdd_sh_adj = sem(adjusted_r_squared_sh, axis=0)

    ax.plot((mean_real_adj)*100, c='blue')
    ax.plot((mean_sh_adj)*100, c="grey")
    ax.plot((mean_real_adj-stdd_real_adj)*100,c='lightblue')
    ax.plot((mean_real_adj+stdd_real_adj)*100,c='lightblue')
    ax.plot((mean_sh_adj-stdd_sh_adj)*100,c='lightgrey')
    ax.plot((mean_sh_adj+stdd_sh_adj)*100,c='lightgrey')

    ax.set_title('% Accuracy vs time - '+title)
    ax.set_ylabel('R_sq')
    ax.set_xlabel('Timepoints')
    t_show = np.arange(-3,4,1)
    ticknum = len(t_show)
    idx_ticks = np.linspace(0, int((l_traj_stats[0].shape[0])/3), ticknum)
    ax.set_xticklabels(t_show)
    ax.set_xticks(idx_ticks)
    plt.ylim([0,100])
    plt.savefig('PCA_finalpicturesNew/SVM/decoderEPM_accuracy_'+title+'.svg')
    plt.show()

    return adjusted_r_squared,adjusted_r_squared_sh
