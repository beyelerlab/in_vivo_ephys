import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem, wilcoxon

# Divide session into trials
# conditions - the mouse is in the prev and the post EPM region for at least 1 sec


def projectdata(PC, data_mat):
    projected_data = np.matmul((np.transpose(data_mat)), PC)
    return projected_data

def PCA_analysis(l_data_PCA, k = 3):
    # k = 3  # nb of dimensions
    l_projected_data = []
    var = []
    data_mat = l_data_PCA.copy()
    # for i in range(l_data_PCA.shape[0]):
    #     data_mat[i,:] = (l_data_PCA[i,:] - np.mean(l_data_PCA[i,:]))/(np.std(l_data_PCA[i,:])) #StandardScaler().fit_transform(l_data_PCA) # l_data_PCA
    #data_mat = l_data_PCA.copy()
    #print(data_mat.shape)
    cov_mat = np.cov(data_mat) # MinCovDet().fit(data_mat.T).covariance_#
   #print(cov_mat.shape)
    eigval, eigvec = np.linalg.eig(cov_mat)
    idx = eigval.argsort()[::-1] # ??
    eigvec_sorted = eigvec[:, idx]
    eigval_sorted = eigval[idx]
    PC = eigvec_sorted[:, :k]
    projected_data = projectdata(PC, data_mat)
    eigval_sorted_normalised = eigval_sorted/np.sum(eigval_sorted)
    l_projected_data.append(projected_data)
    var.append(eigval_sorted_normalised)
    print("Dimension of projected data:")
    print(projected_data.shape)
    return projected_data, eigval_sorted_normalised, PC

def PCA_plot(l_proj_5trials):
    c=0
    len_trial = int((l_proj_5trials.shape[0])/5) # time
    c1 = len_trial # 121
    colours = ['red', 'lightcoral', 'dimgrey', 'darkgrey', 'darkred'] #['red', 'orange','green', 'blue', "black"]
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

    for i in range(len(finalunits_entryexit)):
        sig = 3
        print(c)
        if i ==5:
            continue
        x = gaussian_filter1d(l_proj_5trials[c:c1,0], sigma=sig)
        y = gaussian_filter1d(l_proj_5trials[c:c1,1], sigma=sig)
        z = gaussian_filter1d(l_proj_5trials[c:c1,2], sigma=sig)
        mid = int(x.size/2)
        plt.plot(x,y,z, label = leg[i], c = colours[i])
        ax.scatter3D(x[0], y[0], z[0], c='k',s=10,marker="o")
        ax.scatter3D(x[mid], y[mid], z[mid], c='k',s=10,marker="v")
        ax.set_ylim([-5,5])
        ax.set_zlim([-5,5])
        ax.set_xlim([-5,5])
        #ax.set_facecolor('white')
        #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        c=c1
        c1 += len_trial

    plt.legend()
    plt.title('PCA')
    #plt.savefig('PCA_allmice_allunits_EPM_definedtrials_entryexit_19012024.png') 
    plt.show()

def centre_data(neural):
    # neuraldata of shape (time, units)
    # find avg for whole rray and stddev
    # center
    mean = np.nanmean(neural)
    stdd = np.nanstd(neural)
    neural_scaled = (neural-mean)/stdd
    return neural_scaled


# Divide session into trials
# conditions - the mouse is in the prev and the post EPM region for at least 1 sec

def divide_behavior(neural,neural_25, behavior):
    idx_prev = 0
    idx = 0
    trialarr_neur = []
    trialarr_neur_25 = []
    trialarr_beh = []
    beh_timepoints=[None]*3
    thr = int(3/0.05)+5
    thr_post_extra= 5
    thr_selection = int(1/0.05) # 20 = 1 sec
    behaviour_withinarm = [None]*2
    neural_withinarm= [None]*2
    
    
    for i in range(2):
        behaviour_withinarm[i] = []
        neural_withinarm[i] = []
        beh_timepoints[i] = []
    beh_timepoints[-1] = []
    #print()
    for i in range(neural.shape[0]-1): # for every tp

        idx_prev = behavior[i,0]
        idx = behavior[i+1,0]
        
        if idx_prev != idx and (i-thr>=0 and i+thr+1<neural.shape[0]):
            #print(behavior[i-thr_selection+1:i+1+thr_selection, 0] )
            # print(idx)
            # print(idx_prev)
            if idx != -1 and idx_prev!=-1:
                #print('in')
                beh_prev = behavior[i-thr_selection+1:i+1, 0] #
                beh_post = behavior[i+1:i+1+thr_selection, 0]
                
                beh_post[beh_post == -1] = idx
                beh_prev[beh_prev == -1] = idx_prev
                # print(beh_prev)
                # print(beh_post)
                # diff_prev = #np.diff(beh_prev)
                # diff_post = #np.diff(beh_post)
                if idx_prev != 4: # exit
                    y = np.where(beh_post !=idx)[0].size #diff_post
                    x = np.where(beh_prev !=idx_prev)[0].size#diff_prev
                    
                   ## diff_post = diff_prev
                   # diff_prev = x
                else: # entry
                    x = np.where(beh_post !=idx)[0].size #diff_post
                    y = np.where(beh_prev !=idx_prev)[0].size#diff_prev
                #print(diff_prev)
                #print(diff_post)
                # x[x !=0] = 1
                # num_x = np.sum(x)#len(np.where(x !=0)[0])
                # y[y !=0] = 1
                # num_y = np.sum(y)#len(np.where(x !=0)[0])
                # print(x)
                # print(y) 
                if x<1 and idx_prev != -1: #all(0 == ii for ii in x):# and all(0==ii for ii in y): 
                    #print('selected')
                    trialarr_beh.append(behavior[i-thr:i+thr+1,:])
                    if neural_25 is not None:
                        trialarr_neur_25.append(neural_25[(i-thr)*2:(i+thr+1+thr_post_extra)*2,:])
                    trialarr_neur.append(neural[i-thr:i+thr+1+thr_post_extra,:])
                    beh_timepoints[0].append(i)  
            else:
                if idx_prev == 0 or idx_prev == 1:
                    #print('headdip')
                    trialarr_beh.append(behavior[i-thr:i+thr+1,:])
                    trialarr_neur.append(neural[i-thr:i+thr+1+thr_post_extra,:])
                    if neural_25 is not None:
                        trialarr_neur_25.append(neural_25[(i-thr)*2:(i+thr+1+thr_post_extra)*2,:])
                    beh_timepoints[0].append(i)

    # now for the trials within arm
    start_t = 0 
    end_t = 0
    count = 0
    thr = int(2/0.05)
    thr_perside = int(thr/2)
    for i in range(neural.shape[0]-1):
        idx_prev = behavior[i,0]
        idx = behavior[i+1,0]
        if (idx_prev == -1 and (idx == 0 or idx == 1)):
            idx_prev = idx
        if (idx == -1 and (idx_prev == 1 or idx_prev == 0)):
            idx = idx_prev
            
        if idx_prev == idx and idx != 4 and idx != 5 and idx != -1:
            #print(idx)
            if count == 0:
                #print(idx_prev)
                start_t = i
            count +=1
        
        elif idx_prev != idx and count>=thr and idx == 4:
            #print(count)
            
            #print()
            #print(idx_prev)
           # print()
            end_t = i
            #print(end_t-start_t)
            
            snippet = behavior[start_t:end_t,:]
            snippetneur = neural[start_t:end_t,:]
            #print(snippet.shape)
            #print(snippetneur.shape)
            # now to check for the turning point 
            #print(snippet)
            beh_diff = np.diff(snippet)
            #print(beh_diff)
            direction = np.sign(beh_diff)
            #print(direction)
            diff_direction = np.diff(direction)
            #print(diff_direction)
            criticalpoints = np.where(diff_direction != 0)[0]
            #print(criticalpoints)
            #print(np.unique(diff_direction))
            #print((i == 0 for i in np.unique(diff_direction)))
            #if len(criticalpoints)==1:
            #    print('yay')
            #    if 
            #    behaviour_withinarm.append(behavior[criticalpoints-thr_perside:criticalpoints+thr_perside+1,:])
            #else:
                #plt.plot(snippet)
                #max_dist_idx = np.argmax(np.abs(snippet))
            if idx_prev == 2 or idx_prev == 3:
                behaviour_withinarm[1].append(snippet)
                neural_withinarm[1].append(snippetneur)
                beh_timepoints[2].append(i)
            else:
                behaviour_withinarm[0].append(snippet)
                neural_withinarm[0].append(snippetneur)
                beh_timepoints[1].append(i)
                #print(max_dist_idx)
                #print(criticalpoints)
                #if (i == max_dist_idx for i in criticalpoints):
                
                #    behaviour_withinarm.append(behavior[criticalpoints[0]-thr_perside:criticalpoints[0]+thr_perside+1,:])
            start_t = 0
            end_t = 0
            count = 0
        else: 
            start_t = 0
            end_t = 0
            count = 0
        
    plt.show()
    return trialarr_beh, trialarr_neur, trialarr_neur_25, behaviour_withinarm, neural_withinarm, beh_timepoints

# find idx for entry to OA and exit from OA and entry to CA and exit from CA
# avg across trials 

def check_meanFR(neural):
    # neural is time x units array 
    # check if each unit of this trial is above threshold FR
    thr = 0 # Hz - 0 Hz initially (5.3.24) # changed from 0 to 1 for thesis version - change it back 
    
    mean_FR = np.nanmean(neural, axis=0)/0.05
    x = []
    for i in range(len(mean_FR)):
        if mean_FR[i]<=thr:
            x.append(i)
    # if all(i>=thr for i in mean_FR):
    #     return True, np.array([])
    # else: 
    #     #print('hm')
    #     x = np.argwhere(i<thr for i in mean_FR).squeeze()
    #     print(x)
    #     return False, x
    if len(x) == 0:
        return True, np.array([])
    else:
        # print(x)
        if len(x)>5: # if False:#len(x)>=5: (5.3.24)
            return False, np.array([])
        else:
            return True, np.array([])
        #return False, np.array([])



def heatmaps_matrixgen_avgtrials_old(neural_trials, neural_trials_25, behaviour_trials, mouseidx, beh_timepoints):
    check_flag = 0
    idxx = [None]*5
    idxx_reduced = [None]*5
    for i in range(len(idxx)):
        idxx[i]=[]
        idxx_reduced[i] = []

    # find all the idx for OA, CA entry and exit
    mid = int(behaviour_trials[mouseidx][0].shape[0]/2)
    trialnum = 0
    total_headdip = int(3/0.05)
    thr_headdip = int(0.75/0.05) # 1sec # used to be 0.75sec for good result - 22nd feb '24
    thr_post_headdip = int(0.5/0.05) # 0.5 sec
    thr_headdipbout = int(1/0.05) # make sure that the headdips are at least 1 sec apart  # used to be 1sec for good result - 22nd feb '24
    remunits = []#np.array([])
    num_remunits = np.array([])
    trialtag = []
    reduced_c= 0
    bigtrialnum = 0
    if len(neural_trials_25) == 0:
        dat_25 = False
    else:
        dat_25 = True
    for i in range(len(neural_trials[mouseidx])):
        idx = behaviour_trials[mouseidx][i][mid, 0]
        idx_next = behaviour_trials[mouseidx][i][mid+1, 0]
        cond, rem = check_meanFR(neural_trials[mouseidx][i])
        rem = np.array(rem)
        
        #print('before cond')
        ##print(rem)
        #print(remunits)
        #if rem.size!= 0:
        #print(num_remunits)
       
        bigtrialnum +=1
        if cond==True: #rem.size == 0: # removing trials with units of very little firing 
            trialnum += 1
            if len(rem) != 0:
                if len(remunits) == 0:
                    remunits = np.array(rem)
                else:
                    remunits = np.append(remunits, np.array(rem))

            if idx == 4 and (idx_next == 0 or idx_next == 1):
                idxx[0].append(i)
                idxx_reduced[0].append(reduced_c)
                reduced_c+=1
                trialtag.append([0])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                

            elif (idx == 0 or idx == 1) and idx_next == 4:
                idxx[1].append(i)
                idxx_reduced[1].append(reduced_c)
                reduced_c+=1
                trialtag.append([1])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                

            elif idx == 4 and (idx_next == 2 or idx_next == 3):
                idxx[2].append(i)
                trialtag.append([2])
                idxx_reduced[2].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                

            elif (idx == 2 or idx == 3) and idx_next == 4:
                idxx[3].append(i)
                trialtag.append([3])
                idxx_reduced[3].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                

            elif (idx == 0 or idx == 1) and idx_next == -1: # headdip
                idx_prev_arr =behaviour_trials[mouseidx][i][mid-thr_headdip:mid,0]
                c=0
                for j in range(len(idx_prev_arr)):
                    if idx_prev_arr[j] == idx:
                        c+=1
                idx_post_arr = behaviour_trials[mouseidx][i][mid+1:mid+1+thr_post_headdip,0]
                c1=0
                for j in range(len(idx_post_arr)):
                    if idx_post_arr[j] == idx_next:
                        #print(idx_post_arr[j])
                        c1+=1
                if c == len(idx_prev_arr) and c1 == len(idx_post_arr):# and i+2+total_headdip<behaviour_trials[mouseidx][i].shape[0]:
                    
                    if len(idxx[4]) == 0 or i-idxx[4][-1] >= thr_headdipbout:
                        idxx[4].append(i)
                        trialtag.append([4])
                        idxx_reduced[4].append(reduced_c)
                        reduced_c+=1
                        if all(num_remunits.shape) != 0:
                            #print('cond')
                            #print(rem)
                            #print(remunits)
                            #remunits= np.concatenate((remunits, rem))
                            #  print(i)
                            num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                        else:
                            #remunits = rem
                            # print(i)
                            # print()
                            num_remunits = np.array(rem.shape)
                        
            else:
                trialnum -= 1
                bigtrialnum -=1
                #print('no trial')
                #print(idx)
                #print(idx_next)
        # else:
        #     print('hm')


    perc =  trialnum*100/bigtrialnum #len(indices_toremove)*100/trialnum
    print("trials retained: " +  str(perc))
    #print(trialtags_rem)
    dividedtrials = [None]*5
    dividedtrials_25 = [None]*5
    dividedtrials_beh = [None]*5
    trialtimes = [None]*5
    for i in range(len(dividedtrials)):
        if len(idxx[i]) == 0:
            arr = np.array([])
            dividedtrials[i] = arr
            dividedtrials_beh[i] = arr
            trialtimes[i] = []
            check_flag = 1
            continue
        else:
            #neural
            tt = []
            arr = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], neural_trials[mouseidx][0].shape[1])) # (trialnum, time, units)
            if dat_25:
                arr_25 = np.zeros((len(idxx[i]), neural_trials_25[mouseidx][0].shape[0], neural_trials_25[mouseidx][0].shape[1])) # (trialnum, time, units)
            for j in range(len(idxx[i])):
                arr[j,:,:] = neural_trials[mouseidx][idxx[i][j]] #np.mean(np.array(neural_trials[mouseidx][idxx[i][j]]), axis=-1)
                if dat_25:
                    print(neural_trials_25[mouseidx][idxx[i][j]].shape)
                    arr_25[j,:,:] = neural_trials_25[mouseidx][idxx[i][j]] 
                tt.append(beh_timepoints[mouseidx][0][idxx[i][j]])
            tt = np.array(tt)
            dividedtrials[i] = arr
            if dat_25:
                dividedtrials_25[i] = arr_25

            if len(remunits) != 0:
                remunits = np.unique(np.array(remunits).squeeze())
                #print(i)
                print(remunits)
                dividedtrials[i] = np.delete(dividedtrials[i], remunits, axis=-1)
            
            #print(arr.shape)
            #print(unitstoremove)
            # behaviour
            arr1 = np.zeros((len(idxx[i]), behaviour_trials[mouseidx][0].shape[0], 2)) # (trialnum, time, 2)
            for j in range(len(idxx[i])):
                arr1[j,:,:] = behaviour_trials[mouseidx][idxx[i][j]]
            
            dividedtrials_beh[i] = arr1


            trialtimes[i] = tt


    fig = plt.figure(constrained_layout=True, figsize=(25,10))
    axx = fig.subfigures(1,3)#plt.subplots(1,3, figsize = (15,5))
    axx= axx.flatten()
    titles = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]

    ax = axx[0].subplots(ncols=2, nrows=3)
    
    ax=ax.flatten().flatten() # i have trust issues

    # average across trials
    dividedtrials_avgtrials = [None]*5
    dividedtrials_avgtrials_25 = [None]*5

    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            dividedtrials_avgtrials[i] = np.mean(np.array(dividedtrials[i]), axis=0).T # (time, units)
            if dat_25:
                dividedtrials_avgtrials_25[i] = np.mean(np.array(dividedtrials_25[i]), axis=0).T # (time, units)
        else:
            dividedtrials_avgtrials[i] = []
            dividedtrials_avgtrials_25[i] = []

    #fig, ax = plt.subplots(1,3, figsize = (15,5))
    #ax= ax.flatten()
    #print(idxx[0])
    #print(len(idxx[0]))
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            x = ax[i].imshow(dividedtrials_avgtrials[i], cmap='cividis') # cmap='viridis'
        ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    number_trials = np.zeros(len(idxx))
    
    for kk in range(len(idxx)):
        number_trials[kk] = len(idxx[kk])

    
    ax[-1].bar(titles, number_trials)
    #ax[-1].set_xticklabels(np.arange(0,len(idxx)),titles)
    ax[-1].set_title('Number of trials')
    axx[0].suptitle('Neural activity for '+str(mouseidx))
    #plt.show()

    
    ax = axx[1].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].plot(dividedtrials_beh[i][:,:,1].T)
        ax[i].axvline(x=mid, color='r')
        ax[i].set_ylim([-250, 250])
        ax[i].axhline(y=-25,color='k')
        ax[i].axhline(y=25,color='k')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    # print(idxx[-1])
    #ax[-1].eventplot(beh_timepoints[mouseidx][np.array(idxx[-1]).astype(int)])
    ax[-1].eventplot(trialtimes[-1])
    ax[-1].set_title('Raster of headdips')
    axx[1].suptitle('Behavior activity for mouse '+str(mouseidx))
    #plt.show()

    ax = axx[2].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].eventplot(trialtimes[i])
        #ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[2].suptitle('Behavior activity for mouse '+str(mouseidx))
    plt.show()

    plt.eventplot(trialtimes[-1])
    plt.show()
    

    return dividedtrials_avgtrials,dividedtrials_avgtrials_25, dividedtrials_beh, check_flag, dividedtrials, dividedtrials_25, dividedtrials_beh, trialtimes





'''
def heatmaps_matrixgen_avgtrials_old(neural_trials, behaviour_trials, mouseidx, beh_timepoints):
    check_flag = 0
    idxx = [None]*5
    idxx_reduced = [None]*5
    for i in range(len(idxx)):
        idxx[i]=[]
        idxx_reduced[i] = []

    # find all the idx for OA, CA entry and exit
    mid = int(neural_trials[mouseidx][0].shape[0]/2)
    trialnum = 0
    total_headdip = int(3/0.05)
    thr_headdip = int(0.75/0.05) # 1sec # used to be 0.75sec for good result - 22nd feb '24
    thr_post_headdip = int(0.5/0.05) # 0.5 sec
    thr_headdipbout = int(1/0.05) # make sure that the headdips are at least 1 sec apart  # used to be 1sec for good result - 22nd feb '24
    remunits = []#np.array([])
    num_remunits = np.array([])
    trialtag = []
    reduced_c= 0
    bigtrialnum = 0
    for i in range(len(neural_trials[mouseidx])):
        idx = behaviour_trials[mouseidx][i][mid, 0]
        idx_next = behaviour_trials[mouseidx][i][mid+1, 0]
        cond, rem = check_meanFR(neural_trials[mouseidx][i])
        rem = np.array(rem)
        
        #print('before cond')
        ##print(rem)
        #print(remunits)
        #if rem.size!= 0:
        #print(num_remunits)
       
        bigtrialnum +=1
        if rem.size == 0: # removing trials with units of very little firing 
            trialnum += 1
            
            if idx == 4 and (idx_next == 0 or idx_next == 1):
                idxx[0].append(i)
                idxx_reduced[0].append(reduced_c)
                reduced_c+=1
                trialtag.append([0])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 0 or idx == 1) and idx_next == 4:
                idxx[1].append(i)
                idxx_reduced[1].append(reduced_c)
                reduced_c+=1
                trialtag.append([1])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif idx == 4 and (idx_next == 2 or idx_next == 3):
                idxx[2].append(i)
                trialtag.append([2])
                idxx_reduced[2].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 2 or idx == 3) and idx_next == 4:
                idxx[3].append(i)
                trialtag.append([3])
                idxx_reduced[3].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 0 or idx == 1) and idx_next == -1: # headdip
                idx_prev_arr =behaviour_trials[mouseidx][i][mid-thr_headdip:mid,0]
                c=0
                for j in range(len(idx_prev_arr)):
                    if idx_prev_arr[j] == idx:
                        c+=1
                idx_post_arr = behaviour_trials[mouseidx][i][mid+1:mid+1+thr_post_headdip,0]
                c1=0
                for j in range(len(idx_post_arr)):
                    if idx_post_arr[j] == idx_next:
                        #print(idx_post_arr[j])
                        c1+=1
                if c == len(idx_prev_arr) and c1 == len(idx_post_arr):# and i+2+total_headdip<behaviour_trials[mouseidx][i].shape[0]:
                    
                    if len(idxx[4]) == 0 or i-idxx[4][-1] >= thr_headdipbout:
                        idxx[4].append(i)
                        trialtag.append([4])
                        idxx_reduced[4].append(reduced_c)
                        reduced_c+=1
                        if all(num_remunits.shape) != 0:
                            #print('cond')
                            #print(rem)
                            #print(remunits)
                            #remunits= np.concatenate((remunits, rem))
                            #  print(i)
                            num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                        else:
                            #remunits = rem
                            # print(i)
                            # print()
                            num_remunits = np.array(rem.shape)
                        remunits.append(rem)
            else:
                trialnum -= 1
                bigtrialnum -=1
                #print('no trial')
                #print(idx)
                #print(idx_next)
        # else:
        #     print('hm')


    perc =  trialnum*100/bigtrialnum #len(indices_toremove)*100/trialnum
    print("trials retained: " +  str(perc))
    #print(trialtags_rem)
    dividedtrials = [None]*5
    dividedtrials_beh = [None]*5
    trialtimes = [None]*5
    for i in range(len(dividedtrials)):
        if len(idxx[i]) == 0:
            arr = np.array([])
            dividedtrials[i] = arr
            dividedtrials_beh[i] = arr
            trialtimes[i] = []
            check_flag = 1
            continue
        else:
            #neural
            tt = []
            arr = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], neural_trials[mouseidx][0].shape[1])) # (trialnum, time, units)
            for j in range(len(idxx[i])):
                arr[j,:,:] = neural_trials[mouseidx][idxx[i][j]] #np.mean(np.array(neural_trials[mouseidx][idxx[i][j]]), axis=-1)
                tt.append(beh_timepoints[mouseidx][0][idxx[i][j]])
            tt = np.array(tt)
            dividedtrials[i] = arr
            #print(arr.shape)
            #print(unitstoremove)
            # behaviour
            arr1 = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], 2)) # (trialnum, time, 2)
            for j in range(len(idxx[i])):
                arr1[j,:,:] = behaviour_trials[mouseidx][idxx[i][j]]
            
            dividedtrials_beh[i] = arr1

            trialtimes[i] = tt


    fig = plt.figure(constrained_layout=True, figsize=(25,10))
    axx = fig.subfigures(1,3)#plt.subplots(1,3, figsize = (15,5))
    axx= axx.flatten()
    titles = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]

    ax = axx[0].subplots(ncols=2, nrows=3)
    
    ax=ax.flatten().flatten() # i have trust issues

    # average across trials
    dividedtrials_avgtrials = [None]*5

    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            dividedtrials_avgtrials[i] = np.mean(np.array(dividedtrials[i]), axis=0).T # (time, units)
        else:
            dividedtrials_avgtrials[i] = []

    #fig, ax = plt.subplots(1,3, figsize = (15,5))
    #ax= ax.flatten()
    #print(idxx[0])
    #print(len(idxx[0]))
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            x = ax[i].imshow(dividedtrials_avgtrials[i], cmap='viridis')
        ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    number_trials = np.zeros(len(idxx))
    
    for kk in range(len(idxx)):
        number_trials[kk] = len(idxx[kk])

    
    ax[-1].bar(titles, number_trials)
    #ax[-1].set_xticklabels(np.arange(0,len(idxx)),titles)
    ax[-1].set_title('Number of trials')
    axx[0].suptitle('Neural activity for '+str(mouseidx))
    #plt.show()

    
    ax = axx[1].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].plot(dividedtrials_beh[i][:,:,1].T)
        ax[i].axvline(x=mid, color='r')
        ax[i].set_ylim([-250, 250])
        ax[i].axhline(y=-25,color='k')
        ax[i].axhline(y=25,color='k')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    # print(idxx[-1])
    #ax[-1].eventplot(beh_timepoints[mouseidx][np.array(idxx[-1]).astype(int)])
    ax[-1].eventplot(trialtimes[-1])
    ax[-1].set_title('Raster of headdips')
    axx[1].suptitle('Behavior activity for mouse '+str(mouseidx))
    #plt.show()

    ax = axx[2].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].eventplot(trialtimes[i])
        #ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[2].suptitle('Behavior activity for mouse '+str(mouseidx))
    plt.show()

    plt.eventplot(trialtimes[-1])
    plt.show()
    

    return dividedtrials_avgtrials, dividedtrials_beh, check_flag, dividedtrials, dividedtrials_beh, trialtimes
'''




def heatmaps_matrixgen_avgtrials(neural_trials, behaviour_trials, mouseidx, beh_timepoints):
    check_flag = 0
    idxx = [None]*5
    idxx_reduced = [None]*5
    for i in range(len(idxx)):
        idxx[i]=[]
        idxx_reduced[i] = []

    # find all the idx for OA, CA entry and exit
    mid = int(behaviour_trials[mouseidx][0].shape[0]/2)
    trialnum = 0
    total_headdip = int(3/0.05)
    thr_headdip = int(0.75/0.05) # 1sec # used to be 0.75sec for good result - 22nd feb '24
    thr_post_headdip = int(0.5/0.05) # 0.5 sec
    thr_headdipbout = int(1/0.05) # make sure that the headdips are at least 1 sec apart  # used to be 1sec for good result - 22nd feb '24
    remunits = []#np.array([])
    num_remunits = np.array([])
    trialtag = []
    reduced_c= 0
    for i in range(len(neural_trials[mouseidx])):
        idx = behaviour_trials[mouseidx][i][mid, 0]
        idx_next = behaviour_trials[mouseidx][i][mid+1, 0]
        cond, rem = check_meanFR(neural_trials[mouseidx][i])
        rem = np.array(rem)
        
        #print('before cond')
        ##print(rem)
        #print(remunits)
        #if rem.size!= 0:
        #print(num_remunits)
       
        
        if cond: # removing trials with units of very little firing 
            trialnum += 1
            
            if idx == 4 and (idx_next == 0 or idx_next == 1):
                idxx[0].append(i)
                idxx_reduced[0].append(reduced_c)
                reduced_c+=1
                trialtag.append([0])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 0 or idx == 1) and idx_next == 4:
                idxx[1].append(i)
                idxx_reduced[1].append(reduced_c)
                reduced_c+=1
                trialtag.append([1])
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif idx == 4 and (idx_next == 2 or idx_next == 3):
                idxx[2].append(i)
                trialtag.append([2])
                idxx_reduced[2].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 2 or idx == 3) and idx_next == 4:
                idxx[3].append(i)
                trialtag.append([3])
                idxx_reduced[3].append(reduced_c)
                reduced_c+=1
                if all(num_remunits.shape) != 0:
                    #print('cond')
                    #print(rem)
                    #print(remunits)
                    #remunits= np.concatenate((remunits, rem))
                    #  print(i)
                    num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                else:
                    #remunits = rem
                    # print(i)
                    # print()
                    num_remunits = np.array(rem.shape)
                remunits.append(rem)

            elif (idx == 0 or idx == 1) and idx_next == -1: # headdip
                idx_prev_arr =behaviour_trials[mouseidx][i][mid-thr_headdip:mid,0]
                c=0
                for j in range(len(idx_prev_arr)):
                    if idx_prev_arr[j] == idx:
                        c+=1
                idx_post_arr = behaviour_trials[mouseidx][i][mid+1:mid+1+thr_post_headdip,0]
                c1=0
                for j in range(len(idx_post_arr)):
                    if idx_post_arr[j] == idx_next:
                        #print(idx_post_arr[j])
                        c1+=1
                if c == len(idx_prev_arr) and c1 == len(idx_post_arr):# and i+2+total_headdip<behaviour_trials[mouseidx][i].shape[0]:
                    
                    if len(idxx[4]) == 0 or i-idxx[4][-1] >= thr_headdipbout:
                        idxx[4].append(i)
                        trialtag.append([4])
                        idxx_reduced[4].append(reduced_c)
                        reduced_c+=1
                        if all(num_remunits.shape) != 0:
                            #print('cond')
                            #print(rem)
                            #print(remunits)
                            #remunits= np.concatenate((remunits, rem))
                            #  print(i)
                            num_remunits = np.concatenate((num_remunits, np.array(rem.shape)))
                        else:
                            #remunits = rem
                            # print(i)
                            # print()
                            num_remunits = np.array(rem.shape)
                        remunits.append(rem)
            else:
                trialnum -= 1
                #print('no trial')
                #print(idx)
                #print(idx_next)
        else:
            print('hm')

    # trial removal 
    #histt, idx = np.hist(num_remunits)
    #print(len(remunits))
    trialtag = np.array(trialtag).squeeze()
    #print((trialtag))
    #print(len(num_remunits))
    #print(trialnum)
    
    hist, xedges = np.histogram(num_remunits, density=True, bins=100)
    plt.hist(num_remunits, density=True, bins=100)
    density_check = np.cumsum(xedges)#np.cumsum(hist * np.diff(xedges))
    print(density_check)
    if density_check[0]>0.9:
        idx_95perc = np.where(density_check<=5)[0][-1] # 0
    else:
        idx_95perc = np.where(density_check<=5)[0][-1] #np.where(density_check<=0.9)[0][-1]
    
    print('95 perc')
    print(idx_95perc)
    #print(np.where(density_check<=0.95))
    # Initialize an empty list to store indices of data points within each bin
    indices_in_bins = []
    indices_toremove = np.array([])
    trialtags_rem=np.array([])
    unitstoremove_notunique = np.array([])
    

    # Loop through the x and y bin edges
    for i in range(len(xedges)-1):
        # Create a boolean mask to identify data points within the current bin
        in_bin = ((xedges[i] <= num_remunits) & (num_remunits < xedges[i+1]))
        
        # Find the indices of data points within the current bin
        indices = np.where(in_bin)[0] 
        #print(in_bin)
        #print(indices)
        #trilasofindices = trialtag
        # Append the indices to the 'indices_in_bins' list
        indices_in_bins.append(indices)
        if i>idx_95perc:
            if (indices_toremove).size == 0:
                indices_toremove = np.array(indices)
                trialtags_rem = np.array(trialtag[indices])
            else:
                indices_toremove = np.concatenate((indices_toremove, indices))
                trialtags_rem = np.concatenate((trialtags_rem,trialtag[indices]))
        else:
            if unitstoremove_notunique.size == 0:
                for iii in range(len(indices)):
                    unitstoremove_notunique = np.array(remunits[indices[iii]])
                    if iii>0:
                        unitstoremove_notunique = np.concatenate((unitstoremove_notunique, remunits[indices[iii]]))
            else:
                for iii in range(len(indices)):
                    unitstoremove_notunique = np.concatenate((unitstoremove_notunique, remunits[indices[iii]]))

    trials_toremove = np.unique(indices_toremove)
    print(trials_toremove)
    print(indices_toremove)
    perc = len(indices_toremove)*100/trialnum
    print("trials removed: " +  str(perc))
    # Divide raw neural array into 4 (trial, time, unit) arrays
    unitstoremove = np.unique(unitstoremove_notunique)
    perc = len(unitstoremove)*100/(neural_trials[mouseidx][0].shape[-1])
    print("units removed: " +  str(perc))
    #print(trialtags_rem)
    dividedtrials = [None]*5
    dividedtrials_beh = [None]*5
    trialtimes = [None]*5
    for i in range(len(dividedtrials)):
        if len(idxx[i]) == 0:
            arr = np.array([])
            dividedtrials[i] = arr
            dividedtrials_beh[i] = arr
            trialtimes[i] = []
            check_flag = 1
            continue
        else:
            #neural
            tt = []
            arr = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], neural_trials[mouseidx][0].shape[1])) # (trialnum, time, units)
            for j in range(len(idxx[i])):
                arr[j,:,:] = neural_trials[mouseidx][idxx[i][j]] #np.mean(np.array(neural_trials[mouseidx][idxx[i][j]]), axis=-1)
                tt.append(beh_timepoints[mouseidx][0][idxx[i][j]])
            tt = np.array(tt)
            dividedtrials[i] = arr
            #print(arr.shape)
            #print(unitstoremove)
            # behaviour
            arr1 = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], 2)) # (trialnum, time, 2)
            for j in range(len(idxx[i])):
                arr1[j,:,:] = behaviour_trials[mouseidx][idxx[i][j]]
            
            dividedtrials_beh[i] = arr1

            if len(unitstoremove) != 0:
                #print(i)
                print(unitstoremove)
                dividedtrials[i] = np.delete(dividedtrials[i], unitstoremove, axis=-1)

            if len(trialtags_rem) != 0:
                #print(trialtag)
                #print(indices_toremove)
                #print(trialtags_rem)
                #print(idxx_reduced[i])
                if len(trialtags_rem[trialtags_rem == i]) != 0:
                    #rem_trial =  np.arange(0,trials_toremove[trialtags_rem == i])
                    rem_trial = []
                    for jj in range(len(idxx[i])):
                        x= (np.argwhere(idxx_reduced[i][jj] == indices_toremove[trialtags_rem==i]))
                        if x.size != 0:
                            rem_trial.append(x)
                    rem_trial = np.array(rem_trial).squeeze()
                    #print(rem_trial)
                    #print(trials_toremove[trialtags_rem==i])
                    #print(len(idxx[i]))
                    #print(len(trialtag[trialtag==i]))
                    #print(trialtag[trialtag==i])
                    #print(rem_trial)

                    dividedtrials[i] = np.delete(dividedtrials[i], rem_trial, axis=0)
                    dividedtrials_beh[i] = np.delete(arr1, rem_trial, axis=0)
                    tt = np.delete(tt, rem_trial)

            trialtimes[i] = tt


    fig = plt.figure(constrained_layout=True, figsize=(25,10))
    axx = fig.subfigures(1,3)#plt.subplots(1,3, figsize = (15,5))
    axx= axx.flatten()
    titles = ["OA entry", "OA exit", "CA entry", "CA exit", "Headdips"]

    ax = axx[0].subplots(ncols=2, nrows=3)
    
    ax=ax.flatten().flatten() # i have trust issues

    # average across trials
    dividedtrials_avgtrials = [None]*5

    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            dividedtrials_avgtrials[i] = np.mean(np.array(dividedtrials[i]), axis=0).T # (time, units)
        else:
            dividedtrials_avgtrials[i] = []

    #fig, ax = plt.subplots(1,3, figsize = (15,5))
    #ax= ax.flatten()
    #print(idxx[0])
    #print(len(idxx[0]))
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            x = ax[i].imshow(dividedtrials_avgtrials[i], cmap='cividis') # cmap='viridis'
        ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    number_trials = np.zeros(len(idxx))
    
    for kk in range(len(idxx)):
        number_trials[kk] = len(idxx[kk])

    
    ax[-1].bar(titles, number_trials)
    #ax[-1].set_xticklabels(np.arange(0,len(idxx)),titles)
    ax[-1].set_title('Number of trials')
    axx[0].suptitle('Neural activity for '+str(mouseidx))
    #plt.show()

    
    ax = axx[1].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].plot(dividedtrials_beh[i][:,:,1].T)
        ax[i].axvline(x=mid, color='r')
        ax[i].set_ylim([-250, 250])
        ax[i].axhline(y=-25,color='k')
        ax[i].axhline(y=25,color='k')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    # print(idxx[-1])
    #ax[-1].eventplot(beh_timepoints[mouseidx][np.array(idxx[-1]).astype(int)])
    ax[-1].eventplot(trialtimes[-1])
    ax[-1].set_title('Raster of headdips')
    axx[1].suptitle('Behavior activity for mouse '+str(mouseidx))
    #plt.show()

    ax = axx[2].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].eventplot(trialtimes[i])
        #ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[2].suptitle('Behavior activity for mouse '+str(mouseidx))
    plt.show()

    plt.eventplot(trialtimes[-1])
    plt.show()
    

    return dividedtrials_avgtrials, dividedtrials_beh, check_flag, dividedtrials, dividedtrials_beh, trialtimes


def heatmaps_matrixgen_avgtrials_8trialtypes(neural_trials, behaviour_trials, mouseidx):
    check_flag = 0
    idxx = [None]*5
    for i in range(len(idxx)):
        idxx[i]=[]

    # find all the idx for OA, CA entry and exit
    mid = int(neural_trials[mouseidx][0].shape[0]/2)
    trialnum = 0
    total_headdip = int(3/0.05)
    thr_headdip = int(1/0.05) # 1sec
    thr_post_headdip = int(0.5/0.05) # 0.5 sec
    thr_headdipbout = int(1/0.05) # make sure that the headdips are at least 1 sec apart 
    for i in range(len(neural_trials[mouseidx])):
        idx = behaviour_trials[mouseidx][i][mid, 0]
        idx_next = behaviour_trials[mouseidx][i][mid+1, 0]
        first = behaviour_trials[mouseidx][i][0, 0]
        last = behaviour_trials[mouseidx][i][-1, 0]
        if check_meanFR(neural_trials[mouseidx][i]): # removing trials with units of very little firing 
            trialnum += 1
            if idx == 4 and (idx_next == 0 or idx_next == 1) and (first == 0 or first == 1) and (last == 0 or last==1): # OA - OA - OA entry
                idxx[0].append(i)
            if (idx == 0 or idx == 1) and idx_next == 4 and (first == 0 or first == 1) and (last == 0 or last==1): # OA - OA - OA exit
                idxx[0].append(i)

            if idx == 4 and (idx_next == 2 or idx_next == 3) and (first == 2 or first == 3) and (last == 2 or last==3): # CA - CA - CA entry
                idxx[1].append(i)
            if (idx == 2 or idx == 3) and idx_next == 4 and (first == 2 or first == 3) and (last == 2 or last==3): # CA - CA - CA exit
                idxx[1].append(i)

            if idx == 4 and (idx_next == 2 or idx_next == 3) and (first == 1 or first == 0) and (last == 2 or last==3): # OA - CA - CA entry
                idxx[2].append(i)
            if (idx == 0 or idx == 1) and idx_next == 4 and (first == 0 or first == 1) and (last == 2 or last==3): # OA - OA - CA exit
                idxx[2].append(i)

            if idx == 4 and (idx_next == 0 or idx_next == 1) and (first == 2 or first == 3) and (last == 0 or last==1): # CA - OA - OA entry
                idxx[3].append(i)
            if (idx == 2 or idx == 3) and idx_next == 4 and (first == 2 or first == 3) and (last == 0 or last==1): # CA - CA - OA exit
                idxx[3].append(i)

            if (idx == 0 or idx == 1) and idx_next == -1: # headdip
                idx_prev_arr =behaviour_trials[mouseidx][i][i-thr_headdip:i,0]
                c=0
                for j in range(len(idx_prev_arr)):
                    if idx_prev_arr[j] == idx:
                        c+=1
                idx_post_arr = behaviour_trials[mouseidx][i][i+1:i+2+thr_post_headdip,0]
                c1=0
                for j in range(len(idx_post_arr)):
                    if idx_post_arr[j] == idx_next:
                        #print(idx_post_arr[j])
                        c1+=1
                if c == len(idx_prev_arr) and c1 == len(idx_post_arr) and i-total_headdip>0 and i-idxx[4][-1]>=thr_headdipbout:# and i+2+total_headdip<behaviour_trials[mouseidx][i].shape[0]:
                    idxx[4].append(i)

    perc = trialnum*100/len(neural_trials[mouseidx])
    print("trials retained: " +  str(perc))
    # Divide raw neural array into 4 (trial, time, unit) arrays

    dividedtrials = [None]*5
    dividedtrials_beh = [None]*5

    for i in range(len(dividedtrials)):
        if len(idxx[i]) == 0:
            arr = np.array([])
            dividedtrials[i] = arr
            dividedtrials_beh[i] = arr
            check_flag = 1
            continue
        else:
            #neural
            arr = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0],neural_trials[mouseidx][0].shape[1])) # (trialnum, time, units)
            for j in range(len(idxx[i])):
                arr[j,:,:] = neural_trials[mouseidx][idxx[i][j]] #np.mean(np.array(neural_trials[mouseidx][idxx[i][j]]), axis=-1)

            dividedtrials[i] = arr
            
            # behaviour
            arr = np.zeros((len(idxx[i]), neural_trials[mouseidx][0].shape[0], 2)) # (trialnum, time, 2)
            for j in range(len(idxx[i])):
                arr[j,:,:] = behaviour_trials[mouseidx][idxx[i][j]]
            dividedtrials_beh[i] = arr

    fig = plt.figure(constrained_layout=True, figsize=(25,10))
    axx = fig.subfigures(1,3)#plt.subplots(1,3, figsize = (15,5))
    axx= axx.flatten()
    titles = ["OA - OA", "CA - CA", "OA - CA", "CA - OA","Headdips"]

    ax = axx[0].subplots(ncols=2, nrows=3)
    
    ax=ax.flatten().flatten() # i have trust issues

    # average across trials
    dividedtrials_avgtrials = [None]*5

    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            dividedtrials_avgtrials[i] = np.mean(np.array(dividedtrials[i]), axis=0).T # (time, units)
        else:
            dividedtrials_avgtrials[i] = []

    #fig, ax = plt.subplots(1,3, figsize = (15,5))
    #ax= ax.flatten()

    for i in range(len(dividedtrials_avgtrials)):
        if len(dividedtrials[i]) != 0:
            x = ax[i].imshow(dividedtrials_avgtrials[i], cmap='cividis') #cmap='viridis'
        ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[0].suptitle('Neural activity for '+str(mouseidx))
    #plt.show()

    
    ax = axx[1].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials_avgtrials)):
        if i == 4:
            if len(dividedtrials[i]) != 0:
                ax[i].plot(dividedtrials_beh[i][:,:,0].T)
        else:
            if len(dividedtrials[i]) != 0:
                ax[i].plot(dividedtrials_beh[i][:,:,1].T)
        ax[i].axvline(x=mid, color='r')
        ax[i].set_ylim([-250, 250])
        ax[i].axhline(y=-25,color='k')
        ax[i].axhline(y=25,color='k')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[1].suptitle('Behavior activity for mouse '+str(mouseidx))
    #plt.show()

    ax = axx[2].subplots(ncols=2, nrows=3)
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(dividedtrials)):
        if len(dividedtrials[i]) != 0:
            ax[i].plot(dividedtrials_beh[i][:,:,0].T)
        ax[i].axvline(x=mid, color='r')
        ax[i].set_title(titles[i])
        #fig.colorbar(x, ax=ax[i],shrink=0.5)
        # plt.colorbar(x)
    axx[2].suptitle('Behavior activity for mouse '+str(mouseidx))
    plt.show()

    return dividedtrials_avgtrials, dividedtrials_beh, check_flag, dividedtrials, dividedtrials_beh


def get_raw_beh(l_data, l_data_25,l_beh_lowdim_cont):
    behaviour_withinarm = [None]*len(l_data)
    neural_withinarm = [None]*len(l_data)
    
    behaviour_trials = [None]*len(l_data)
    neural_trials = [None]*len(l_data)
    neural_trials_25 = [None]*len(l_data)
    beh_timepoints = [None]*len(l_data)
    if len(l_data_25) == 0:
        print('no 25ms')
        l_data_25 = [None]*len(l_data)

    for i in range(len(l_data)):
        #print('yay')
        
        behaviour_trials[i], neural_trials[i],neural_trials_25[i], behaviour_withinarm[i],neural_withinarm[i], beh_timepoints[i]  = divide_behavior(l_data[i],l_data_25[i], l_beh_lowdim_cont[i])
    return behaviour_trials,neural_trials, neural_trials_25, behaviour_withinarm, neural_withinarm, beh_timepoints


def plot_raw_beh(l_mouse_name, behaviour_trials):
    fig, ax = plt.subplots(int(len(l_mouse_name)/4), int(len(l_mouse_name)/3), figsize=(25,15))
    ax = ax.flatten()
    for j in range(len(behaviour_trials)):
        for i in range(len(behaviour_trials[j])):
            ax[j].plot(behaviour_trials[j][i][:,1])
        ax[j].set_title(l_mouse_name[j])
    plt.suptitle('Divided Beh for mouse '+str(l_mouse_name[j]))
    plt.show()

    trialnums = []
    for i in range(len(l_mouse_name)):
        trialnums.append(len(behaviour_trials[i]))
    trialnums = np.array(trialnums)

    plt.figure(figsize=(20,5))
    plt.bar(l_mouse_name, trialnums)
    plt.ylabel('Trial #')
    plt.title('Total number of trials in EPM per session')
    plt.show()

def plot_raw_beh_withinarms(l_mouse_name, behaviour_withinarm):

    fig, ax = plt.subplots(int(len(l_mouse_name)/4), int(len(l_mouse_name)/3), figsize=(25,15))
    ax = ax.flatten()
    for j in range(len(behaviour_withinarm)):
        for k in range(len(behaviour_withinarm[j])):
            for i in range(len(behaviour_withinarm[j][k])):
                ax[j].plot(behaviour_withinarm[j][k][i][:,1])
        ax[j].set_title(l_mouse_name[j])
    plt.suptitle('Divided Beh within arm for mouse '+str(l_mouse_name[j]))
    plt.show()

    trialnums = [None]*2
    for i in range(2):
        trialnums[i] = []

    for i in range(len(l_mouse_name)):
        
        trialnums[0].append(len(behaviour_withinarm[i][0]))
        trialnums[1].append(len(behaviour_withinarm[i][1]))


    plt.figure(figsize=(20,5))
    plt.bar(l_mouse_name, trialnums[0], label = 'OA')
    plt.bar(l_mouse_name, trialnums[1], label = 'CA')
    plt.ylabel('Trial #')
    plt.title('Total number of beh within arms trials in EPM per session')
    plt.legend()
    plt.show()


def plot_comparedtraces_withinarm(l_mouse_name, trialsOA, trialsCA):
    fig, ax = plt.subplots(int(len(l_mouse_name)/4), int(len(l_mouse_name)/3), figsize=(25,15))
    ax = ax.flatten()
    for j in range(len(trialsOA)):
        for i in range(len(trialsOA[j])):
            ax[j].plot(trialsOA[j][i], c='r')
        for i in range(len(trialsCA[j])):
            ax[j].plot(trialsCA[j][i], c='g')
        ax[j].set_title(l_mouse_name[j])
    plt.suptitle('Divided Beh within arm for mouse '+str(l_mouse_name[j]))
    plt.show()


    trialnums = [None]*2
    for i in range(2):
        trialnums[i] = []

    for i in range(len(l_mouse_name)):
        
        trialnums[0].append(len(trialsOA[i]))
        trialnums[1].append(len(trialsCA[i]))


    fig, ax = plt.subplots(1,1,figsize=(20,5))
    plt.bar(np.linspace(0,len(l_mouse_name)*2,12, endpoint=False), trialnums[1], label = 'CA')
    ax.set_xticks(np.linspace(0,len(l_mouse_name)*2,12, endpoint=False))
    ax.set_xticklabels(l_mouse_name)
    plt.bar(np.linspace(1,len(l_mouse_name)*2-1, 12, endpoint=True), trialnums[0], label = 'OA')
    plt.ylabel('Trial #')

    plt.title('Total number of beh within arms trials in EPM per session')
    plt.legend()
    plt.show()
 
def get_preprocessed_trials(l_mouse_name, behaviour_trials, neural_trials,neural_trials_25, beh_timepoints, trialtype):
    allmice_behavior_avgtrials = [None]*5
    allmice_neural_avgtrials = [None]*5
    allmice_dividedtrials = [None]*5
    allmice_neural_avgtrials_25 = [None]*5
    allmice_dividedtrials_25 = [None]*5
    allmice_dividedtrials_beh = [None]*5
    trialtimes = [None]*len(l_mouse_name)
    if neural_trials_25 is not None:
        dat_25 = True

    for i in range(len(allmice_behavior_avgtrials)):
        allmice_behavior_avgtrials[i] = np.array([])
        allmice_neural_avgtrials[i] = np.array([])
        allmice_dividedtrials[i] = []
        allmice_neural_avgtrials_25[i] = np.array([])
        allmice_dividedtrials_25[i] = []
        allmice_dividedtrials_beh[i] = []
        

    for i in range(len(l_mouse_name)):
        if trialtype == 'entryexit':
            dividedtrials_avgtrials, dividedtrials_avgtrials_25, dividedtrials_beh_avgtrials, check_flag,dividedtrials, dividedtrials_25, dividedtrials_beh, trialtimes[i] = heatmaps_matrixgen_avgtrials_old(behaviour_trials=behaviour_trials, neural_trials=neural_trials,mouseidx=i,beh_timepoints=beh_timepoints, neural_trials_25=neural_trials_25)
        if trialtype== '8t':
            dividedtrials_avgtrials , dividedtrials_beh_avgtrials, check_flag,dividedtrials, dividedtrials_beh, trialtimes[i] = heatmaps_matrixgen_avgtrials_8trialtypes(behaviour_trials=behaviour_trials, neural_trials=neural_trials,mouseidx=i)
        #trialsOA, trialsCA = compare_traces(trials1=behaviour_withinarm[i][0], trials2=behaviour_withinarm[i][1], thr=0.05)
        print(check_flag)
        if check_flag != 1:
            
            for j in range(len(allmice_behavior_avgtrials)):
                if len(allmice_behavior_avgtrials[j]) == 0:
                    allmice_behavior_avgtrials[j] = dividedtrials_beh_avgtrials [j]
                    allmice_neural_avgtrials[j] = dividedtrials_avgtrials[j]
                    allmice_dividedtrials[j].append(dividedtrials[j])
                    allmice_neural_avgtrials_25[j] = dividedtrials_avgtrials_25[j]
                    allmice_dividedtrials_25[j].append(dividedtrials_25[j])
                    allmice_dividedtrials_beh[j].append(dividedtrials_beh[j])
                else:
                
                    allmice_behavior_avgtrials[j] = np.vstack((allmice_behavior_avgtrials[j],dividedtrials_beh_avgtrials[j])) # dividedtrials_beh[j]
                    allmice_neural_avgtrials[j] = np.vstack((allmice_neural_avgtrials[j],dividedtrials_avgtrials[j])) # dividedtrials[j]
                    allmice_dividedtrials[j].append(dividedtrials[j])
                    allmice_neural_avgtrials_25[j] = np.vstack((allmice_neural_avgtrials_25[j],dividedtrials_avgtrials_25[j])) # dividedtrials[j]
                    allmice_dividedtrials_25[j].append(dividedtrials_25[j])
                    allmice_dividedtrials_beh[j].append(dividedtrials_beh[j])
    
    return  allmice_behavior_avgtrials, allmice_neural_avgtrials,allmice_neural_avgtrials_25, allmice_dividedtrials, allmice_dividedtrials_25, allmice_dividedtrials_beh, trialtimes


# z scoring

def z_score(array, ignoreunits, window_bin = 0.05):
   # if array.shape[-1]>array.shape[0]:
    array = array.T
    mid = int(array.shape[0]/2) # time length
    # baseline : wrt t=0sec (critical point of the trial)
    st = -3#-0.75
    ed = -1#-0.25
    onethird = mid + int(st/window_bin) #int(mid/3)
    twothird = mid + int(ed/window_bin) #int(mid/3)*2
    #print(array.shape)
    baseline = array[onethird:twothird, :]
    #print(onethird)
    #print(twothird)
    mean_ = np.nanmean(baseline, axis=0)
    std_ = np.nanstd(baseline, axis=0)
    # remove units with baseline std = 0 
    for i in range(std_.size):
        #print(std_[i])
        if std_[i] == 0.0:
            #print('yay')
            ignoreunits.append(i)
    zscore_ = (array - mean_)
    heat_map = (zscore_ / std_)
    
    #print(ignoreunits)
    return heat_map.T, ignoreunits

def z_score_headdips(avg_neur_allsess, ignoreunits, window_bin = 0.05):
    avg_neur_allsess = avg_neur_allsess.T
    mid = int(avg_neur_allsess.shape[0]/2)+1
    st = mid + int(-3/window_bin) #-1.5 - old
    ed = mid+ int(-1/window_bin) # -0.25 - old
    means = np.nanmean(np.array(avg_neur_allsess[st:ed,:]), axis=0)
    print(means.shape)
    std_ = np.nanstd(np.array(avg_neur_allsess[st:ed,:]), axis = 0)
    for i in range(std_.size):
        #print(std_[i])
        if std_[i] == 0.0:
            #print('yay')
            ignoreunits.append(i)
    zscored_neur_headdips = (avg_neur_allsess - means)/std_
    return zscored_neur_headdips.T, ignoreunits

def orderHM(hm, vmax, vmin, needorder=0):
    mid = int(hm.shape[0]/2) # time length
    st = mid + int(0/0.05)
    ed = mid + int(2/0.05)
    #stdd = np.std(hm)
    #m = np.mean(hm)
    #hm2 = (hm-m)/stdd
    #hm1 = hm
    #hm2 = np.array([])
    #for i in range(hm.shape[0]):
    #    max = np.max(hm[i,:])
    #    min = np.min(hm[i,:])
    #    xx = (hm[i,:]-min)/(max-min)
    #    xxx = xx*(vmax-vmin) + vmin
    #    if i == 0:
    #        hm2 = xx
    #    else:
    #        hm2 = np.vstack((hm2, xx))
    #plt.imshow(hm2)
    #plt.show()
    avgs = np.nanmean(hm[:,st:], axis=-1)
    idx = np.argsort(avgs)#[::-1].reverse() # reverse array after sorting it
    #print(idx)
    hm1 = hm[idx,:][::-1]
    if needorder == 1:
        return hm1, idx
    else:
        return hm1



def z_score_orderedhm(allmice_neural_avgtrials, window_bin = 0.05, trialtype='entry_exit', savefig=False):
    vmin= -2
    vmax = 3
    ignoreunits = []
    allMice_neural_zscored_perunit = [None]*len(allmice_neural_avgtrials)
    for i in range(len(allmice_neural_avgtrials)):
        if i<4:
            allMice_neural_zscored_perunit[i], ignoreunits = z_score(allmice_neural_avgtrials[i], ignoreunits, window_bin=window_bin)
        else:
            allMice_neural_zscored_perunit[i], ignoreunits = z_score_headdips(allmice_neural_avgtrials[i], ignoreunits, window_bin=window_bin)
    # post processing 
    #print(allMice_neural_zscored_perunit[0].shape)
    #print(ignoreunits)
    ignoreunits = np.unique(np.array(ignoreunits))#.sort()
    #print(ignoreunits)

    if len(ignoreunits) != 0:
        for i in range(len(allmice_neural_avgtrials)):
            #print(allMice_neural_zscored_perunit[i].shape)
            allMice_neural_zscored_perunit[i] = np.delete(allMice_neural_zscored_perunit[i], ignoreunits, axis=0)


    fig,ax = plt.subplots(ncols=2, nrows=3, figsize=(15,15))
    ax=ax.flatten().flatten() # i have trust issues
    for i in range(len(allMice_neural_zscored_perunit)):
        
        #allMice_neural_zscored_perunit[i] = orderHM(allMice_neural_zscored_perunit[i], vmax, vmin)
        x = ax[i].imshow(allMice_neural_zscored_perunit[i], cmap='cividis',vmin=vmin, vmax=vmax) #cmap='viridis'
        ax[i].axvline(x=int(allMice_neural_zscored_perunit[i].shape[1]/2), color='r')
        fig.colorbar(x, ax=ax[i])
        # plt.colorbar(x)
    if trialtype == "8t":
        ax[0].set_title('OA - OA')
        ax[1].set_title('CA - CA')
        ax[2].set_title('OA - CA')
        ax[3].set_title('CA - OA')
        plt.suptitle('Neural activity for all mice')
    else:
        ax[0].set_title('OA Entry')
        ax[1].set_title('OA Exit')
        ax[2].set_title('CA Entry')
        ax[3].set_title('CA Exit')
        plt.suptitle('Neural activity for all mice')
    if savefig==True:
        plt.savefig('PCA_finalpicturesNew/Heatmaps/Heatmap_EPM.svg')#check the path if used 
    plt.show()
    return allMice_neural_zscored_perunit, ignoreunits


def avg_sem_obtainer(ehm, sig=3):
    ehm = np.array(ehm)
    avg_ehm = gaussian_filter1d(np.mean(ehm, axis=0), sig)
    semm = gaussian_filter1d(sem(ehm, axis=0), sig)
    semm_pos = avg_ehm + semm
    semm_neg = avg_ehm - semm
    return avg_ehm, semm_pos, semm_neg

def unit_psth_plotter(ax,avg_ehm, semm_pos, semm_neg, t, title, trialtype,ylim_manual = [-1.5, 1.5]):
    
    ax.plot(avg_ehm, c='k')
    ax.plot(semm_pos, c='grey')
    ax.plot(semm_neg, c='grey')
    mid = int(avg_ehm.shape[-1]/2)
    ax.axvline(x=mid, c='r')
    ax.set_xlabel('Time')
    ax.set_title(title)
    t_show = np.arange(-3,4,1)
    
    t = np.round(np.linspace(-3.25, 3.25, avg_ehm.shape[-1], endpoint=True),2)
    ticknum = len(t_show)
    idx_ticks = []
    for i in range(ticknum):
        idx_ticks.append(np.where(t<=t_show[i])[-1][-1])
        
    idx_ticks = np.array(idx_ticks).squeeze()
    #idx_ticks =np.where(t==t_show)[0] #np.linspace(0,avg_ehm.shape[-1], ticknum)
    ax.set_xticks(idx_ticks)
    ax.set_xticklabels(t_show)
    ax.set_ylim(ylim_manual)
    '''
    if trialtype=="Headdips":
        ax.set_ylim([-2,2])
    else:
        ax.set_ylim([-1.5,1.5])
    '''
    ax.set_ylim(ylim_manual)

def ehm_ihm_rhm_pertrial(neural, trialtype, savefigs=False):
    j = 0
    rhm = []
    ehm = []
    ihm = []
    ehm_idx = []
    ihm_idx = []
    rhm_idx = []
    mid = int(neural.shape[-1]/2)
    st = mid 
    ed = mid + int(1/0.05)
    for i in range(neural.shape[0]):
        comp = np.mean(neural[i,st:ed])
        if comp>0.5:
            #print('wahoo')
            ehm.append(neural[i,:])
            ehm_idx.append(i)

        elif comp<-0.5:
            #print('yippee')
            ihm.append(neural[i,:])
            ihm_idx.append(i)
        else:
            rhm.append(neural[i,:])
            rhm_idx.append(i)

    avg_ehm, semm_pos_ehm, semm_neg_ehm = avg_sem_obtainer(ehm)
    avg_ihm, semm_pos_ihm, semm_neg_ihm = avg_sem_obtainer(ihm)
    avg_rhm, semm_pos_rhm, semm_neg_rhm = avg_sem_obtainer(rhm)
    plt.plot(avg_ehm)
    plt.plot(avg_ihm)
    plt.plot(avg_rhm)
    plt.title('Excited, Inhibited, Neutral units, '+trialtype)
    if savefigs == True:
        plt.savefig("PCA_finalpicturesNew/PSTH/PSTH_combinedUnittypes.svg")#check the path if used
    plt.show()

    fig, ax = plt.subplots(1,3, figsize = (15, 5))
    ax = ax.flatten()
    t = np.linspace(-3.25, 3.25, avg_ehm.size)

    for i in range(ax.size):
        unit_psth_plotter(ax[0],avg_ehm, semm_pos_ehm, semm_neg_ehm, t, title='Excited', trialtype=trialtype)
        unit_psth_plotter(ax[1],avg_ihm, semm_pos_ihm, semm_neg_ihm, t, title='Inhibited', trialtype=trialtype)
        unit_psth_plotter(ax[2],avg_rhm, semm_pos_rhm, semm_neg_rhm, t, title='Neutral', trialtype=trialtype)
        ax[0].set_ylabel('Z-score')
        
    plt.suptitle(trialtype)
    if savefigs == True:
        plt.savefig("PCA_finalpicturesNew/PSTH/PSTH_"+trialtype+".svg")#check the path if used 
    plt.show()

    # pie charts
    total = neural.shape[0]
    ehmperc = len(ehm)*100/total
    ihmperc = len(ihm)*100/total
    rhmperc = len(rhm)*100/total
    plt.pie([ehmperc, ihmperc, rhmperc], autopct='%1.1f%%')
    plt.legend(["ex", "inh", "neut"])
    plt.title(trialtype)
    if savefigs == True:
        plt.savefig("PCA_finalpicturesNew/PieCharts/PSTH_unitperc"+trialtype+".svg")#check the path if used
    plt.show()

    return np.array(ehm), np.array(ihm), np.array(rhm), np.array(ehm_idx), np.array(ihm_idx), np.array(rhm_idx)

def ehm_ihm_rhm_pertrial_new(neural, trialtype, window_bin=0.05, savefigs=True, dir=[]):
    j = 0
    rhm = []
    ehm = []
    ihm = []
    ehm_idx = []
    ihm_idx = []
    rhm_idx = []
    mid = int(neural.shape[-1]/2)
    st = mid 
    ed = mid + int(3/window_bin)
    z_exc = 1.65
    z_inh = -1.65
    for i in range(neural.shape[0]):
        # the idea is to take a sliding window and check if all the values
        # in the sliding window are greater than z=1
        # size of sliding window = 4 (200ms)
        window = 4#int(0.200/window_bin)
        counter = 0
        exc_c = 0
        inh_c = 0
        for j in range(neural[i,st:ed-window].size):
            comp = neural[i,st+j:st+j+window]
            if all(c>z_exc for c in comp):
                exc_c+=1
                # print('wahoo')
                # print(comp)
                # ehm.append(neural[i,:])
                # ehm_idx.append(i)
                # break

            if all(c<z_inh for c in comp):
                inh_c +=1
                # print('yippee')
                # print(comp)
                # ihm.append(neural[i,:])
                # ihm_idx.append(i)
                # break
            counter+=1

        if (exc_c>0 and inh_c>0) or (exc_c>0 and inh_c==0):
            ehm.append(neural[i,:])
            ehm_idx.append(i)
        
        elif inh_c>0 and exc_c==0:
            ihm.append(neural[i,:])
            ihm_idx.append(i)

        elif counter==neural[i,st:ed-4].size:
            rhm.append(neural[i,:])
            rhm_idx.append(i)
        else:
            print()

    avg_ehm, semm_pos_ehm, semm_neg_ehm = avg_sem_obtainer(ehm)
    avg_ihm, semm_pos_ihm, semm_neg_ihm = avg_sem_obtainer(ihm)
    avg_rhm, semm_pos_rhm, semm_neg_rhm = avg_sem_obtainer(rhm)
    plt.plot(avg_ehm)
    plt.plot(avg_ihm)
    plt.plot(avg_rhm)
    plt.title('Excited, Inhibited, Neutral units, '+trialtype)
    if savefigs == True:
        if len(dir) == 0:
            plt.savefig("RESULTS/NOT-INJECTED/PCA_finalpicturesNew/PSTH_combinedUnittypes.svg")
        else:
            plt.savefig(dir+"/PSTH_combinedUnittypes.svg")
    plt.show()

    fig, ax = plt.subplots(1,3, figsize = (15, 5))
    ax = ax.flatten()
    t = np.linspace(-3.25, 3.25, avg_ehm.size)

    for i in range(ax.size):
        unit_psth_plotter(ax[0],avg_ehm, semm_pos_ehm, semm_neg_ehm, t, title='Excited', trialtype=trialtype)
        unit_psth_plotter(ax[1],avg_ihm, semm_pos_ihm, semm_neg_ihm, t, title='Inhibited', trialtype=trialtype)
        unit_psth_plotter(ax[2],avg_rhm, semm_pos_rhm, semm_neg_rhm, t, title='Neutral', trialtype=trialtype)
        ax[0].set_ylabel('Z-score')
        
    plt.suptitle(trialtype)
    if savefigs == True:
        if len(dir) == 0:
            plt.savefig("RESULTS/NOT-INJECTED/PCA_finalpicturesNew/PSTH_"+trialtype+".svg")
        else:
            plt.savefig(dir+"/PSTH_"+trialtype+".svg")
    plt.show()

    # pie charts
    total = neural.shape[0]
    ehmperc = len(ehm)*100/total
    ihmperc = len(ihm)*100/total
    rhmperc = len(rhm)*100/total
    
    colors = ["dodgerblue", "darkorange", "limegreen"]
    plt.pie([ehmperc, ihmperc, rhmperc], autopct='%1.1f%%', startangle = 90, colors=colors, counterclock=False)
    plt.legend(["ex", "inh", "neut"])
    plt.title(trialtype)
    if savefigs == True:
        if len(dir) == 0:

            plt.savefig("PCA_finalpicturesNew/PSTH_unitperc"+trialtype+".svg")
        else:
            plt.savefig(dir+"/PSTH_unitperc"+trialtype+".svg")
    plt.show()


    

    return np.array(ehm), np.array(ihm), np.array(rhm), np.array(ehm_idx), np.array(ihm_idx), np.array(rhm_idx)


def plotheatmap_orderOAentry(ehm, ihm, rhm,allmice_neural_avgtrials_zscored, trialtitle, trialsubtypes, savefig=False):
    # 4 types per ehm, ihm, rhm 
    ehm_ordered = []
    ihm_ordered = []
    rhm_ordered = []
    allunits = []
    mid = int(ehm[0].shape[1]/2)
    t_show = np.arange(-3,4,1)
    ticknum = len(t_show)
    t = np.round(np.linspace(-3.25, 3.25, ehm[0].shape[-1], endpoint=True),2)
    idx_ticks = []
    for i in range(ticknum):
        idx_ticks.append(np.where(t==t_show[i]))
    idx_ticks = np.array(idx_ticks).squeeze()
    
    #print(idx_ticks)
    #print(t_show)
    vmin = -2
    vmax = 4
    for i in range(len(ehm)):
        #print(i)
        if i == 0:
            ehmx, idxehm = orderHM(ehm[i], vmin=vmin, vmax=vmax, needorder=1)
            ihmx, idxihm = orderHM(ihm[i], vmin=vmin, vmax=vmax, needorder=1)
            rhmx, idxrhm = orderHM(rhm[i], vmin=vmin, vmax=vmax, needorder=1)
            idxihm = np.array(idxihm + len(ehmx))
            idxrhm = np.array(idxrhm + len(ehmx) + len(ihmx))
            ehm_ordered.append(ehmx)
            ihm_ordered.append(ihmx)
            rhm_ordered.append(rhmx)
            
            idx_order = np.hstack((idxehm, idxihm, idxrhm))
            #print(np.unique(idx_order))
            allunitspertrial = np.vstack((ehm_ordered[i],ihm_ordered[i],rhm_ordered[i]))
            allunits.append(allunitspertrial)
        else:
            #ehmx = ehm[i][idxehm,:]
            #ihmx = ehm[i][idxihm,:]
            #rhmx = ehm[i][idxrhm,:]
            #ehm_ordered.append(ehmx)
            #ihm_ordered.append(ihmx)
            #rhm_ordered.append(rhmx)
            allunitspertrial = allmice_neural_avgtrials_zscored[i][idx_order,:]
            #allunitspertrial = np.vstack((ehm_ordered[i],ihm_ordered[i],rhm_ordered[i]))
            allunits.append(allunitspertrial)

    fig, ax = plt.subplots(2,3, figsize=(15,15))
    ax = ax.flatten()

    for i in range(len(allunits)):
        #print(ax.shape)
        x = ax[i].imshow(allunits[i], cmap='cividis', vmin=vmin, vmax=vmax) #cmap='viridis'
        fig.colorbar(x, ax=ax[i])
        #ehmnum = ehm[i].shape[0]
        #ihmnum = ihm[i].shape[0]
        #ax[i].axhline(ehmnum, c='k')
        #ax[i].axhline(ihmnum+ehmnum, c='k')
        
        ax[i].set_title(trialsubtypes[i])
        ax[i].set_xticks(idx_ticks)
        ax[i].set_xticklabels(t_show)
        ax[i].axvline(x=mid, c='r')
        ax[i].set_xlabel('Time (s)')

    ax[0].set_ylabel('# Unit')
    ax[2].set_ylabel('# Unit')
    plt.suptitle(trialtitle)
    if savefig == True:
        plt.savefig("PCA_finalpicturesNew/Heatmaps/Heatmap_EPM_orderedOAEntry.svg")#check path 
    plt.show()
    return allunits

def plotheatmap(ehm, ihm, rhm, trialtitle, trialsubtypes, savefig=False, cmap= 'cividis', vmax=4, vmin=-2, dir = []):
    # 4 types per ehm, ihm, rhm 
    ehm_ordered = []
    ihm_ordered = []
    rhm_ordered = []
    allunits = []
    mid = int(ehm[0].shape[1]/2)
    #t = np.linspace(-3,3,ehm[0].shape[1])
    t_show = np.arange(-3,4,1)
    ticknum = len(t_show)
    t = np.round(np.linspace(-3.25, 3.25, ehm[0].shape[-1], endpoint=True),2)
    idx_ticks = []
    for i in range(ticknum):
        idx_ticks.append(np.where(t<=t_show[i])[-1][-1])
    idx_ticks = np.array(idx_ticks).squeeze()
    #idx_ticks = np.linspace(0,ehm[0].shape[-1], ticknum)
    #print(idx_ticks)
    #print(t_show)
    
    for i in range(len(ehm)):
        #print(i)
        ehm_ordered.append(orderHM(ehm[i], vmin=vmin, vmax=vmax))
        ihm_ordered.append(np.flip(orderHM(ihm[i], vmin=vmin, vmax=vmax), axis=0))
        rhm_ordered.append(orderHM(rhm[i], vmin=vmin, vmax=vmax))
        allunitspertrial = np.vstack((ehm_ordered[i],ihm_ordered[i],rhm_ordered[i]))
        allunits.append(allunitspertrial)
    
    fig, ax = plt.subplots(2,3, figsize=(10,15))
    ax = ax.flatten()

    for i in range(len(allunits)):
        #print(ax.shape)
        x = ax[i].imshow(allunits[i], cmap=cmap, vmin=vmin, vmax=vmax) #cmap='viridis'
        fig.colorbar(x, ax=ax[i])
        ehmnum = ehm[i].shape[0]
        ihmnum = ihm[i].shape[0]
        ax[i].axhline(ehmnum, c='k')
        ax[i].axhline(ihmnum+ehmnum, c='k')
        ax[i].set_title(trialsubtypes[i])
        ax[i].set_xticks(idx_ticks)
        ax[i].set_xticklabels(t_show)
        ax[i].axvline(x=mid, c='r')
        ax[i].set_xlabel('Time (s)')

    ax[0].set_ylabel('# Unit')
    ax[2].set_ylabel('# Unit')
    plt.suptitle(trialtitle)
    if savefig == True:
        if len(dir) == 0:
            plt.savefig("RESULTS/NOT-INJECTED/PCA_finalpicturesNew/Heatmaps/Heatmap_EPM_ordered.svg")
        else:
            plt.savefig(dir+"/Heatmap_EPM_ordered.svg")
    plt.show()
    return allunits



def get_ehm_ihm_rhm_ranksum(allmice_dividedtrials, trialtypes):
    # done per unit - need trials for this 
    # x: actual
    # y: baseline
    # allmice_dividedtrials - 5 trials
    # allmice_dividedtrials[i] - 12 sessions
    # allmice_dividedtrials[i][j] - (trials, trialtime, units)
    unittype = [None]*len(allmice_dividedtrials)
    ehm = [None]*len(allmice_dividedtrials)
    ihm = [None]*len(allmice_dividedtrials)
    rhm = [None]*len(allmice_dividedtrials)
    ehm_num = [None]*len(allmice_dividedtrials)
    ihm_num = [None]*len(allmice_dividedtrials)
    rhm_num = [None]*len(allmice_dividedtrials)
    mid = int(allmice_dividedtrials[0][0].shape[1]/2)
    for i in range(len(allmice_dividedtrials)):
        unittype[i] = [None]*len(allmice_dividedtrials[i])
        ehm[i] = [None]*len(allmice_dividedtrials[i])
        ihm[i] = [None]*len(allmice_dividedtrials[i])
        rhm[i] = [None]*len(allmice_dividedtrials[i])
        ehm_num[i] = [None]*len(allmice_dividedtrials[i])
        ihm_num[i] = [None]*len(allmice_dividedtrials[i])
        rhm_num[i] = [None]*len(allmice_dividedtrials[i])
        
        for j in range(len(allmice_dividedtrials[i])):
            unittype[i][j] = np.zeros(allmice_dividedtrials[i][j].shape[-1]) #num units
            for k in range(len(unittype[i][j])):
                array = allmice_dividedtrials[i][j][:,:,k].T # (time, trials)
                st = -3#-0.75
                ed = -1#-0.25
                if i == len(allmice_dividedtrials): # different for headdips
                    st = -1.5
                    ed = -0.25
                onethird = mid + int(st/0.05) #int(mid/3)
                twothird = mid + int(ed/0.05) #int(mid/3)*2
                #print(array.shape)
                baseline_pertrial_1unit = np.mean(array[onethird:twothird, :], axis=0)
                firing_pertrial_1unit = np.mean(array[twothird:,:], axis=0)
                statistic, pval = wilcoxon(baseline_pertrial_1unit,firing_pertrial_1unit, alternative='two-sided')
               # print(pval)
                if pval>0.1:
                    unittype[i][j][k] = 0
                else:
                    statistic, pval = wilcoxon(baseline_pertrial_1unit,firing_pertrial_1unit, alternative='greater')
                    if pval>0.1: # inhibited
                        unittype[i][j][k] = 1
                    else: 
                        unittype[i][j][k] = -1
            
            
            ehm[i][j] = np.where(unittype[i][j] == 1)[0]
            ehm_num[i][j] = len(ehm[i][j])
            ihm[i][j] = np.where(unittype[i][j] == -1)[0]
            ihm_num[i][j] = len(ihm[i][j])
            rhm[i][j] = np.where(unittype[i][j] == 0)[0]
            rhm_num[i][j] = len(rhm[i][j])
            

    
    ehm_tot = np.zeros(len(allmice_dividedtrials))
    ihm_tot = np.zeros(len(allmice_dividedtrials))
    rhm_tot = np.zeros(len(allmice_dividedtrials))
    fig = plt.figure()
    for i in range(len(allmice_dividedtrials)): # per trialtype
        ehm_tot[i]= np.sum(unittype[0][i], axis=0)
        ihm_tot[i] = np.sum(unittype[1][i], axis=0)
        rhm_tot[i] = np.sum(unittype[2][i], axis=0)
        plt.pie([ehm_tot[i], rhm_tot[i], ihm_tot[i]], autopct='%1.1f%%')
        plt.legend(["ex", "rnh", "ihm"])
        plt.title(trialtypes[i])
        plt.show()
    return ehm_num, ihm_num, rhm_num, ehm, ihm, rhm  




# # signed ranksum for unittypes 
# from scipy.stats import wilcoxon


# # z score 1.65 is p-value of 0.1
# # z score 1 is p-value of 0.317
# def get_ehm_ihm_rhm_ranksum(allmice_dividedtrials, trialtypes):
#     # done per unit - need trials for this 
#     # x: actual
#     # y: baseline
#     # allmice_dividedtrials - 5 trials
#     # allmice_dividedtrials[i] - 12 sessions
#     # allmice_dividedtrials[i][j] - (trials, trialtime, units)
#     unittype = [None]*len(allmice_dividedtrials)
#     ehm = [None]*len(allmice_dividedtrials)
#     ihm = [None]*len(allmice_dividedtrials)
#     rhm = [None]*len(allmice_dividedtrials)
#     ehm_num = [None]*len(allmice_dividedtrials)
#     ihm_num = [None]*len(allmice_dividedtrials)
#     rhm_num = [None]*len(allmice_dividedtrials)
#     rhm_neural,ehm_neural,ihm_neural=[None]*len(allmice_dividedtrials),[None]*len(allmice_dividedtrials),[None]*len(allmice_dividedtrials)
#     mid = int((allmice_dividedtrials[0][0].shape[1]-5)/2)

#     for i in range(len(allmice_dividedtrials)):
        
#         rhm_neural[i],ehm_neural[i],ihm_neural[i] = [], [], []

#         unittype[i] = [None]*len(allmice_dividedtrials[i])
#         ehm[i] = [None]*len(allmice_dividedtrials[i])
#         ihm[i] = [None]*len(allmice_dividedtrials[i])
#         rhm[i] = [None]*len(allmice_dividedtrials[i])
#         ehm_num[i] = [None]*len(allmice_dividedtrials[i])
#         ihm_num[i] = [None]*len(allmice_dividedtrials[i])
#         rhm_num[i] = [None]*len(allmice_dividedtrials[i])
        
#         for j in range(len(allmice_dividedtrials[i])):
#             unittype[i][j] = np.zeros(allmice_dividedtrials[i][j].shape[-1]) #num units
#             for k in range(len(unittype[i][j])):
#                 array1 = np.copy(allmice_dividedtrials[i][j][:,:,k].T) # (time, trials)
#                 window = 5
                
#                 for t in range(array1.shape[0]):
#                     array1[t,:] = np.sum(array1[t:t+window,:], axis=0)
#                 array = array1[:-5,:]
#                 st = -3#-0.75
#                 ed = -1#-0.25
#                 st_post = 0
#                 ed_post = 1
#                 onethird = mid + int(st/0.05) #int(mid/3)
#                 twothird = mid + int(ed/0.05) #int(mid/3)*2
#                 post_0 = mid + int(st_post/0.05)
#                 post_1 = mid + int(ed_post/0.05)
#                 #print(array.shape)
#                 baseline_pertrial_1unit = np.mean(array[onethird:twothird, :], axis=0)
#                 firing_pertrial_1unit = np.mean(array[post_0:post_1,:], axis=0)
#                 statistic, pval = wilcoxon(firing_pertrial_1unit,baseline_pertrial_1unit, alternative='greater')
#                 #print(pval)
#                 if pval< 0.2: #excited
#                     unittype[i][j][k] = 1
#                     # print(np.mean(allmice_dividedtrials[i][j][:,:,k], axis=0).shape)
#                     ehm_neural[i].append((np.mean(array, axis=-1)))
#                 else:
#                     statistic, pval = wilcoxon(firing_pertrial_1unit,baseline_pertrial_1unit, alternative='less')
#                     if pval< 0.2: # inhibited
#                         unittype[i][j][k] = -1
#                         ihm_neural[i].append((np.mean(array, axis=-1)))
#                     else: 
#                         unittype[i][j][k] = 0
#                         rhm_neural[i].append((np.mean(array, axis=-1)))

#             ehm[i][j] = np.where(unittype[i][j] == 1)[0]
#             ehm_num[i][j] = len(ehm[i][j])
#             ihm[i][j] = np.where(unittype[i][j] == -1)[0]
#             ihm_num[i][j] = len(ihm[i][j])
#             rhm[i][j] = np.where(unittype[i][j] == 0)[0]
#             rhm_num[i][j] = len(rhm[i][j])
     
#     # ehm_neural = np.mean(np.array(ehm_neural), axis=0)
#     # ihm_neural = np.mean(np.array(ihm_neural), axis=0)
#     # rhm_neural = np.mean(np.array(rhm_neural), axis=0)  

#     return np.array(ehm_num), np.array(ihm_num), np.array(rhm_num), ehm, ihm, rhm ,ehm_neural,rhm_neural,ihm_neural

# trialsubtypes_entryexit=["OA Entry", "OA Exit", "CA Entry", "CA Exit", "Headdips"]
# ehm_num_wil, ihm_num_wil, rhm_num_wil, ehm_wil, ihm_wil, rhm_wil,ehm_neural_wil,rhm_neural_wil,ihm_neural_wil = get_ehm_ihm_rhm_ranksum(allmice_dividedtrials.copy(), trialsubtypes_entryexit)


