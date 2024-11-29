# Has functions for dataloading, dataset loading (only relevant for pytorch builds) and plotter function

import numpy as np
import torch
import cebra
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# from poissonprocessNeuralDataGen import gen_alt_binomial


def plotLoss(ax, modelname, c,micenum):
    model = torch.load(modelname)
    n=0
    for i in range(len(list(modelname))):
        if list(modelname)[i] == '/':
            n = i
    legendname = modelname[0:n]
    
    for i in range(len(list(modelname))):
        if list(modelname)[i] == '_':
            n = i
    offsetname = modelname[0:n]
    #make_interp_spline(x, y)
    #iterations = np.linspace()
    #print(ax)
    ax.plot(model['loss'],linewidth=0.5, color = c, label=legendname)
    ax.grid(False)
    ax.set_ylabel('Loss')
    ax.set_xlabel('iter')
    ax.set_title(offsetname+' loss over '+str(micenum)+' mice')
    ax.legend()
    #ax.zaxis.pane.set_edgecolor('w')
        
    return ax


def check_fr(neural):
    ignore = []
    mean_fr = np.nanmean(neural, axis=0)/0.05 # mean fr per unit
    binvec = np.arange(0, np.max(mean_fr), 2)
    #plt.hist(mean_fr, bins=binvec)
    #plt.show()
    for i in range(mean_fr.size):
        #print(mean_fr[i])
        if mean_fr[i]<0.5:
            #print(i)
            ignore.append(i)
    return ignore

def binshuffle_15min(l_data, binwidth=0.05):
    np.random.seed(42)
    l_data_sh = []
    for i in range(len(l_data)):
        neural = np.array(l_data[i])
        summ = np.sum(neural)
        timep = l_data[i].shape[0]
        freq = 30000
        #t_ignore = int(30*0.05*30000)
        timeb_total = int(timep*freq*binwidth)# - t_ignore
        y = np.zeros((timeb_total, neural.shape[-1])).astype(int)
        for i in range(neural.shape[-1]):
            summ = np.sum(neural[:,i])
            t = (timeb_total*1)
            y[:,i] = np.random.binomial(n=1, p=summ/t, size=(timeb_total)).astype(int)
        new= np.zeros((neural.shape[0], neural.shape[-1]))
        c=int(30000*binwidth)
        #print(new.shape)
        for i in range(new.shape[0]):
            new[i,:] =np.sum(y[i*c:(i+1)*c,:], axis=0)
        
        l_data_sh.append(new)
        del new
    return l_data_sh

def load_datavariables(l_mouse_name, l_date, datatype= 0, sorter = "tdc", l_data_25_bool = False):
    # datatype 0: no headdip idx
    # datatype 1:includes headdip idx
    # extract data 
    # very important for cebra analysis : (l_data, l_beh_lowdim_cont) and (l_data_shuffled, l_beh_lowdim_shuffled_cont)

    np.random.seed(1)

    l_data = []
    l_beh= []
    l_beh_lowdim = []
    l_x_open= []
    l_y_open= []
    l_x_closed= []
    l_y_closed= []
    l_x_center= []
    l_y_center= []
    origin = []
    l_data_25 = []

    for i in range(len(l_mouse_name)):

        if datatype == 1: # headdip information included
            if sorter =="kilo":
                with np.load('NPZ_FILES_EPM/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_ALLIDX_kilo.npz') as f:
                    l_data.append(f['l_data'])
                    l_beh.append(f['l_beh'])
                    l_beh_lowdim.append(f['l_beh_lowdim'])
                    l_x_open.append(f['l_x_open'])
                    l_y_open.append(f['l_y_open'])
                    l_x_closed.append(f['l_x_closed'])
                    l_y_closed.append(f['l_y_closed'])
                    l_x_center.append(f['l_x_center'])
                    l_y_center.append(f['l_y_center'])
                    origin.append(f['origin'])
                    if l_data_25_bool == True:
                        # print('25 ms')
                        l_data_25.append(f['l_data_25'])
                    else:
                        # print('no 25 ms')
                        l_data_25 = []


        if datatype == 0: # headdip within OA
            if sorter == "kilo":
                with np.load('NPZ_FILES_EPM/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_check_kilo.npz') as f:
                    l_data.append(f['l_data'])
                    l_beh.append(f['l_beh'])
                    l_beh_lowdim.append(f['l_beh_lowdim'])
                    l_x_open.append(f['l_x_open'])
                    l_y_open.append(f['l_y_open'])
                    l_x_closed.append(f['l_x_closed'])
                    l_y_closed.append(f['l_y_closed'])
                    l_x_center.append(f['l_x_center'])
                    l_y_center.append(f['l_y_center'])
                    origin.append(f['origin'])
                    if l_data_25_bool:
                        l_data_25.append(f['l_data_25'])
                    else:
                        l_data_25 = []
 
    # Flip dimensions for CEBRA (x axis - time, y axis - neural dim)
    l_beh_shuffled = len(l_mouse_name)*[None]
    l_beh_shuffled_highdim = len(l_mouse_name)*[None]
    #l_data_shuffled = len(l_mouse_name)*[None]
    #l_data_shuffled = len(l_mouse_name)*[None]
    for i in range(len(l_mouse_name)):
        ignore = check_fr(l_data[i].T)
        # print("Ignore units for mouse", l_mouse_name[i])
        print(ignore)
        
        #ignore1 = check_fr(l_data_shuffled[i])
        if len(ignore ) != 0:
            print(i)
            print(l_data[i].shape)
            l_data[i] = np.delete(l_data[i].T, np.array(ignore), axis = -1)
            if len(l_data_25) != 0:
                l_data_25[i] = np.delete(l_data_25[i].T, np.array(ignore), axis = -1)
            print(l_data[i].shape)
        else: 
            l_data[i] = l_data[i].T
            if len(l_data_25) != 0:
                l_data_25[i] = l_data_25[i].T

        #if len(ignore1) != 0:
        #    l_data_shuffled[i] = np.delete(l_data_shuffled[i], np.array(ignore1), axis = -1)
    
    l_data_shuffled = binshuffle_15min(l_data.copy()) #gen_alt_binomial(l_data, l_mouse_name)
    # l_data_shuffled_25 = binshuffle_15min(l_data_25.copy(), binwidth=0.025)

    for i in range(len(l_mouse_name)):
        #print(i)
        l_data[i] = torch.from_numpy(l_data[i]).type(torch.FloatTensor)
        if len(l_data_25) != 0:
                
            l_data_25[i] = torch.from_numpy(l_data_25[i]).type(torch.FloatTensor)
        l_beh_lowdim[i] = torch.from_numpy(l_beh_lowdim[i].T).type(torch.FloatTensor)
        l_beh_shuffled[i]= np.copy(l_beh_lowdim[i]) 
        l_beh_shuffled[i][:,0] = np.random.permutation(l_beh_shuffled[i][:,0])
        l_beh_shuffled[i][:,1] = np.random.permutation(l_beh_shuffled[i][:,1])
        l_beh_shuffled[i] = torch.from_numpy(l_beh_shuffled[i]).type(torch.FloatTensor)
        

        # l_data is a list of neural data, with each element being a 2D array of size [time x num_units] for each session
        # this loop runs for each session idx 'i'
        
        # l_data_shuffled[i] = torch.tensor(l_data_shuffled[i]).type(torch.FloatTensor)
        l_beh[i] = torch.from_numpy(l_beh[i].T).type(torch.FloatTensor) # not necessary !
        l_beh_shuffled_highdim[i]= np.copy(l_beh[i]) # not necessary !
        l_beh_shuffled_highdim[i][:,0] = np.random.permutation(l_beh_shuffled_highdim[i][:,0]) # not necessary !
        l_beh_shuffled_highdim[i][:,1] = np.random.permutation(l_beh_shuffled_highdim[i][:,1]) # not necessary !
        l_beh_shuffled_highdim[i][:,2] = np.random.permutation(l_beh_shuffled_highdim[i][:,2]) # not necessary !
        l_beh_shuffled_highdim[i] = torch.from_numpy(l_beh_shuffled_highdim[i]).type(torch.FloatTensor) # not necessary !
    
    l_beh_lowdim_cont = l_beh_lowdim.copy() # not necessary !

    for i in range(len(l_mouse_name)):
        #idx_90 = l_beh_lowdim_cont[i][:,0] == 2
        #idx_180 = l_beh_lowdim_cont[i][:,0] == 1
        #idx_270 = l_beh_lowdim_cont[i][:,0] == 3
        #idx_centre_4 = l_beh_lowdim_cont[i][:,0] == 4 
        #idx_centre_5 = l_beh_lowdim_cont[i][:,0] == 5 
        #idx_centre_6 = l_beh_lowdim_cont[i][:,0] == 6 
        #idx_centre_7 = l_beh_lowdim_cont[i][:,0] == 7

        #l_beh_lowdim_cont[i][idx_90,0] = 1
        #l_beh_lowdim_cont[i][idx_180,0] = 2
        #l_beh_lowdim_cont[i][idx_270,0] = 3
        #l_beh_lowdim_cont[i][idx_centre_4,0] = 0
        #l_beh_lowdim_cont[i][idx_centre_5,0] = 2
        #l_beh_lowdim_cont[i][idx_centre_6,0] = 1
        #l_beh_lowdim_cont[i][idx_centre_7,0] = 3

        idx_ca2 = l_beh_lowdim_cont[i][:,0] == 3
        idx_oa2 = l_beh_lowdim_cont[i][:,0] == 1
        idx_ca1 = l_beh_lowdim_cont[i][:,0] == 2
        idx_centre_4 = l_beh_lowdim_cont[i][:,0] == 4
        idx_centre_5 = l_beh_lowdim_cont[i][:,0] == 5
        idx_centre_6 = l_beh_lowdim_cont[i][:,0] == 6 
        idx_centre_7 = l_beh_lowdim_cont[i][:,0] == 7
        l_beh_lowdim_cont[i][idx_centre_6,1] = - l_beh_lowdim_cont[i][idx_centre_6,1]
        l_beh_lowdim_cont[i][idx_centre_7,1] = - l_beh_lowdim_cont[i][idx_centre_7,1]
        l_beh_lowdim_cont[i][idx_ca2,1] = - l_beh_lowdim_cont[i][idx_ca2,1]
        l_beh_lowdim_cont[i][idx_ca1,1] = - l_beh_lowdim_cont[i][idx_ca1,1]
        l_beh_lowdim_cont[i][idx_ca2,0] = 2
        l_beh_lowdim_cont[i][idx_oa2,0] = 0
        #time = np.linspace(0,l_beh_lowdim_cont[i].shape[0])
        #print(l_beh_lowdim_cont[i].shape)
        #plt.scatter(time, l_beh_lowdim_cont[i])

    l_beh_shuffled_cont = l_beh_shuffled.copy()
    for i in range(len(l_mouse_name)):

        idx_ca2 = l_beh_lowdim_cont[i][:,0] == 3
        idx_ca1 = l_beh_lowdim_cont[i][:,0] == 2
        idx_centre_4 = l_beh_lowdim_cont[i][:,0] == 4
        idx_centre_5 = l_beh_lowdim_cont[i][:,0] == 5
        idx_centre_6 = l_beh_lowdim_cont[i][:,0] == 6 
        idx_centre_7 = l_beh_lowdim_cont[i][:,0] == 7
        l_beh_shuffled_cont[i][idx_centre_6,1] = - l_beh_shuffled_cont[i][idx_centre_6,1]
        l_beh_shuffled_cont[i][idx_centre_7,1] = - l_beh_shuffled_cont[i][idx_centre_7,1]
        l_beh_shuffled_cont[i][idx_ca2,1] = - l_beh_shuffled_cont[i][idx_ca2,1]
        l_beh_shuffled_cont[i][idx_ca1,1] = - l_beh_shuffled_cont[i][idx_ca1,1]
        time = np.linspace(0,l_beh_shuffled_cont[i].shape[0])
        
    l_beh_lowdim_cont = simpleEPMcentre(l_beh=l_beh_lowdim_cont)
    l_beh_shuffled_cont = simpleEPMcentre(l_beh=l_beh_shuffled_cont)

    # removing units with firing rate less than 1Hz
    '''
    for i in range(len(l_mouse_name)):
        ignore = check_fr(l_data[i])
        ignore1 = check_fr(l_data_shuffled[i])
        if len(ignore ) != 0:
            print(i)
            print(l_data[i].shape)
            l_data[i] = np.delete(l_data[i], np.array(ignore), axis = -1)
            print(l_data[i].shape)
        if len(ignore1) != 0:
            l_data_shuffled[i] = np.delete(l_data_shuffled[i], np.array(ignore1), axis = -1)
    '''    
    
    return l_data, l_data_25, l_data_shuffled,[], l_beh_lowdim, l_beh_shuffled, l_beh, l_beh_lowdim_cont,l_beh_shuffled_cont



def simpleEPMcentre(l_beh):
    # can be used for all beh arrays with a armidx dimension
    for i in range(len(l_beh)):
        idx_centre_4 = l_beh[i][:,0] == 4
        idx_centre_5 = l_beh[i][:,0] == 5
        idx_centre_6 = l_beh[i][:,0] == 6 
        idx_centre_7 = l_beh[i][:,0] == 7
        l_beh[i][idx_centre_4,0] = 4
        l_beh[i][idx_centre_5,0] = 4
        l_beh[i][idx_centre_6,0] = 4
        l_beh[i][idx_centre_7,0] = 4

    return l_beh

def simpleEPMarms(l_beh):
    # can be used for all beh arrays with a armidx dimension
    for i in range(len(l_beh)):
        idx_ca2 = l_beh[i][:,0] == 3
        idx_OA2 = l_beh[i][:,0] == 1
        l_beh[i][idx_ca2,0] = 2
        l_beh[i][idx_OA2,0] = 0

    return l_beh

# HELPER FUNCTIONS - use for cebra's pytorch build

def multisess_dataset_loader(l_mouse_name, l_data, l_data_shuffled, l_beh_lowdim, l_beh_shuffled):
    Datasets = [None]*len(l_mouse_name)
    Datasets_sh = [None]*len(l_mouse_name)
    Datasets_timesh = [None]*len(l_mouse_name)

    for i in range(len(l_mouse_name)):
        neural = l_data[i].type(torch.FloatTensor)
        continuous = l_beh_lowdim[i][:,1:2]
        continuous_sh = l_beh_shuffled[i][:,1:2]
        #continuous_sh = (l_beh_shuffled[i][:,1:2])
        #continuous = (l_beh_lowdim[i][:,1:2])
        #discrete_sh = l_beh_shuffled[i][:,0].type(torch.LongTensor)
        #idx_neg_sh = np.logical_or(np.array(discrete_sh) == 2, np.array(discrete_sh) == 3)
        #continuous_sh[idx_neg_sh,0] = - continuous_sh[idx_neg_sh,0]
        Datasets[i] = cebra.data.TensorDataset(neural, continuous) # actual dataset
        Datasets_sh[i] = cebra.data.TensorDataset(neural, continuous_sh) # shuffled behavior dataset

        neural_sh = l_data_shuffled[i].type(torch.FloatTensor)
        
        #discrete = l_beh_lowdim[i][:,0].type(torch.LongTensor)
        #idx_neg = np.logical_or(np.array(discrete) == 2, np.array(discrete) == 3) # linearising the EPM to make continuous data 1D
        #continuous[idx_neg,0] = - continuous[idx_neg,0]
        
        Datasets_timesh[i] = cebra.data.TensorDataset(neural_sh, continuous)

        #idx_centre_not1 = np.logical_or(np.array(discrete) == 5, np.array(discrete) == 6)
        #idx_centre_not2 = np.logical_or(np.array(discrete) == 7, np.array(discrete) == 4)
        #idx_centre_not = np.logical_or(idx_centre_not1, idx_centre_not2)
        #idx_centre_not1_sh = np.logical_or(np.array(discrete_sh) == 5, np.array(discrete_sh) == 6)
        #idx_centre_not2_sh = np.logical_or(np.array(discrete_sh) == 7, np.array(discrete_sh) == 4)
        #idx_centre_not_sh = np.logical_or(idx_centre_not1_sh, idx_centre_not2_sh)
        #continuous[idx_centre_not] = 0
        #continuous_sh[idx_centre_not_sh] = 0

    Multisess_dataset = cebra.data.datasets.DatasetCollection(Datasets[i] for i in range(len(l_mouse_name)))
    Multisess_dataset_sh = cebra.data.datasets.DatasetCollection(Datasets_sh[i] for i in range(len(l_mouse_name)))

    Multisess_dataset_timesh = cebra.data.datasets.DatasetCollection(Datasets_timesh[i] for i in range(len(l_mouse_name)))

    return Multisess_dataset, Multisess_dataset_sh, Multisess_dataset_timesh

def PlotMultisess(sess,cebraposdir_dataset1_emb, cebratime_dataset1_emb,cebrashufflebeh_dataset1_emb,cebrashuffletime_dataset1_emb,pos_data, l_mouse_name, suffix):
    #pos_data = l_beh_lowdim[sess]

    for idx in range(4):
        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot(141, projection='3d')
        ax2 = plt.subplot(142, projection='3d')
        ax3 = plt.subplot(143, projection='3d')
        ax4 = plt.subplot(144, projection='3d')

        ax1=plot_insulaEPMsingleArm(ax1, cebraposdir_dataset1_emb, (pos_data),'y',idx)
        ax2=plot_insulaEPMsingleArm(ax2,  cebratime_dataset1_emb, (pos_data),'y',idx)
        ax3=plot_insulaEPMsingleArm(ax3, cebrashufflebeh_dataset1_emb, (pos_data),'y',idx)
        ax4=plot_insulaEPMsingleArm(ax4, cebrashuffletime_dataset1_emb, (pos_data),'x',idx)

        ax1.set_title('Spiking (e)')
        ax2.set_title('Spiking (isi, t)')
        ax3.set_title('Spiking (e_sh)')
        ax4.set_title('Spiking (isi, t_sh)')
        fig.suptitle('For arm idx '+str(idx)+', mouse '+ l_mouse_name[sess])
        plt.savefig('Plots\\PlotsNew\\PyTorch_multisess\\separatedPlots\\'+l_mouse_name[sess]+'_idx'+str(idx)+suffix+'.png')
        plt.show()


    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131, projection='3d')
    ax2 = plt.subplot(132, projection='3d')
    ax3 = plt.subplot(133, projection='3d')

    #ax1=plot_insulaEPMsingleArm(ax1, cebra_pos2dim_offset10, (pos_data_lowdim),'y',0)
    ax1=plot_insulaEPMsingleArm(ax1, cebraposdir_dataset1_emb, (pos_data),'y',0)
    ax2=plot_insulaEPMsingleArm(ax2, cebraposdir_dataset1_emb, (pos_data),'y',1)
    ax3 =plot_insulaEPMsingleArm(ax3, cebraposdir_dataset1_emb, (pos_data),'y',0)
    ax3 =plot_insulaEPMsingleArm(ax3, cebraposdir_dataset1_emb, (pos_data),'y',1)

    ax1.set_title('idx 0 - open arm')
    ax2.set_title('idx 1 - open arm')
    ax3.set_title('combined OA')
    fig.suptitle('Mouse '+l_mouse_name[sess])
    plt.savefig('Plots\\PlotsNew\\PyTorch_multisess\\Combined_openArm\\'+l_mouse_name[sess]+'_OA'+suffix+'.png')
    #plt.savefig('Plots\\PlotsNew\Combined_openArm\\'+l_mouse_name[sess]+'_OA.png')
    plt.show()

    fig = plt.figure(figsize=(15,5))

    ax1 = plt.subplot(131, projection='3d')
    ax2 = plt.subplot(132, projection='3d')
    ax3 = plt.subplot(133, projection='3d')

    #ax1=plot_insulaEPMsingleArm(ax1, cebra_pos2dim_offset10, (pos_data_lowdim),'y',0)
    ax1=plot_insulaEPMsingleArm(ax1, cebraposdir_dataset1_emb, (pos_data),'y',2)


    ax2=plot_insulaEPMsingleArm(ax2, cebraposdir_dataset1_emb, (pos_data),'y',3)

    ax3 =plot_insulaEPMsingleArm(ax3, cebraposdir_dataset1_emb, (pos_data),'y',2)
    ax3 =plot_insulaEPMsingleArm(ax3, cebraposdir_dataset1_emb, (pos_data),'y',3)

    ax1.set_title('idx 2 - closed arm')
    ax2.set_title('idx 3 - closed arm')
    ax3.set_title('combined CA')
    fig.suptitle('Mouse '+l_mouse_name[sess])
    plt.savefig('Plots\\PlotsNew\\PyTorch_multisess\\Combined_closedArm\\'+l_mouse_name[sess]+'_CA'+suffix+'.png')
    plt.show()

    # Grand picture
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(141, projection='3d')
    ax2 = plt.subplot(142, projection='3d')
    ax3 = plt.subplot(143, projection='3d')
    ax4 = plt.subplot(144, projection='3d')

    ax1=plot_insulaEPM(ax1, cebraposdir_dataset1_emb, (pos_data),'y')
    ax2=plot_insulaEPM(ax2,  cebratime_dataset1_emb, (pos_data),'y')
    ax3=plot_insulaEPM(ax3, cebrashufflebeh_dataset1_emb, (pos_data),'y')
    ax4=plot_insulaEPM(ax4, cebrashuffletime_dataset1_emb, (pos_data),'x')

    ax1.set_title('CEBRA-Behavior')
    ax2.set_title('CEBRA-Time')
    ax3.set_title('CEBRA-Shuffled Behavior labels')
    ax4.set_title('CEBRA-Shuffled Time labels')
    '''
    ax1.set_title('Spiking (e)')
    ax2.set_title('Spiking (isi, t)')
    ax3.set_title('Spiking (e_sh)')
    ax4.set_title('Spiking (isi, t_sh)')
    '''
    fig.suptitle('Mouse '+ l_mouse_name[sess])
    plt.savefig('Plots\\PlotsNew\\PyTorch_multisess\\CombinedPlot\\'+l_mouse_name[sess]+'_idx'+str(idx)+suffix+'.png')
    plt.show()


def plot_insulaEPM(ax, embedding, label, dir, gray = False, idx_order = (0,1,2)):
    #r_ind = label[:,1] == 1
    #l_ind = label[:,2] == 1
    if dir == 'x':
        dir = 1
    else:
        dir = 1#1
    epm0 = label[:,0] == 0
    epm1 = label[:,0] == 1
    epm2 = label[:,0] == 2
    epm3 = label[:,0] == 3
    epm4 = label[:,0] == 4
    epm5 = label[:,0] == 5
    epm6 = label[:,0] == 6
    epm7 = label[:,0] == 7
    
    if not gray:
        e0_cmap = 'Oranges'#'viridis'#'cool'
        e2_cmap = 'Blues'#'cool'#'summer' #'spring'
        e1_cmap = 'Oranges'#'viridis'#'Oranges'#'cool'#'winter'
        e3_cmap = 'Blues'#'cool' #'Blues' #'summer'#'autumn'
        e4_cmap = 'YlOrBr'#'copper'
        e6_cmap = 'YlGnBu'
        e0_c = label[epm0, dir]
        e1_c = label[epm1, dir]
        e2_c = label[epm2, dir]
        e3_c = label[epm3, dir]
        e4_c = label[epm4, dir]
        e5_c = label[epm5, dir]
        e6_c = label[epm6, dir]
        e7_c = label[epm7, dir]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'
    
    idx1, idx2, idx3 = idx_order
    e0=ax.scatter(embedding [epm0,idx1], 
               embedding [epm0,idx2], 
               embedding [epm0,idx3], 
               c=e0_c,
               cmap=e0_cmap, s=0.5)
    e1=ax.scatter(embedding [epm1,idx1], 
               embedding [epm1,idx2], 
               embedding [epm1,idx3], 
               c=e1_c,
               cmap=e1_cmap, s=0.5)
    e2=ax.scatter(embedding [epm2,idx1], 
               embedding [epm2,idx2], 
               embedding [epm2,idx3], 
               c=e2_c,
               cmap=e2_cmap, s=0.5)  
    e3=ax.scatter(embedding [epm3,idx1], 
               embedding [epm3,idx2], 
               embedding [epm3,idx3], 
               c=e3_c,
               cmap=e3_cmap, s=0.5)
    e4=ax.scatter(embedding [epm4,idx1], 
               embedding [epm4,idx2], 
               embedding [epm4,idx3], 
               c=e4_c,
               cmap=e4_cmap, s=0.5)
    e5=ax.scatter(embedding [epm5,idx1], 
               embedding [epm5,idx2], 
               embedding [epm5,idx3], 
               c=e5_c,
               cmap=e4_cmap, s=0.5) 
    e6=ax.scatter(embedding [epm6,idx1], 
               embedding [epm6,idx2], 
               embedding [epm6,idx3], 
               c=e6_c,
               cmap=e6_cmap, s=0.5) 
    e7=ax.scatter(embedding [epm7,idx1], 
               embedding [epm7,idx2], 
               embedding [epm7,idx3], 
               c=e7_c,
               cmap=e6_cmap, s=0.5) 
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
        
    return ax


def plot_insulaEPM_2D(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    #r_ind = label[:,1] == 1
    #l_ind = label[:,2] == 1
    epm0 = label[:,0] == 0
    epm1 = label[:,0] == 1
    epm2 = label[:,0] == 2
    epm3 = label[:,0] == 3
    epm4 = label[:,0] == 4
    epm5 = label[:,0] == 5
    epm6 = label[:,0] == 6
    epm7 = label[:,0] == 7
    
    if not gray:
        e0_cmap = 'Oranges'#'viridis'#'cool'
        e2_cmap = 'Blues'#'cool'#'summer' #'spring'
        e1_cmap = 'Oranges'#'viridis'#'Oranges'#'cool'#'winter'
        e3_cmap = 'Blues'#'cool' #'Blues' #'summer'#'autumn'
        e4_cmap = 'YlOrBr'#'copper'
        e6_cmap = 'YlGnBu'
        e0_c = label[epm0, dir]
        e1_c = label[epm1, dir]
        e2_c = label[epm2, dir]
        e3_c = label[epm3, dir]
        e4_c = label[epm4, dir]
        e5_c = label[epm5, dir]
        e6_c = label[epm6, dir]
        e7_c = label[epm7, dir]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'
    
    idx1, idx2, idx3 = idx_order
    e0=ax.scatter(embedding [epm0,idx1], 
               embedding [epm0,idx2], 
               c=e0_c,
               cmap=e0_cmap, s=0.5)
    e1=ax.scatter(embedding [epm1,idx1], 
               embedding [epm1,idx2], 
               c=e1_c,
               cmap=e1_cmap, s=0.5)
    e2=ax.scatter(embedding [epm2,idx1], 
               embedding [epm2,idx2], 
               c=e2_c,
               cmap=e2_cmap, s=0.5)  
    e3=ax.scatter(embedding [epm3,idx1], 
               embedding [epm3,idx2], 
               c=e3_c,
               cmap=e3_cmap, s=0.5)
    e4=ax.scatter(embedding [epm4,idx1], 
               embedding [epm4,idx2], 
               c=e4_c,
               cmap=e4_cmap, s=0.5)
    e5=ax.scatter(embedding [epm5,idx1], 
               embedding [epm5,idx2], 
               c=e5_c,
               cmap=e4_cmap, s=0.5) 
    e6=ax.scatter(embedding [epm6,idx1], 
               embedding [epm6,idx2], 
               c=e6_c,
               cmap=e6_cmap, s=0.5) 
    e7=ax.scatter(embedding [epm7,idx1], 
               embedding [epm7,idx2], 
               c=e7_c,
               cmap=e6_cmap, s=0.5) 
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
        
    return ax



def plot_insulaEPMsingleArm(ax, embedding, label, dir, arm_idx,gray = False, idx_order = (0,1,2)):
    #r_ind = label[:,1] == 1
    #l_ind = label[:,2] == 1
    epm0 = label[:,0] == arm_idx
    if dir == 'x':
        dir = 1
    else:
        dir =1# 2
    #epm1 = label[:,0] == 1
    #epm2 = label[:,0] == 2
    #epm3 = label[:,0] == 3
    #epm4 = label[:,0] == 4
    
    if not gray:
        if arm_idx ==0 or arm_idx == 1:
            e0_cmap = 'Oranges'#'cool'
        elif arm_idx == 2 or arm_idx == 3:
            e0_cmap ='Blues'#'summer' #'spring'
        elif arm_idx == 4 or arm_idx == 5:
            e0_cmap = 'YlOrBr'#'copper'
        else:
            e0_cmap = 'YlGnBu'

        #e1_cmap = 'spring'
        #e2_cmap = 'winter'
        #e3_cmap = 'autumn'
        #e4_cmap = 'copper'
        e0_c = label[epm0, dir]
        #e1_c = label[epm1, 0]
        #e2_c = label[epm2, 0]
        #e3_c = label[epm3, 0]
        #e4_c = label[epm4, 0]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'
    
    idx1, idx2, idx3 = idx_order
    e0=ax.scatter(embedding [epm0,idx1], 
               embedding [epm0,idx2], 
               embedding [epm0,idx3], 
               c=e0_c,
               cmap=e0_cmap, s=0.5)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
        
    return ax

