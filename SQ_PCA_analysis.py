from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
import random

def orderHM(hm, st, ed):
    avgs = np.nanmean(hm[:,st:ed], axis=-1)
    idx = np.argsort(avgs)[::-1] # reverse array after sorting it
    hm1 = hm[idx,:]
    return hm1

def SQ_plotavgSeparatedUnits(eohm_total_S, iohm_total_S, rhm_total_S, eohm_total_Q, iohm_total_Q, rhm_total_Q, timevec,
                             counts_SQ):
    eohm = []
    iohm = []
    # eihm = []
    rhm = []
    # print(timevec)
    # print(eohm_total_S)
    eohm_total_S1 = eohm_total_S.copy()
    iohm_total_S1 = iohm_total_S.copy()
    # eihm_total_S1 = eihm_total_S.copy()
    rhm_total_S1 = rhm_total_S.copy()
    eohm_total_Q1 = eohm_total_Q.copy()
    iohm_total_Q1 = iohm_total_Q.copy()
    # eihm_total_Q1 = eihm_total_Q.copy()
    rhm_total_Q1 = rhm_total_Q.copy()
    # timevec = np.linspace(-9.5, 9.5, 401)[:-1] #np.linspace(-5, 4.5, 20) #
    # timevec=timevec[:-1]
    timepoints = timevec.size
    time_min = np.min(timevec)
    time_max = np.max(timevec)
    binwidth = 0.025  # set
    timevec = np.linspace(time_min + 0.025, time_max - 0.025, timepoints - 1)

    '''
    for i in range(len(eohm_total_S)):
        # z score everything
        #print(len(eohm_total_S))
        #print(eohm_total_S[i].shape)
        if eohm_total_S[i].shape[-1] != 0:
            #if eohm_total_S[i]
            #pprint.pprint(eohm_total_S[i])
            eohm_total_S[i],_,_ = calcAvgZscore(eohm_total_S[i], timevec, baselinetime_s=[-5, -3])
            #pprint.pprint((eohm_total_S[i]))

        else:
            eohm_total_S[i] = np.array([])
        if iohm_total_S[i].shape[-1] != 0:
            iohm_total_S[i],_,_ = calcAvgZscore(iohm_total_S[i], timevec, baselinetime_s=[-5, -3])
        else:
            iohm_total_S[i] = np.array([])

        #if eihm_total_S[i].shape[-1] != 0:
        #    eihm_total_S[i],_,_ = calcAvgZscore(eihm_total_S[i], timevec, baselinetime_s=[-5, -3])
        #else:
        #    eihm_total_S[i] = np.array([])

        if rhm_total_S[i].shape[-1] != 0:
            rhm_total_S[i],_,_ = calcAvgZscore(rhm_total_S[i], timevec, baselinetime_s=[-5, -3])
        else:
            rhm_total_S[i] = np.array([])
        if eohm_total_Q[i].shape[-1] != 0:
            eohm_total_Q[i],_,_ = calcAvgZscore(eohm_total_Q[i], timevec, baselinetime_s=[-5, -3])
        else:
            eohm_total_Q[i] = np.array([])
        if iohm_total_Q[i].shape[-1] != 0:
            iohm_total_Q[i],_,_ = calcAvgZscore(iohm_total_Q[i], timevec, baselinetime_s=[-5, -3])
        else:
            iohm_total_Q[i] = np.array([])
        #if eihm_total_Q[i].shape[-1] != 0:

        #    eihm_total_Q[i],_,_ = calcAvgZscore(eihm_total_Q[i], timevec, baselinetime_s=[-5, -3])
        #else:
        #    eihm_total_Q[i] = np.array([])
        if rhm_total_Q[i].shape[-1] != 0:

            rhm_total_Q[i],_,_ = calcAvgZscore(rhm_total_Q[i], timevec, baselinetime_s=[-5, -3])
        else:
            rhm_total_Q[i] = np.array([])
    #print(eohm_total_S)
    '''
    for i in range(len(eohm_total_S)):
        # print(eohm_total_S[i])
        if eohm_total_S[i].shape[-1] != 0:
            # to ignore infinities
            eohm.append(np.nanmean(eohm_total_S[i], axis=1))
        if iohm_total_S[i].shape[-1] != 0:
            iohm.append(np.nanmean(iohm_total_S[i], axis=1))
        # if eihm_total_S[i].shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_S[i], axis=1))
        if rhm_total_S[i].shape[-1] != 0:
            rhm.append(np.nanmean(rhm_total_S[i], axis=1))

    eohm = np.array(eohm)
    iohm = np.array(iohm)
    # eihm = np.array(eihm)
    rhm = np.array(rhm)  # session x time
    # print(rhm)
    # print(eohm)

    eohm_totavg = np.nanmean(eohm, axis=0)
    # print(eohm_totavg)
    eohm_totsd = sem(eohm, axis=0, nan_policy='omit')
    iohm_totavg = np.nanmean(iohm, axis=0)
    iohm_totsd = sem(iohm, axis=0, nan_policy='omit')
    # eihm_totavg = np.nanmean(eihm, axis=0)
    # eihm_totsd =  sem(eihm, axis=0,nan_policy='omit')
    rhm_totavg = np.nanmean(rhm, axis=0)
    rhm_totsd = sem(rhm, axis=0, nan_policy='omit')

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].plot(timevec, gaussian_filter1d(eohm_totavg, sigma=3), c='k')
    ax[0, 0].plot(timevec, gaussian_filter1d(eohm_totavg + eohm_totsd, sigma=3), c='grey')
    ax[0, 0].plot(timevec, gaussian_filter1d(eohm_totavg - eohm_totsd, sigma=3), c='grey')
    # print(iohm_totavg)
    # print(iohm_totavg - iohm_totsd)
    ax[0, 1].plot(timevec, gaussian_filter1d(iohm_totavg, sigma=3), c='k')
    ax[0, 1].plot(timevec, gaussian_filter1d(iohm_totavg + iohm_totsd, sigma=3), c='grey')
    ax[0, 1].plot(timevec, gaussian_filter1d(iohm_totavg - iohm_totsd, sigma=3), c='grey')
    # ax[0, 2].plot(timevec,gaussian_filter1d(eihm_totavg, sigma = 3), c='k')
    # ax[0, 2].plot(timevec,gaussian_filter1d(eihm_totavg + eihm_totsd, sigma = 3), c='grey')
    # ax[0, 2].plot(timevec,gaussian_filter1d(eihm_totavg - eihm_totsd, sigma = 3), c='grey')
    ax[0, 2].plot(timevec, gaussian_filter1d(rhm_totavg, sigma=3), c='k')
    ax[0, 2].plot(timevec, gaussian_filter1d(rhm_totavg + rhm_totsd, sigma=3), c='grey')
    ax[0, 2].plot(timevec, gaussian_filter1d(rhm_totavg - rhm_totsd, sigma=3), c='grey')

    eohm = []
    iohm = []
    # eihm = []
    rhm = []
    for i in range(len(eohm_total_Q)):
        # print(eohm_total_S[i])
        if (eohm_total_Q[i]).shape[-1] != 0:
            eohm.append(np.nanmean(eohm_total_Q[i], axis=1))
        if (iohm_total_Q[i]).shape[-1] != 0:
            # print(iohm_total_Q[i])
            iohm.append(np.nanmean(iohm_total_Q[i], axis=1))
        # if (eihm_total_S[i]).shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_Q[i], axis=1))
        if (rhm_total_Q[i]).shape[-1] != 0:
            rhm.append(np.nanmean(rhm_total_Q[i], axis=1))

    eohm = np.array(eohm)  # .T
    iohm = np.array(iohm)  # .T
    # eihm = np.array(eihm)#.T
    rhm = np.array(rhm)  # .T  # session x time

    # print(eohm)
    eohm_totavg = np.nanmean(eohm, axis=0)
    eohm_totsd = sem(eohm, axis=0, nan_policy='omit')
    iohm_totavg = np.nanmean(iohm, axis=0)
    iohm_totsd = sem(iohm, axis=0, nan_policy='omit')
    # eihm_totavg = np.nanmean(eihm, axis=0)
    # eihm_totsd =  sem(eihm, axis=0,nan_policy='omit')
    rhm_totavg = np.nanmean(rhm, axis=0)
    rhm_totsd = sem(rhm, axis=0, nan_policy='omit')

    ax[1, 0].plot(timevec, gaussian_filter1d(eohm_totavg, sigma=3), c='k')
    ax[1, 0].plot(timevec, gaussian_filter1d(eohm_totavg + eohm_totsd, sigma=3), c='grey')
    ax[1, 0].plot(timevec, gaussian_filter1d(eohm_totavg - eohm_totsd, sigma=3), c='grey')
    ax[1, 1].plot(timevec, gaussian_filter1d(iohm_totavg, sigma=3), c='k')
    ax[1, 1].plot(timevec, gaussian_filter1d(iohm_totavg + iohm_totsd, sigma=3), c='grey')
    ax[1, 1].plot(timevec, gaussian_filter1d(iohm_totavg - iohm_totsd, sigma=3), c='grey')
    # ax[1, 2].plot(timevec, gaussian_filter1d(eihm_totavg, sigma=3), c='k')
    # ax[1, 2].plot(timevec, gaussian_filter1d(eihm_totavg + eihm_totsd, sigma=3), c='grey')
    # ax[1, 2].plot(timevec, gaussian_filter1d(eihm_totavg - eihm_totsd, sigma=3), c='grey')
    ax[1, 2].plot(timevec, gaussian_filter1d(rhm_totavg, sigma=3), c='k')
    ax[1, 2].plot(timevec, gaussian_filter1d(rhm_totavg + rhm_totsd, sigma=3), c='grey')
    ax[1, 2].plot(timevec, gaussian_filter1d(rhm_totavg - rhm_totsd, sigma=3), c='grey')

    ax[0, 0].axvline(x=0, color='b')
    ax[0, 1].axvline(x=0, color='b')
    ax[0, 2].axvline(x=0, color='b')
    # ax[0, 3].axvline(x=0, color='b')
    ax[1, 0].axvline(x=0, color='b')
    ax[1, 1].axvline(x=0, color='b')
    ax[1, 2].axvline(x=0, color='b')
    # ax[1, 3].axvline(x=0, color='b')

    ax[0, 0].set_ylim([-4.5, 5])
    ax[0, 1].set_ylim([-4.5, 5])
    ax[0, 2].set_ylim([-4.5, 5])
    ax[1, 0].set_ylim([-4.5, 5])
    ax[1, 1].set_ylim([-4.5, 5])
    ax[1, 2].set_ylim([-4.5, 5])

    ax[0, 0].set_ylabel('Z scores - Sucrose')
    ax[1, 0].set_ylabel('Z scores - Quinine')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 2].set_xlabel('Time (s)')
    # ax[1, 3].set_xlabel('Time (s)')
    ax[0, 0].set_title('Excited')
    ax[0, 1].set_title('Inhibited')
    # ax[0, 2].set_title('Excited and Inhibited')
    ax[0, 2].set_title('No Change')

    # plt.savefig('SucroseQuininePSTH.svg')
    plt.close()

    # HEATMAPS
    eohm = []
    iohm = []
    # eihm = []
    rhm = []
    S_totalunits = np.zeros(3)

    for i in range(len(eohm_total_S)):
        # print(eohm_total_S[i])
        if eohm_total_S[i].shape[-1] != 0:
            # to ignore infinities
            eohm.append(np.nanmean(eohm_total_S[i], axis=1))
        if iohm_total_S[i].shape[-1] != 0:
            iohm.append(np.nanmean(iohm_total_S[i], axis=1))
        # if eihm_total_S[i].shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_S[i], axis=1))
        if rhm_total_S[i].shape[-1] != 0:
            rhm.append(np.nanmean(rhm_total_S[i], axis=1))

    eohm = np.array(eohm)
    iohm = np.array(iohm)
    # eihm = np.array(eihm)
    rhm = np.array(rhm)  # session x time

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    pcm1 = ax[0, 0].pcolormesh(eohm, cmap='cividis')  # (time, unit)
    fig.colorbar(pcm1, ax=ax[0, 0])
    pcm2 = ax[0, 1].pcolormesh(iohm, cmap='cividis')
    fig.colorbar(pcm2, ax=ax[0, 1])
    # pcm3 = ax[0, 2].pcolormesh(eihm, cmap='cividis')
    # fig.colorbar(pcm3, ax=ax[0, 2])
    pcm4 = ax[0, 2].pcolormesh(rhm, cmap='cividis')
    fig.colorbar(pcm4, ax=ax[0, 2])

    eohm = []
    iohm = []
    # eihm = []
    rhm = []
    Q_totalunits = np.zeros(3)

    for i in range(len(eohm_total_Q)):
        # print(eohm_total_S[i])
        if eohm_total_Q[i].shape[-1] != 0:
            # to ignore infinities
            eohm.append(np.nanmean(eohm_total_Q[i], axis=1))
        if iohm_total_Q[i].shape[-1] != 0:
            iohm.append(np.nanmean(iohm_total_Q[i], axis=1))
        # if eihm_total_S[i].shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_S[i], axis=1))
        if rhm_total_Q[i].shape[-1] != 0:
            rhm.append(np.nanmean(rhm_total_Q[i], axis=1))

    eohm = np.array(eohm)
    iohm = np.array(iohm)
    # eihm = np.array(eihm)
    rhm = np.array(rhm)  # session x time

    eohm = np.array(eohm)
    iohm = np.array(iohm)
    # eihm = np.array(eihm)
    rhm = np.array(rhm)
    # print(iohm)

    pcm5 = ax[1, 0].pcolormesh(eohm, cmap='cividis')  # (time, unit)
    fig.colorbar(pcm5, ax=ax[1, 0])
    pcm6 = ax[1, 1].pcolormesh(iohm, cmap='cividis')
    fig.colorbar(pcm6, ax=ax[1, 1])
    # pcm7 = ax[1, 2].pcolormesh(eihm, cmap='cividis')
    # fig.colorbar(pcm7, ax=ax[1, 2])
    pcm8 = ax[1, 2].pcolormesh(rhm, cmap='cividis')
    fig.colorbar(pcm8, ax=ax[1, 2])
    ax[0, 0].set_ylabel('Sessions - Sucrose')
    ax[1, 0].set_ylabel('Sessions - Quinine')
    ax[0, 0].set_title('Excited')
    ax[0, 1].set_title('Inhibited')
    # ax[0, 2].set_title('Excited and Inhibited')
    ax[0, 2].set_title('No Change')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 2].set_xlabel('Time (s)')
    # ax[1, 3].set_xlabel('Time (s)')
    plt.savefig('SucroseQuinine_Heatmap_20240707.svg')
    # plt.show()
    plt.close()

    '''
    ## plot pie graph
    labels = ['excited', 'inhibited', 'no change']
    fig, ax = plt.subplots(1,2)
    ax.flatten()
    #print(S_totalunits)
    ax[0].pie(S_totalunits, labels=labels, autopct='%1.1f%%')
    ax[1].pie(Q_totalunits, labels=labels, autopct='%1.1f%%')
    ax[0].set_title("Sucrose")
    ax[1].set_title("Quinine")
    plt.savefig("SQ_unitpercentagecount.png")
    #plt.show()
    plt.close()
    '''
    # MATRIX 3X3 - Counts
    counterrr = counts_SQ.flatten()  # concatenate along rows
    counterrr = counterrr.astype(int)
    # print(counterrr)

    labels = ['ExS', 'IbS', 'ExQ', 'InhQ', 'Exc-both', 'Inh-both', 'ExS/IbQ', 'ExQ/IbS', 'no response']
    fig = plt.figure(figsize=(15, 15), dpi=80)
    plt.pie(counterrr, autopct='%1.1f%%')
    plt.legend(labels)
    # plt.savefig("SQ_unitpercentagecount_combined.svg")
    plt.close()

    # HEATMAPS - combined
    eohm = np.array([])
    iohm = np.array([])
    rhm = np.array([])
    timevecc = np.arange(-5 + 0.025, 5 - 0.025, 1)  # np.arange(-10,9.95,1) #np.linspace(-10, 9.5, 11)
    timepointnum = timevecc.size
    st5sec = int(np.where(timevec <= -5)[0][-1])
    ed5sec = int(np.where(timevec >= 5)[0][0])
    '''
    for i in range(len(eohm_total_S)):
        # print(eohm_total_S[i])
        if (eohm_total_S[i]).shape[-1] != 0 and np.all(np.all(eohm_total_S[i] == eohm_total_S[i])):
            if len(eohm) == 0:
                eohm = (eohm_total_S[i][st5sec:ed5sec].T)
            else:
                eohm = np.vstack((eohm, eohm_total_S[i][st5sec:ed5sec].T))  # eohm.append(eohm_total_S[i].T)

        if (iohm_total_S[i]).shape[-1] != 0 and np.all(np.all(iohm_total_S[i] == iohm_total_S[i])):

            if len(iohm) == 0:
                iohm = iohm_total_S[i][st5sec:ed5sec].T
            else:
                iohm = np.vstack((iohm, iohm_total_S[i][st5sec:ed5sec].T))

        if (rhm_total_S[i]).shape[-1] != 0 and not np.isnan(rhm_total_S[i]).any():
            if len(rhm) == 0:
                rhm = rhm_total_S[i][st5sec:ed5sec].T
            else:
                rhm = np.vstack((rhm, rhm_total_S[i][st5sec:ed5sec].T))

    eohm = np.array(eohm)
    iohm = np.array(iohm)
    rhm = np.array(rhm)
    #print(eohm.shape)
    '''
    eohm = np.array([])
    iohm = np.array([])
    # eihm = []
    rhm = np.array([])
    # print(rhm_total_S[0].shape)

    for i in range(len(eohm_total_S)):
        # print(eohm_total_S[i])
        if (eohm_total_S[i]).shape[-1] != 0:
            if eohm.size == 0:
                eohm = eohm_total_S[i][st5sec:ed5sec, :].T
            else:
                eohm = np.vstack((eohm, eohm_total_S[i][st5sec:ed5sec, :].T))
        if (iohm_total_S[i]).shape[-1] != 0:
            if iohm.size == 0:
                iohm = iohm_total_S[i][st5sec:ed5sec, :].T
            else:
                iohm = np.vstack((iohm, iohm_total_S[i][st5sec:ed5sec, :].T))
        # if (eihm_total_S[i]).shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_Q[i], axis=1))
        if (rhm_total_S[i]).shape[-1] != 0:
            if rhm.size == 0:
                rhm = rhm_total_S[i][st5sec:ed5sec, :].T
            else:
                rhm = np.vstack((rhm, rhm_total_S[i][st5sec:ed5sec, :].T))

    eohm = np.array(eohm)  # .T
    iohm = np.array(iohm)  # .T
    # eihm = np.array(eihm)#.T
    rhm = np.array(rhm)  # .T  # session x time
    combinedS = np.vstack((eohm, iohm, rhm))

    #print(timevec[st5sec:ed5sec])
    st_max = np.where(timevec[st5sec:ed5sec] >= 0)[0][0].astype(int)
    ed_max = np.where(timevec[st5sec:ed5sec] >= 1)[0][0].astype(int)
    #print(eohm.shape)
    eohm = orderHM(eohm, st_max, ed_max)
    iohm = orderHM(iohm, st_max, ed_max)
    rhm = orderHM(rhm, st_max, ed_max)

    combined = np.vstack((eohm, iohm, rhm))
    #combinedS = combined.copy()
    line_ei = eohm.shape[0]
    line_ir = iohm.shape[0] + eohm.shape[0]
    #print(line_ir)
    #print(line_ei)
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax.flatten()
    vmin = -2  # -5
    vmax = 3  # 10
    h0 = ax[0].imshow(combined, vmin=vmin, vmax=vmax)  # ax[0].pcolormesh(combined, cmap='cividis')  # (time, unit)
    ax[0].axhline(y=line_ei, color = 'k')
    ax[0].axhline(y=line_ir, color='k')
    #ax[0].colorbar(h0)
    fig.colorbar(h0, ax=ax[0],shrink=0.5)

    '''
    eohm = []
    iohm = []
    rhm = []
    for i in range(len(eohm_total_S)):
        if (eohm_total_Q[i]).shape[-1] != 0 and not np.isnan(eohm_total_Q[i]).any():
            if len(eohm) == 0:
                eohm = eohm_total_S[i][st5sec:ed5sec].T
            else:
                eohm = np.vstack((eohm, eohm_total_Q[i][st5sec:ed5sec].T))

        if (iohm_total_Q[i]).shape[-1] != 0 and not np.isnan(iohm_total_Q[i]).any():
            if len(iohm) == 0:
                iohm = iohm_total_S[i][st5sec:ed5sec].T
            else:
                iohm = np.vstack((iohm, iohm_total_Q[i][st5sec:ed5sec].T))

        if (rhm_total_Q[i]).shape[-1] != 0 and not np.isnan(rhm_total_Q[i]).any():
            if len(rhm) == 0:
                rhm = rhm_total_Q[i][st5sec:ed5sec].T
            else:
                rhm = np.vstack((rhm, rhm_total_Q[i][st5sec:ed5sec].T))
    '''
    eohm = np.array([])
    iohm = np.array([])
    # eihm = []
    rhm = np.array([])
    # print(rhm_total_Q[0].shape)

    for i in range(len(eohm_total_Q)):
        # print(eohm_total_S[i])
        if (eohm_total_Q[i]).shape[-1] != 0:
            if eohm.size == 0:
                eohm = eohm_total_Q[i][st5sec:ed5sec, :].T
            else:
                eohm = np.vstack((eohm, eohm_total_Q[i][st5sec:ed5sec, :].T))
        if (iohm_total_Q[i]).shape[-1] != 0:
            if iohm.size == 0:
                iohm = iohm_total_Q[i][st5sec:ed5sec, :].T
            else:
                iohm = np.vstack((iohm, iohm_total_Q[i][st5sec:ed5sec, :].T))
        # if (eihm_total_S[i]).shape[-1] != 0:
        #    eihm.append(np.nanmean(eihm_total_Q[i], axis=1))
        if (rhm_total_Q[i]).shape[-1] != 0:
            if rhm.size == 0:
                rhm = rhm_total_Q[i][st5sec:ed5sec, :].T
            else:
                rhm = np.vstack((rhm, rhm_total_Q[i][st5sec:ed5sec, :].T))

    eohm = np.array(eohm)  # .T
    iohm = np.array(iohm)  # .T
    # eihm = np.array(eihm)#.T
    rhm = np.array(rhm)  # .T  # session x time
    combinedQ = np.vstack((eohm, iohm, rhm))
    eohm = orderHM(eohm, st_max, ed_max)
    iohm = orderHM(iohm, st_max, ed_max)
    rhm = orderHM(rhm, st_max, ed_max)

    combined = np.vstack((eohm, iohm, rhm))
    line_ei = eohm.shape[0]
    line_ir = iohm.shape[0] + eohm.shape[0]

    # print(combined.shape)
    # print(iohm)
    h1 = ax[1].imshow(combined, vmin=vmin, vmax=vmax)  # ax[0].pcolormesh(combined, cmap='cividis')  # (time, unit)
    ax[1].axhline(y=line_ei, color='k')
    ax[1].axhline(y=line_ir, color='k')
    fig.colorbar(h1, ax=ax[1],shrink=0.5)
    combinedQ = combined.copy()
    idx = np.zeros(timepointnum)
    for i in range(timepointnum):
        idx[i] = (np.where(timevec[st5sec:ed5sec] <= timevecc[i])[0][-1])

    timevecc = np.floor(timevecc).astype(int)
    ax[0].set_xticks(idx.astype(int))
    ax[1].set_xticks(idx.astype(int))
    ax[1].set_xticklabels((timevecc))
    ax[0].set_xticklabels((timevecc))
    # print(timevec.shape)
    ax[0].set_title('Sucrose')
    ax[1].set_title('Quinine')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_xlabel('Time (s)')
    plt.savefig('SucroseQuinine_Heatmap_combined_20240707.svg')
    plt.show()
    plt.close()

    return combinedS, combinedQ

def calcAvgZscore(avg_neur, timevec, baselinetime_s= [-5, -3], sendnans=False):
    # first average out the trials
    # then check baseline for each unit
    avg_neur = avg_neur.T
    timevec = np.round(timevec, decimals=2)
    st = int(np.where(timevec <= baselinetime_s[0])[0][-1])
    ed = int(np.where(timevec >= baselinetime_s[1])[0][0]) + 1
    baseline = avg_neur[st:ed, :]  # baseline firing rate across all units
    mean_ = np.nanmean(baseline, axis=0)  # across time
    std_ = np.nanstd(baseline, axis=0)  # across time
    check0 = np.where(std_ == 0)[0]
    # print(check0)
    zscore_ = (avg_neur - mean_)  # (time, units)
    heat_map = (zscore_ / std_).T # (units, time)
    '''
    for i in range(len(check0)):
        #print(~np.isfinite(heat_map[i, :]))
        #heat_map[i, ~np.isfinite(heat_map[i, :])] = np.nan
        heat_map[check0[i], :] = np.nan
    '''
    if sendnans:
        return heat_map, check0
    
    return heat_map



def SQ_extract(datadir,  mouse_consider_sq, timevec, type):
    # datadir = Path(r'S:\_Tanmai\behavior_and_ephys_analysis\PSTHdata\Heatmaps_SQ_AllUnits_20240707') # Path(r"C:\Users\tdhanireddy\PycharmProjects\beyeler-yoni\scripts\behavior_and_ephys_analysis\PSTHdata\Heatmaps_SQ_AllUnits") # take data from here
    data_list = datadir.rglob('*.npz')
    # make a matrix now - trial-averaged and arranged by idx_S and idx_Q
    # mouse_consider_sq = ["F23", "M30", "M31", "F2492", "F2493", "F2495", "F2496", "M2497", "M2498", "M2499", "M2502", "M3303", "F3309"]
    neural_S_all = []
    neural_Q_all = []
    idx_S_all = []
    idx_Q_all = []
    heatmap_S = np.array([])
    heatmap_Q = np.array([])
    heatmap_persess = [None]*1
    st5sec = int(np.where(timevec <= -5)[0][-1])
    ed5sec = int(np.where(timevec >= 5)[0][0])
    mousenames = []
    trialnumsS = []
    trialnumsQ = []
    units = []

    for data_path in data_list:
        n = data_path.name
        # print(str(n)[:-13])
        if type == "saline":
            if str(n)[:-20] in mouse_consider_sq:
                print(n)
            else:
                continue
        else:
            if str(n)[:-16] in mouse_consider_sq:
                print(n)
            else:
                continue
        mousenames.append(str(n)[:-4])
        f= np.load(str(datadir)+"\\"+n, allow_pickle=True)
        neural_S_all.append(f['arr_0'])
        neural_Q_all.append(f['arr_1'])
        # print(neural_S_all.shape)
        trialnumsS.append(len(f['arr_0']))
        trialnumsQ.append(len(f['arr_1']))
        idx_S_all.append(f['arr_2'])
        idx_Q_all.append(f['arr_3'])
        Sarr = np.nanmean(f['arr_0'], axis=0).T
        Qarr = np.nanmean(f['arr_1'], axis=0).T

        window = 10
        for i in range(Sarr.shape[1]):
            Sarr[:,i] = np.sum(Sarr[:,i:i+window], axis = 1)
            Qarr[:,i] = np.sum(Qarr[:,i:i+window], axis=1)

        Sarr = calcAvgZscore(Sarr)
        Qarr = calcAvgZscore(Qarr)
        units.append(Sarr.shape[0])
        
        if len(heatmap_S) == 0:
            heatmap_S = Sarr[:,st5sec:ed5sec]
            heatmap_Q = Qarr[:,st5sec:ed5sec]
            heatmap_persess[0] = np.hstack((Sarr[:,st5sec:ed5sec],Qarr[:,st5sec:ed5sec]))
        else:
            heatmap_S = np.vstack((heatmap_S,Sarr[:,st5sec:ed5sec]))
            heatmap_Q = np.vstack((heatmap_Q,Qarr[:,st5sec:ed5sec]))
            heatmap_persess.append(np.hstack((Sarr[:,st5sec:ed5sec],Qarr[:,st5sec:ed5sec])))

    return heatmap_persess, heatmap_S, heatmap_Q, trialnumsS, trialnumsQ, mousenames, neural_S_all, neural_Q_all, units, mousenames





# time_vector = params['ephys']['t']
    # if swap == True:
    #     ch_str_0 = 'ADC3' #s
    #     ch_str_1 = 'ADC2' 
    # else: 
    #     ch_str_1 = 'ADC3'
    #     ch_str_0 = 'ADC2' #s
        
    # # sucrose
    # lick_idx = params['ephys'][f'{ch_str_0}_pulses']['onsets']
    # lick_t = time_vector[lick_idx]
    # lick_t_og_s = np.copy(lick_t)
    # lick_dt = np.diff(lick_t)
    # idx = np.where(lick_dt <= 2)
    # idx = np.array(idx) + 1
    # lick_idx = np.delete(lick_idx, idx)
    

    # quinine
    # lick_idx = params['ephys'][f'{ch_str_1}_pulses']['onsets']
    # lick_t = time_vector[lick_idx]
    # lick_t_og_q = np.copy(lick_t)
    # lick_dt = np.diff(lick_t)
    # idx = np.where(lick_dt <= 2)
    # idx = np.array(idx) + 1
    # lick_idx = np.delete(lick_idx, idx)


def rasterPlot(params,swap, mousename):
    if swap:
        rasterQ = params['beh_analysis']['raster']['lick_ch0']
        rasterS = params['beh_analysis']['raster']['lick_ch1']
    else:
        rasterS = params['beh_analysis']['raster']['lick_ch0']
        rasterQ = params['beh_analysis']['raster']['lick_ch1']
        
    unitnum = len(params['beh_analysis']['raster']['lick_ch1'])
    for i in range(unitnum):
        for j in range(len(rasterS[i])):
            plt.scatter(rasterS[i][j], j*np.ones(len(rasterS[i][j])), s=1, marker='|', c='blue')
        for j in range(len(rasterQ[i])):
            plt.scatter(rasterQ[i][j], (j+1+len(rasterS[0]))*np.ones(len(rasterQ[i][j])), s=1, marker='|', c='orange')

        plt.ylabel('# Unit')
        plt.xlabel('Time (s)')
        plt.title('Unit '+str(i))
        plt.savefig('PCA_finalpictures_SQNew/Rasters/mouse'+mousename+'_Unit'+str(i)+'.svg')
        # plt.show()


def rasterPlot_plt(params, swap, mousename):
        
    # if swap:
    #     lick_t_og_s = params['behavior']['lick_ch0']['ts']
    #     lick_t_og_q = params['behavior']['lick_ch1']['ts']

    # else:
    #     lick_t_og_q = params['behavior']['lick_ch0']['ts']
    #     lick_t_og_s = params['behavior']['lick_ch1']['ts']

    time_vector = params['ephys']['t']
    if swap == True:
        ch_str_0 = 'ADC3' #s
        ch_str_1 = 'ADC2' 
    else: 
        ch_str_1 = 'ADC3'
        ch_str_0 = 'ADC2' #s
        
    # sucrose
    lick_idx = params['ephys'][f'{ch_str_0}_pulses']['onsets']
    # print("i feel it is the number of licks of all the sucrose",lick_idx)
    print(len(lick_idx), "this gives the lenth of the sucrose licks =")
    lick_t = time_vector[lick_idx]
    # print(lick_t, "maybe the lick time data")
    lick_t_og_s = np.copy(lick_t)
    lick_dt = np.diff(lick_t)
    # print(lick_dt, "it is the difference between 2 licks, if the difference is less than 5 ten we discard it, this means it is lick bouts")
    idx = np.where(lick_dt <= 5) # <= 2
    idx_s = np.array(idx) + 1
    lick_idx_s = np.delete(lick_idx, idx_s)
    # print(lick_idx_s, "final number of the lick bout idx")

    # quinine
    lick_idx = params['ephys'][f'{ch_str_1}_pulses']['onsets']
    # print("i feel it is the number of licks of all the quinine",lick_idx)
    print(len(lick_idx), "this gives the lenth of the sucrose quinine =")
    lick_t = time_vector[lick_idx]
    lick_t_og_q = np.copy(lick_t)
    lick_dt = np.diff(lick_t)
    idx = np.where(lick_dt <= 5) # <= 2
    idx_q = np.array(idx) + 1
    lick_idx_q = np.delete(lick_idx, idx_q)
    


    num_licks_s = 0
    num_licks_q = 0

    # print("lick idx s",lick_idx_s)
    # print("lick idx q",lick_idx_q)

    for i in range(len(lick_idx_s)):
        print(i)
        s = lick_idx_s[i]/30000
        # print("s", s)
        licks_bout_idx = np.where(np.abs(lick_t_og_s-s) <=5)[0] # <= 2
        # print("lick bout idx", licks_bout_idx)
        lick_bouts = lick_t_og_s[licks_bout_idx] - lick_t_og_s[licks_bout_idx[0]]#- np.average(lick_t_og_s[licks_bout_idx])
        # print("lick bouts", lick_bouts)
        num_licks_s += len(lick_bouts)
        
        # plt.scatter(lick_bouts, i*np.ones(len(licks_bout_idx)), marker="|", s=1, color='blue')

    for i in range(len(lick_idx_q)):
        q = lick_idx_q[i]/30000
        licks_bout_idx = np.where(np.abs(lick_t_og_q-q) <=5)[0] # <= 2
        lick_bouts = lick_t_og_q[licks_bout_idx]- lick_t_og_q[licks_bout_idx[0]] #- np.average(lick_t_og_q[licks_bout_idx])
        num_licks_q += len(lick_bouts)
        # plt.scatter(lick_bouts, (i+len(lick_idx_s))*np.ones(len(licks_bout_idx)), marker="|", s=1, color='orange')
    
    plt.ylabel('# Lick Bout')
    plt.xlabel('Time (s)')
    plt.title(mousename)
    # plt.savefig('PCA_finalpictures_SQNew/Rasters_LickBouts/mouse'+mousename+'_Unit'+str(i)+'.svg')
    # plt.show()
    

    return num_licks_s, num_licks_q


def projectdata(PC, data_mat):
    projected_data = np.matmul((np.transpose(data_mat)), PC)
    return projected_data

def PCA_analysis(l_data_PCA, k = 100):
    # k = 100  # nb of dimensions
    l_projected_data = []
    var = []
    #data_mat = l_data_PCA
    data_mat = l_data_PCA.copy()
    # for i in range(l_data_PCA.shape[0]):
    #     data_mat[i,:] = (l_data_PCA[i,:] - np.mean(l_data_PCA[i,:]))/(np.std(l_data_PCA[i,:])) #StandardScaler().fit_transform(l_data_PCA) # l_data_PCA
    
    cov_mat = np.cov(data_mat)
    print(data_mat.shape)
    #print(cov_mat)
    eigval, eigvec = np.linalg.eig(cov_mat)
    print(cov_mat.shape)
    eigval_copy = np.copy(eigval)
    eigvec_copy = np.copy(eigvec)
    idx = eigval.argsort()[::-1] # ??
    eigvec_sorted = eigvec[:, idx]
    # print(eigenvec[1].shape)
    eigval_sorted = eigval[idx]
    PC = eigvec_sorted[:, :k]
    print(PC.shape)
    projected_data = projectdata(PC, data_mat)
    eigval_sorted_normalised = eigval_sorted/np.sum(eigval_sorted)
    l_projected_data.append(projected_data)
    var.append(eigval_sorted_normalised)
    print("Dimension of projected data:")
    print(projected_data.shape)
    return projected_data, eigval_sorted_normalised, PC, eigvec_copy, eigval_copy


