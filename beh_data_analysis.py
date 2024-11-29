import numpy as np
from numpy import load
import pandas as pd
from pathlib import Path
import cv2
from PCA_traj_edit import get_coordinates_EPM, get_video_frames_onsets, check_csv_separator
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pprint
import pickle as pkl
from scipy.io import loadmat


def get_coordinates_EPM_d1(l_mouse_name, l_date, l_rotation):
    '''
    path: userlandmarks.npz have coordinates of x and y coordinates.

    '''
    l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center = [], [], [], [], [], [] #Create empty lists for the 3 regions of interests in X and Y coordinates
    for mouse_idx in range(len(l_mouse_name)): #mouse_idx = mouse position in the EPM
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        # path = "C:\\beyeler-yoni\\data\\analysis\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_userlandmarks.npz" #this .npz file is given by the ephys script and gives the coordinates of each zones of the EPM
        #path = "Y:\\Ephys_in_vivo\\02_ANALYSIS\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_userlandmarks.npz" #this .npz file is given by the ephys script and gives the coordinates of each zones of the EPM
        path = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\\npz_files\\" + mouse_name + "_EPM_userlandmarks.npz"
        data = load(path)
        lst = data.files #data.files = location of each zones in the EPM
        f_rotation = l_rotation[mouse_idx]
        #Initiliaze the coordinates for each zones. There are 4 coordinates in X and Y for each zones (because we delineate the zone in 4 points):
        x1_open, x2_open, x3_open, x4_open = 0.0, 0.0, 0.0, 0.0
        y1_open, y2_open, y3_open, y4_open = 0.0, 0.0, 0.0, 0.0
        x1_closed, x2_closed, x3_closed, x4_closed = 0.0, 0.0, 0.0, 0.0
        y1_closed, y2_closed, y3_closed, y4_closed = 0.0, 0.0, 0.0, 0.0
        x1_center, x2_center, y1_center, y2_center = 0.0, 0.0, 0.0, 0.0

        for item in lst:
            if item == 'Open Arm 1':
                if f_rotation == 0:
                    x1_open = np.min(data[item][0, :])
                    x2_open = np.max(data[item][0, :])
                    y1_open = np.min(data[item][1, :])
                    y2_open = np.max(data[item][1, :])
                else:
                    x1_open = np.min(data[item][0, :]) # they're the same?
                    x2_open = np.max(data[item][0, :])
                    y1_open = np.min(data[item][1, :])
                    y2_open = np.max(data[item][1, :])
            elif item == 'Open Arm 2':
                if f_rotation == 0:
                    x3_open = np.min(data[item][0, :])
                    x4_open = np.max(data[item][0, :])
                    y3_open = np.min(data[item][1, :])
                    y4_open = np.max(data[item][1, :])
                    l_x_open.append([np.minimum(x1_open, x3_open), np.maximum(x2_open, x4_open)])
                    l_y_open.append([np.minimum(y1_open, y3_open), np.minimum(y2_open, y4_open), np.maximum(y1_open, y3_open), np.maximum(y2_open, y4_open)])
                else:
                    x3_open = np.min(data[item][0, :])
                    x4_open = np.max(data[item][0, :])
                    y3_open = np.min(data[item][1, :])
                    y4_open = np.max(data[item][1, :])
                    # l_x_open.append([np.minimum(x1_open, x3_open), np.minimum(x2_open, x4_open), np.maximum(x1_open, x3_open), np.maximum(x2_open, x4_open)])
                    l_y_open.append([np.minimum(y1_open, y3_open), np.maximum(y2_open, y4_open)])
            elif item == 'Closed Arm 1':
                if f_rotation == 0:
                    x1_closed = np.min(data[item][0, :])
                    x2_closed = np.max(data[item][0, :])
                    y1_closed = np.min(data[item][1, :])
                    y2_closed = np.max(data[item][1, :])
                else:
                    x1_closed = np.min(data[item][0, :])
                    x2_closed = np.max(data[item][0, :])
                    y1_closed = np.min(data[item][1, :])
                    y2_closed = np.max(data[item][1, :])
            elif item == 'Closed Arm 2':
                if f_rotation == 0:
                    x3_closed = np.min(data[item][0, :])
                    x4_closed = np.max(data[item][0, :])
                    y3_closed = np.min(data[item][1, :])
                    y4_closed = np.max(data[item][1, :])
                    #l_x_closed.append([np.minimum(x1_closed, x3_closed), np.minimum(x2_closed, x4_closed), np.maximum(x1_closed, x3_closed), np.maximum(x2_closed, x4_closed)])
                    l_y_closed.append([np.minimum(y1_closed, y3_closed), np.maximum(y2_closed, y4_closed)])
                else:
                    x3_closed = np.min(data[item][0, :])
                    x4_closed = np.max(data[item][0, :])
                    y3_closed = np.min(data[item][1, :])
                    y4_closed = np.max(data[item][1, :])
                    l_x_closed.append([np.minimum(x1_closed, x3_closed), np.maximum(x2_closed, x4_closed)])
                    l_y_closed.append([np.minimum(y1_closed, y3_closed), np.minimum(y2_closed, y4_closed), np.maximum(y1_closed, y3_closed), np.maximum(y2_closed, y4_closed)])
            elif item == 'Center':
                x1_center = np.min(data[item][0, :])
                x2_center = np.max(data[item][0, :])
                y1_center = np.min(data[item][1, :])
                y2_center = np.max(data[item][1, :])
                l_x_center.append([x1_center, x2_center])
                l_y_center.append([y1_center, y2_center])
                # the same thing could be done for y coords too
                if f_rotation == 0:
                    l_x_closed.append([np.minimum(x1_closed, x3_closed),  x1_center, x2_center,
                                       np.maximum(x2_closed, x4_closed)])
                else:
                    l_x_open.append([np.minimum(x1_open, x3_open), x1_center, x2_center,
                                    np.maximum(x2_open, x4_open)])
            else:
                pass

        print(l_x_open)
        print(l_y_open)
        print(l_x_closed)
        print(l_y_closed)
        print(l_x_center)
        print(l_y_center)
    return l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center

def find_excel_col_idx(partname):
    parts = ['snout', 'leftear', 'rightear', 'shoulder', 'spine1', 'spine2', 'spine3', 'tailbase', 'tail1', 'tail2', 'tailend']
    # 0 - x, 1 - y, 2 - likelihood
    idx = parts.index(partname)
    excelcolidx = idx*3
    #print(excelcolidx)
    return excelcolidx

def get_beh_dataframe(excelcolidx, l_mousename, l_date, electrodetype,epmtype, exptype):
    '''
    Extracts the x, y coordinates of the body part, data extracted using deep lab cut
    '''
    mouseX = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)#[]
    mouseY = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)#[]

    # x-data - outside the loop because the prefix is the same for all sessions
    if excelcolidx == 0:
        prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000" 
    else: 
        prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000"+"."+str(excelcolidx)

    #print(prefix)
    if isinstance(l_mousename, str):
        l_mousename = (l_mousename,)

    for mouse_idx, name in enumerate(l_mousename):
        print(name)

        path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
    
        #added 2 new mice, M3303 AND F3309     
        if name == "M3303":
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\M3303_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
            if excelcolidx == 0:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
            else:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)
            print(prefix)

        if name == "F3309":
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\F3309_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv" 
            if excelcolidx == 0:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
            else:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)
        
        if electrodetype == 0 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_SiliconProbe\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        if electrodetype == 1 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_InNPhy16\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
        # D1-SALINE EPHYS
        if exptype == "D1":
            path_behavior = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\DLC\\" + name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_200000.csv"
            if excelcolidx == 0:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_200000" 
            else:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_200000"+"."+str(excelcolidx)
        

        behavior_data = pd.read_csv(path_behavior, delimiter=',',low_memory=False)
        #pprint.pprint(behavior_data)
        beh_x = behavior_data[prefix][2:]
        #print(beh_x)

        prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000"+"."+str(excelcolidx+1) # y-data
        if name == "F3309" or name == "M3303" :
            prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx+1)
        if exptype == "D1":
            prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_200000"+"."+str(excelcolidx+1)
        print(prefix)
        
        
        # if l_date[mouse_idx] == "20240623" and name == "M3303":
        #     # path_behavior = "S:\_Tanmai\Python Scripts\DLC\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\videos\\"+name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_500000.csv"
        #     # if excelcolidx == 0:
        #     #     prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
        #     # else:
        #         prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)

        
        # if l_date[mouse_idx] == "20240623" and name == "F3309":
        #     # path_behavior = "S:\_Tanmai\Python Scripts\DLC\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\videos\\"+name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_500000.csv" 
        #     # if excelcolidx == 0:
        #     #     prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
        #     # else:
        #         prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)
        
        #print(prefix)
        beh_y = behavior_data[prefix][2:]
        #print(beh_y)
        mouseX[name] = beh_x
        mouseY[name] = beh_y
    #print(mouseX)
    mouseX = pd.DataFrame(mouseX).to_numpy(dtype = 'float')
    mouseY = pd.DataFrame(mouseY).to_numpy(dtype = 'float')
    return mouseX, mouseY


def get_beh_dataframe_backtail(partname, l_mousename, l_date, electrodetype,epmtype, exptype):
    mouseX = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)#[]
    mouseY = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)#[]
    excelcolidx = np.zeros(5)
    parts_taken = [partname, 'tailbase', 'tail1', 'tail2', 'tailend']
    excelcolidx[0] = find_excel_col_idx(partname=partname)
    excelcolidx[1]= find_excel_col_idx(partname="tailbase")
    excelcolidx[2] = find_excel_col_idx(partname="tail1")
    excelcolidx[3]= find_excel_col_idx(partname="tail2")
    excelcolidx[4] = find_excel_col_idx(partname="tailend")

    # x-data - outside the loop because the prefix is the same for all sessions
    
    #print(prefix)
    if isinstance(l_mousename, str):
        l_mousename = (l_mousename,)

    for mouse_idx, name in enumerate(l_mousename):
        path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        if l_date[mouse_idx] == "20230512" and name == "F2491":
            path_behavior = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F2491\\20230512\\dlc\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
        
                #added 2 new mice, M3303 AND F3309     
        if l_date[mouse_idx] == "20240623" and name == "M3303":
            path_behavior = "S:\_Tanmai\Python Scripts\DLC\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\videos\\"+name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_500000.csv"
            if excelcolidx == 0:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
            else:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)

        
        if l_date[mouse_idx] == "20240623" and name == "F3309":
            path_behavior = "S:\_Tanmai\Python Scripts\DLC\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\videos\\"+name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_500000.csv" 
            if excelcolidx == 0:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000" 
            else:
                prefix = "DLC_resnet50_D1_ephys_YCMar26shuffle1_500000"+"."+str(excelcolidx)
        
        if electrodetype == 0 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_SiliconProbe\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        if electrodetype == 1 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_InNPhy16\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
        for i in range(len(excelcolidx)):
            
            excelcolidx[i]=int(excelcolidx[i])
            if excelcolidx[i] == 0:
                prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000" 
            else: 
                prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000"+"."+str(int(excelcolidx[i]))

        
            behavior_data = pd.read_csv(path_behavior, delimiter=',',low_memory=False)
            #pprint.pprint(behavior_data)
            beh_x = behavior_data[prefix][2:]
            #print(beh_x)

            prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000"+"."+str(int(excelcolidx[i]+1)) # y-data
            beh_y = behavior_data[prefix][2:]
            #print(beh_y)
            if i==0:
                mouseX[name+"_"+parts_taken[i]] = beh_x
                mouseY[name+"_"+parts_taken[i]] = beh_y
            else:
                mouseX[name+"_"+parts_taken[i]] = beh_x
                mouseY[name+"_"+parts_taken[i]] = beh_y

    #print(mouseX)
    mouseX = pd.DataFrame(mouseX).to_numpy(dtype = 'float')
    mouseY = pd.DataFrame(mouseY).to_numpy(dtype = 'float')
    #print(mouseX) # n x 5

    return mouseX, mouseY

'''
def get_arm_idx(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname, electrodetype,epmtype):
    l_arm_idx = []
    l_X, l_Y = [], [] #create two lists, one l_X for the coordinates in X and one l_Y for the coordinates in Y
    origin = []
    for mouse_idx in range(len(l_mouse_name)):
        #origin = origin.append([np.mean(l_x_center[mouse_idx]),np.mean(l_y_center[mouse_idx])])
        #print(origin)
        l_arm_idx_indiv = []
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        #path = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_bonsai.csv"
        excelcolidx = find_excel_col_idx(partname)
        #print(excelcolidx)
        #print(mouse_name)
        #print(date)
        mouseX, mouseY = get_beh_dataframe(excelcolidx, mouse_name, date, electrodetype,epmtype) 
        
        l_X.append(mouseX) # might have nans??
        l_Y.append(mouseY)
        ref_x_open = l_x_open[mouse_idx]
        ref_y_open = l_y_open[mouse_idx]
        ref_x_closed = l_x_closed[mouse_idx]
        ref_y_closed = l_y_closed[mouse_idx]
        ref_x_center = l_x_center[mouse_idx]
        ref_y_center = l_y_center[mouse_idx]
        origin = [np.mean(ref_x_center),np.mean(ref_y_center)]
        count_outside = 0 # this is to count the first few counts outside (assign idx=8 to these), and then assign -1 to the rest (headdips)
        outsidestopped = 0 # flag variable to tell us if the mouse is now on the EPM and not in the process of being placed
        isOAprev = 0
        #print(origin)
        for i in range(len(mouseX)): # the length of the coordinates data in X and Y
            #print(mouseX[i])
            x = mouseX[i] # read the x location of the mouse
            y = mouseY[i] # read the y location of the mouse
            arm_idx = 8 #there are 5 idx (0: open arm 1 /1: open arm 2/ 2: closed arm 1 /3: closed arm 2 /4: center)
            #print(l_rotation)
            if l_rotation[mouse_idx] == 0:
                #l_arm_idx_indiv.append(arm_idx) # why?
                if ref_y_closed[0] < y < ref_y_closed[1]: # why +5?
                    if ref_x_closed[0] -20 < x < ref_x_closed[1] :
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_x_closed[2] < x < ref_x_closed[3]+ 20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the open arm
                if ref_x_open[0] < x < ref_x_open[1]: # why +5?
                    if ref_y_open[0]< y < ref_y_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_y_open[2] < y < ref_y_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    #arm_idx = 4
                    A = [ref_x_center[0] - origin[0], ref_y_center[1] - origin[1]]
                    B = [ref_x_center[1] - origin[0], ref_y_center[1] - origin[1]]
                    C = [ref_x_center[1] - origin[0], ref_y_center[0] - origin[1]]
                    D = [ref_x_center[0] - origin[0], ref_y_center[0] - origin[1]]
                    E = [x - origin[0], y - origin[1]]
                    isOAprev = 0
                    if check_cross_prod(A, B, E):
                        arm_idx = 4
                    elif check_cross_prod(B, C, E):
                        arm_idx = 6
                    elif check_cross_prod(C, D, E):
                        arm_idx = 5
                    else:
                        arm_idx = 7

            else:
                # check whether it is in the open arm
                if ref_y_open[0]< y < ref_y_open[1]: # why +5?
                    if ref_x_open[0]< x < ref_x_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_x_open[2] < x < ref_x_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the closed arm
                if ref_x_closed[0] < x < ref_x_closed[1]:
                    if ref_y_closed[0] -20 < y < ref_y_closed[1]:
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_y_closed[2] < y < ref_y_closed[3]+20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    isOAprev = 0
                    arm_idx = 4

            if i==0 and arm_idx == 8:
                count_outside += 1 # initially checking if mouse is outside
            if arm_idx == 8 and outsidestopped == 0:
                count_outside += 1 # checking if mouse is still outside and not yet in EPM
            if count_outside>0 and arm_idx != 8 and l_arm_idx_indiv[i-1] == 8:
                outsidestopped = 1 # this is the point when mouse is fully inside EPM and experiment starts here
            if arm_idx == 8 and outsidestopped == 1 and isOAprev == 1:
                arm_idx = -1 # everything outside the EPM from OA is now treated as headdips 

            l_arm_idx_indiv.append(arm_idx)
        l_arm_idx.append(l_arm_idx_indiv)
    return l_arm_idx, l_X, l_Y
'''


def get_arm_idx(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname, electrodetype,epmtype, exptype):
    l_arm_idx = []
    l_X, l_Y = [], [] #create two lists, one l_X for the coordinates in X and one l_Y for the coordinates in Y
    origin = []
    outside = 8 
    
   
    for mouse_idx in range(len(l_mouse_name)):
        #origin = origin.append([np.mean(l_x_center[mouse_idx]),np.mean(l_y_center[mouse_idx])])
        #print(origin)
        l_arm_idx_indiv = []
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        #path = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_bonsai.csv"
        excelcolidx = find_excel_col_idx(partname)
        #print(excelcolidx)
        #print(mouse_name)
        #print(date)
        mouseX, mouseY = get_beh_dataframe(excelcolidx, mouse_name, date, electrodetype,epmtype, exptype) #get_beh_dataframe_backtail(partname, l_mousename, l_date, electrodetype,epmtype) 
        # print(mouseX)
        l_X.append(mouseX) # might have nans??
        l_Y.append(mouseY)
        isOAprev = 0
        
        for i in range(len(mouseX)): # the length of the coordinates data in X and Y
            x = mouseX[i] # read the x location of the mouse
            y = mouseY[i] # read the y location of the mouse
            arm_idx = outside #there are 5 idx (0: open arm 1 /1: open arm 2/ 2: closed arm 1 /3: closed arm 2 /4: center)
            ref_x_open = l_x_open[mouse_idx]
            ref_y_open = l_y_open[mouse_idx]
            ref_x_closed = l_x_closed[mouse_idx]
            ref_y_closed = l_y_closed[mouse_idx]
            ref_x_center = l_x_center[mouse_idx]
            ref_y_center = l_y_center[mouse_idx]
            origin = [np.mean(ref_x_center),np.mean(ref_y_center)]
            if l_rotation[mouse_idx] == 0:
                #l_arm_idx_indiv.append(arm_idx) # why?
                if ref_y_closed[0] < y < ref_y_closed[1]: # why +5?
                    if ref_x_closed[0] -20 < x < ref_x_closed[1] :
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_x_closed[2] < x < ref_x_closed[3]+ 20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the open arm
                if ref_x_open[0] < x < ref_x_open[1]: # why +5?
                    if ref_y_open[0]< y < ref_y_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_y_open[2] < y < ref_y_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    #arm_idx = 4
                    A = [ref_x_center[0] - origin[0], ref_y_center[1] - origin[1]]
                    B = [ref_x_center[1] - origin[0], ref_y_center[1] - origin[1]]
                    C = [ref_x_center[1] - origin[0], ref_y_center[0] - origin[1]]
                    D = [ref_x_center[0] - origin[0], ref_y_center[0] - origin[1]]
                    E = [x - origin[0], y - origin[1]]
                    isOAprev = 0
                    if check_cross_prod(A, B, E):
                        arm_idx = 4
                    elif check_cross_prod(B, C, E):
                        arm_idx = 6
                    elif check_cross_prod(C, D, E):
                        arm_idx = 5
                    else:
                        arm_idx = 7
                
                if arm_idx == outside and isOAprev == 1:
                    arm_idx = -1
                l_arm_idx_indiv.append(arm_idx)

            else:
                # check whether it is in the open arm
                if ref_y_open[0] - 5 < y < ref_y_open[1] + 5: # why +5?
                    if ref_x_open[0] < x < ref_x_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_x_open[2] < x < ref_x_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the closed arm
                if ref_x_closed[0] < x < ref_x_closed[1]:
                    if ref_y_closed[0] -20 < y < ref_y_closed[1]:
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_y_closed[2] < y < ref_y_closed[3]+20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    #arm_idx = 4
                    A = [ref_x_center[0] - origin[0], ref_y_center[1] - origin[1]]
                    B = [ref_x_center[1] - origin[0], ref_y_center[1] - origin[1]]
                    C = [ref_x_center[1] - origin[0], ref_y_center[0] - origin[1]]
                    D = [ref_x_center[0] - origin[0], ref_y_center[0] - origin[1]]
                    E = [x - origin[0], y - origin[1]]
                    isOAprev = 0
                    if check_cross_prod(A, B, E):
                        arm_idx = 6
                    elif check_cross_prod(B, C, E):
                        arm_idx = 4
                    elif check_cross_prod(C, D, E):
                        arm_idx = 7
                    else:
                        arm_idx = 5
                
                if arm_idx == outside and isOAprev == 1:
                    arm_idx = -1
                l_arm_idx_indiv.append(arm_idx)
        l_arm_idx.append(l_arm_idx_indiv)
    return l_arm_idx, l_X, l_Y



def get_arm_idx_backtail(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname, electrodetype,epmtype, exptype):
    l_arm_idx = []
    l_X, l_Y = [], [] #create two lists, one l_X for the coordinates in X and one l_Y for the coordinates in Y
    origin = []
    outside = 8 # 5
    for mouse_idx in range(len(l_mouse_name)):
        #origin = origin.append([np.mean(l_x_center[mouse_idx]),np.mean(l_y_center[mouse_idx])])
        #print(origin)
        l_arm_idx_indiv = []
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        #path = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_bonsai.csv"
        
        mouseX, mouseY =get_beh_dataframe_backtail(partname, mouse_name, date, electrodetype,epmtype, exptype) 
        
        l_X.append(mouseX) # might have nans??
        l_Y.append(mouseY)
        isOAprev = 0
        print(print(mouseX.shape))
        for i in range(len(mouseX)): # the length of the coordinates data in X and Y
            x = mouseX[i,0] # read the x location of the mouse
            y = mouseY[i,0] # read the y location of the mouse
            
            #print(mouseX.shape)
            arm_idx = outside #there are 5 idx (0: open arm 1 /1: open arm 2/ 2: closed arm 1 /3: closed arm 2 /4: center)
            ref_x_open = l_x_open[mouse_idx]
            ref_y_open = l_y_open[mouse_idx]
            ref_x_closed = l_x_closed[mouse_idx]
            ref_y_closed = l_y_closed[mouse_idx]
            ref_x_center = l_x_center[mouse_idx]
            ref_y_center = l_y_center[mouse_idx]
            origin = [np.mean(ref_x_center),np.mean(ref_y_center)]
            if l_rotation[mouse_idx] == 0:
                #l_arm_idx_indiv.append(arm_idx) # why?
                if ref_y_closed[0] < y < ref_y_closed[1]: # why +5?
                    if ref_x_closed[0] -20 < x < ref_x_closed[1] :
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_x_closed[2] < x < ref_x_closed[3]+ 20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the open arm
                if ref_x_open[0] < x < ref_x_open[1]: # why +5?
                    if ref_y_open[0]< y < ref_y_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_y_open[2] < y < ref_y_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    #arm_idx = 4
                    A = [ref_x_center[0] - origin[0], ref_y_center[1] - origin[1]]
                    B = [ref_x_center[1] - origin[0], ref_y_center[1] - origin[1]]
                    C = [ref_x_center[1] - origin[0], ref_y_center[0] - origin[1]]
                    D = [ref_x_center[0] - origin[0], ref_y_center[0] - origin[1]]
                    E = [x - origin[0], y - origin[1]]
                    isOAprev = 0
                    if check_cross_prod(A, B, E):
                        arm_idx = 4
                    elif check_cross_prod(B, C, E):
                        arm_idx = 6
                    elif check_cross_prod(C, D, E):
                        arm_idx = 5
                    else:
                        arm_idx = 7
                
                if arm_idx == outside and isOAprev == 1:
                    arm_idx = -1
                l_arm_idx_indiv.append(arm_idx)

            else:
                # check whether it is in the open arm
                if ref_y_open[0] - 5 < y < ref_y_open[1] + 5: # why +5?
                    if ref_x_open[0] < x < ref_x_open[1]:
                        arm_idx = 0
                        isOAprev = 1
                    elif ref_x_open[2] < x < ref_x_open[3]:
                        arm_idx = 1
                        isOAprev = 1
                    else:
                        pass
                # check whether it is in the closed arm
                if ref_x_closed[0] < x < ref_x_closed[1]:
                    if ref_y_closed[0] -20 < y < ref_y_closed[1]:
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_y_closed[2] < y < ref_y_closed[3]+20:
                        arm_idx = 3
                        isOAprev = 0
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    #arm_idx = 4
                    A = [ref_x_center[0] - origin[0], ref_y_center[1] - origin[1]]
                    B = [ref_x_center[1] - origin[0], ref_y_center[1] - origin[1]]
                    C = [ref_x_center[1] - origin[0], ref_y_center[0] - origin[1]]
                    D = [ref_x_center[0] - origin[0], ref_y_center[0] - origin[1]]
                    E = [x - origin[0], y - origin[1]]
                    isOAprev = 0
                    if check_cross_prod(A, B, E):
                        arm_idx = 6
                    elif check_cross_prod(B, C, E):
                        arm_idx = 4
                    elif check_cross_prod(C, D, E):
                        arm_idx = 7
                    else:
                        arm_idx = 5
                
                if arm_idx == outside and isOAprev == 1:
                    arm_idx = -1
                l_arm_idx_indiv.append(arm_idx)
        l_arm_idx.append(l_arm_idx_indiv)
    return l_arm_idx, l_X, l_Y


def select_events(l_arm_idx, l_X, l_Y, l_sample_rate_video, l_mouse_name, l_mouse_date, exp_type="reg"):
    threshold = 1 #minimal threshold when the mouse stays in the defined region for >=1 s (1s = 20/20Hz)
    l_event_idx, l_duration_idx, l_starting_idx, l_end_idx= [], [], [], []
    l_X_selected, l_Y_selected = [], []
    outside = 8 # 5
    for mouse_idx in range(len(l_arm_idx)):
        l_event_idx_indiv = []
        l_duration_idx_indiv = [] #indiv = for each mouse
        l_starting_idx_indiv = []
        l_end_idx_indiv = []
        l_X_indiv, l_Y_indiv = [], []
        mouse_arm_idx = l_arm_idx[mouse_idx]
        # onset_ephys_idx = get_video_frames_onsets(path_cdata, total_ch_number=27,
        #                                           ch_id=19)  # to synchronize with the behavior
        # offset = int(np.ceil((onset_ephys_idx) / 30000))
        #new
        if exp_type == "reg":
            path_offset = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy"+"\\"+l_mouse_name[mouse_idx]+"\\"+l_mouse_date[mouse_idx]+"\\"+l_mouse_name[mouse_idx]+"_EPM_startingtime.txt"
        else: 
            path_offset = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\\"+l_mouse_name[mouse_idx]+"\\"+l_mouse_date[mouse_idx]+"\\"+l_mouse_name[mouse_idx]+"_EPM_startingtime.txt"
        
        if l_mouse_name[mouse_idx] == "M3303" or l_mouse_name[mouse_idx] == "F3309":
            path_offset = "S:\\___DATA\\in_vivo_ePhys\\DATA_PAPER\\Raw\\"+str(l_mouse_name[mouse_idx])+"\\"+str(l_mouse_date[mouse_idx])+"\\"+str(l_mouse_name[mouse_idx])+"_EPM_startingtime.txt"
        
        offset = int(float(open(path_offset,'r').read())*l_sample_rate_video[mouse_idx])
        print('offset:')
        print(offset)

        l_X_indiv = []#np.array(l_X[mouse_idx])
        l_Y_indiv = [] #np.array(l_Y[mouse_idx])
        # l_starting_idx.append(np.arange(0,np.array(mouse_arm_idx).size))
        # l_duration_idx.append(np.ones(np.array(mouse_arm_idx).size))
        # l_event_idx.append(np.array(mouse_arm_idx))
        # l_end_idx.append(np.arange(0,np.array(mouse_arm_idx).size))

        duration_counter = 1 #?
        for i in range(offset, len(mouse_arm_idx)-1):
            if mouse_arm_idx[i+1] == mouse_arm_idx[i]:
                duration_counter+=1 #how many times we have the same arm idx in the same event
            else:
                if duration_counter>=1:
                    if (duration_counter >= threshold) and (mouse_arm_idx[i]!=outside):# and (mouse_arm_idx[i]!=4) and (mouse_arm_idx[i]!=5): #do not consider idx4: center and idx:5 outside the EPM
                        l_starting_idx_indiv.append(i-duration_counter+1)
                        l_end_idx_indiv.append(i) #mouse_arm idx(i) different from mouse_arm idx(i+1): new idx (that's why we have start and end idx)
                        l_event_idx_indiv.append(mouse_arm_idx[i]*np.ones(len(l_X[mouse_idx][i-duration_counter+1:i+1])))
                        l_duration_idx_indiv.append(duration_counter)
                        l_X_indiv.append(l_X[mouse_idx][i-duration_counter+1:i+1]) #all of the X coordinates for the indiv selected events
                        l_Y_indiv.append(l_Y[mouse_idx][i-duration_counter+1:i+1]) #all of the Ycoordinates for the indiv selected events
                    else:
                        pass
                duration_counter = 1
        # outside=8
        # for i in range(int(offset), len(mouse_arm_idx)-1):
        #     if mouse_arm_idx[i] !=outside:
        #         l_starting_idx_indiv.append(i)
        #         l_end_idx_indiv.append(i)
        #         l_event_idx_indiv.append(mouse_arm_idx[i])
        #         l_duration_idx_indiv.append(1)
        #         l_X_indiv.append(l_X[mouse_idx][i:i+1])
        #         l_Y_indiv.append(l_Y[mouse_idx][i:i+1])
        
        l_event_idx.append(l_event_idx_indiv)
        l_duration_idx.append(l_duration_idx_indiv)
        l_starting_idx.append(l_starting_idx_indiv)
        l_end_idx.append(l_end_idx_indiv)
        l_X_selected.append(l_X_indiv)
        l_Y_selected.append(l_Y_indiv)
    return l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected


def get_electrophysiological_data(l_mouse_name, l_date, l_rotation, l_sample_rate_video, l_path_cdata, l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected, exptype, sorter):
    l_data_PCA = []
    l_arm_idx_PCA = []
    l_data_PCA_25ms=[]
    cluster_id_list =[]
    for mouse_idx in range(len(l_mouse_name)):
        l_t_starting_video = np.asarray(l_starting_idx[mouse_idx]) / l_sample_rate_video[mouse_idx]
        l_t_end_video = np.asarray(l_end_idx[mouse_idx]) / l_sample_rate_video[mouse_idx]
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        print(mouse_name)
        # path_ephys = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting_Kilosort.csv"
        if sorter == "kilo":
            path_ephys = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting_Kilosort.csv"
            if mouse_name == "M3303" or mouse_name == "F3309":
                path_ephys = "S:\\___DATA\\in_vivo_ePhys\\DATA_PAPER\\Raw\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting_Kilosort.csv"
            
        
        if exptype == "D1":
            if sorter == "kilo":
                path_ephys = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting.mat"
            else:
                path_ephys = "y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting_remov.csv"
            
        path_cdata = l_path_cdata[mouse_idx]
        onset_ephys_idx = get_video_frames_onsets(path_cdata, total_ch_number=27,
                                                  ch_id=19)  # to synchronize with the behavior
        
        if len(l_X_selected[mouse_idx][0].shape) == 1:
            n_beh = 1
        else:
            n_beh = l_X_selected[mouse_idx][0].shape[1]
        separator = 0
        if sorter == "kilo" and exptype == "D1":
            # path_ephys = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting_Kilosort.csv"
            matfile = loadmat(path_ephys)['neuronsinfoG']
            idx = np.round(30000*matfile[:,-1])
            cluster_id = matfile[:,0]
            # print(cluster_id)
            separator = "mat"
        else: 
            separator = check_csv_separator(path_ephys)
            
            idx, cluster_id = np.genfromtxt(path_ephys, dtype=int, skip_header=1, unpack=True,delimiter=separator)

        print(list(set(cluster_id))) #prints the cluster id list 

        if separator is not None: 
            l_neuron_id = list(set(cluster_id))
            
            n_neurons = len(l_neuron_id)
            n_bins_total = 0  # total num of bins across all events
            n_bins_total_25 = 0

            l_t_ephys = (idx - onset_ephys_idx) / 30000  # beginning of the ephys recording
            bin_size = round(1 / l_sample_rate_video[mouse_idx],6)  # 50ms, arbitrary fixed (because, frame rate is 20 Hz, so 1/20=50ms)
            bin_size_15 = round(1 / l_sample_rate_video[mouse_idx],6)
            for i in range(len(l_t_starting_video)):
                t_starting_bin = round(l_t_starting_video[i], 6)
                t_end_bin = round(l_t_end_video[i], 6)
                bin_vec = np.around(t_starting_bin - round(bin_size / 2, 6) + np.arange(0, round((t_end_bin + round(bin_size / 2, 6) - (t_starting_bin - round(bin_size / 2, 6))) / bin_size) + 1) * bin_size,6)  # ?? - time bins from starttime - timebin/2 to endtime+timebin/2
                bin_vec_25ms = np.linspace(start=bin_vec[0], stop=bin_vec[-1], num=int(len(bin_vec)*2 )-1)
            

                for j in range(len(l_neuron_id)):
                    neuron_id = l_neuron_id[j]
                    neuron_idx_spike = np.where(cluster_id == neuron_id)[0]
                    spike_timing_neuron = l_t_ephys[neuron_idx_spike]
                    # print(spike_timing_neuron)
                    hist_neuron = np.histogram(spike_timing_neuron, bin_vec)[0]
                    hist_neur_25ms = np.histogram(spike_timing_neuron, bin_vec_25ms)[0]
                    
                    print(hist_neur_25ms)
                    if j == 0:
                        n_bins_total += len(hist_neuron)  # this is to calculate total number of bins across all events
                        n_bins_total_25 += len(hist_neur_25ms)

            data_PCA = np.zeros((n_neurons, n_bins_total)) * np.nan
            data_PCA_25ms = np.zeros((n_neurons, n_bins_total_25)) * np.nan
            arm_idx_PCA = np.zeros((1+n_beh+n_beh,n_bins_total)) * np.nan #np.zeros((3,n_bins_total)) * np.nan #np.zeros((1+n_beh+n_beh,n_bins_total)) * np.nan
            

            n_bins_idx = 0
            n_bins_25ms_idx = 0
            offset_indlc = len(l_X_selected[mouse_idx])-n_bins_total

            for i in range(len(l_t_starting_video)):
                t_starting_bin = round(l_t_starting_video[i], 6)
                t_end_bin = round(l_t_end_video[i], 6) 
                bin_vec = np.around(t_starting_bin - round(bin_size / 2, 6) + np.arange(0, round((t_end_bin + round(bin_size / 2, 6) - (t_starting_bin - round(bin_size / 2, 6))) / bin_size) + 1) * bin_size, 6)  # ??
                #bin_vec_25ms = np.around(t_starting_bin - round(bin_size / 4, 6) + np.arange(0, round((t_end_bin + round(bin_size / 4, 6) - (t_starting_bin - round(bin_size / 4, 6))) / (bin_size/2)) + 1) * bin_size/2, 6)  # ??
                
                bin_vec_25ms = np.linspace(start=bin_vec[0], stop=bin_vec[-1], num=int(len(bin_vec)*2)-1)
                

                for j in range(len(l_neuron_id)):  # j = idx for access one neuron
                    neuron_id = l_neuron_id[j]
                    neuron_idx_spike = np.where(cluster_id == neuron_id)[0]
                    spike_timing_neuron = l_t_ephys[neuron_idx_spike]
                    hist_neuron = np.histogram(spike_timing_neuron, bin_vec)[0]
                    data_PCA[j, n_bins_idx:n_bins_idx + len(hist_neuron)] = hist_neuron

                    hist_neur_25ms = np.histogram(spike_timing_neuron, bin_vec_25ms)[0]
                    data_PCA_25ms[j, n_bins_25ms_idx:n_bins_25ms_idx + len(hist_neur_25ms)] = hist_neur_25ms

                    if j == len(l_neuron_id) - 1:
                        # swap
                        #print(np.array(l_X_selected[mouse_idx][i]).squeeze().size)
                        arm_idx_PCA[0,n_bins_idx:n_bins_idx + len(hist_neuron)] = l_event_idx[mouse_idx][i]  # swap these two lines of code ???
                        arm_idx_PCA[1:1+n_beh,n_bins_idx:n_bins_idx + len(hist_neuron)] = np.array(l_X_selected[mouse_idx][i]).T # swap these two lines of code ???
                        arm_idx_PCA[1+n_beh:1+2*n_beh,n_bins_idx:n_bins_idx + len(hist_neuron)] = np.array(l_Y_selected[mouse_idx][i]).T  # swap these two lines of code ???
                        n_bins_idx += len(hist_neuron)
                        n_bins_25ms_idx += len(hist_neur_25ms)

            l_data_PCA.append(data_PCA)
            l_arm_idx_PCA.append(arm_idx_PCA)
            l_data_PCA_25ms.append(data_PCA_25ms)
            cluster_id_list.append(neuron_id)
    return l_data_PCA, l_arm_idx_PCA, l_data_PCA_25ms,cluster_id_list


def data_centering(l_X, l_Y, l_x_center, l_y_center):
    origin = [None]*(len(l_x_center))
    for i in range(len(l_x_center)):
        origin[i] = [np.mean(l_x_center[i]),np.mean(l_y_center[i]) ]
        l_X[i] = l_X[i] - origin[i][0]
        l_Y[i] = l_Y[i] - origin[i][1]
    return l_X, l_Y, origin

def distanceGen(l_beh):
    l_beh_lowdim = [None]*len(l_beh)
    for i in range(len(l_beh)):
        l_beh_lowdim[i] = np.empty((2, l_beh[i].shape[-1]))
        l_beh_lowdim[i][0,:] = l_beh[i][0,:]
        l_beh_lowdim[i][1,:] = (l_beh[i][1,:]**2 + l_beh[i][2,:]**2)**0.5
    return l_beh_lowdim

def distanceGen_tailback(l_beh):
    l_beh_lowdim = [None]*len(l_beh)
    for i in range(len(l_beh)):
        l_beh_lowdim[i] = np.empty((6, l_beh[i].shape[-1]))
        l_beh_lowdim[i][0,:] = l_beh[i][0,:]
        for j in range(1,6):
            l_beh_lowdim[i][j,:] = (l_beh[i][j,:]**2 + l_beh[i][j+5,:]**2)**0.5
    return l_beh_lowdim

def calc_thresh(excelcolidx, l_mouse_name, exptype):
    thresh = np.zeros(len(l_mouse_name))
    
    # for j, name in enumerate(l_mouse_name):
        # path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_skeleton.csv"
        # if exptype == "D1":
        #     path_behavior = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\D1-Ephys\\Data\DLC\\" + name+"_EPMDLC_resnet50_D1_ephys_YCMar26shuffle1_200000.csv"
            
        # skeleton_data = pd.read_csv(path_behavior, delimiter=',',low_memory=False)
        # skeleton_data = pd.DataFrame(skeleton_data).to_numpy()
        # # print(skeleton_data)
        # if exptype == "D1":
        #     skeleton_data = skeleton_data[2:,:].astype('float')
        # else:
        #     skeleton_data = skeleton_data[1:,:].astype('float')
            

        # if excelcolidx<=3*3:
        #     #thresh[j] = 0# abs(part-spine1)
        #     if excelcolidx == 0:
        #         #excelcolidx = 6
        #         for i in range(6, 3*3+1, 3):
        #             thresh[j] = thresh[j]+np.average(skeleton_data[:,i+1])
        #     else:
        #         for i in range(excelcolidx, 4*3+1, 3):
        #             if excelcolidx == 3 and i == 6:
        #                 continue
        #             else:
        #                 thresh[j] = thresh[j]+np.average(skeleton_data[:,i+1])

        # elif excelcolidx>7*3:
        #     #thresh = 0#abs(part-tailbase)
        #     for i in range(excelcolidx-2, 7*3, -3):
        #         thresh[j] = thresh[j]+np.average(skeleton_data[:,i])

        # else:
        #     thresh = 0
    thresh[:] = 35
    return thresh

def widen_EPM(l_x_open, l_y_open, thresh):
    
    for i in range(len(l_x_open)):
        l_x_open[i] = np.sort(l_x_open[i])
        l_y_open[i] = np.sort(l_y_open[i])
        # print(l_x_open[i].size)
        if (l_x_open[i].size==2):
            l_x_open[i][0] = l_x_open[i][0] - thresh[i]
            l_x_open[i][1] = l_x_open[i][1] + thresh[i]
            l_y_open[i][0] = l_y_open[i][0] - thresh[i]
            #l_y_open[i][1] = l_y_open[i][1] - thresh[i]
            #l_y_open[i][2] = l_y_open[i][2] + thresh[i]
            l_y_open[i][3] = l_y_open[i][3] + thresh[i]
        else:
            l_y_open[i][0] = l_y_open[i][0] - thresh[i]
            l_y_open[i][1] = l_y_open[i][1] + thresh[i]
            l_x_open[i][0] = l_x_open[i][0] - thresh[i]
            #l_x_open[i][1] = l_x_open[i][1] - thresh[i]
            #l_x_open[i][2] = l_x_open[i][2] + thresh[i]
            l_x_open[i][3] = l_x_open[i][3] + thresh[i]
    return l_x_open, l_y_open


def behData_removeOutliers(X, Y, l_x_closed, l_y_closed, l_x_open, l_y_open):
    
    # 640 x 512
    # take max distance being able to travel as 20 pix
    # find pix per cm 

    maxdist = 10# in cm
    epm_width, epm_length = [],[]
    
    for i in range(len(l_x_closed)):
        print(l_x_open[i])
        if len(l_x_open[i])==2:
            epm_width.append(max(-np.min(l_x_closed[i]) + np.max(l_x_closed[i]), -np.min(l_x_open[i]) + np.max(l_x_open[i])))
            epm_length.append(max(-np.min(l_y_closed[i]) + np.max(l_y_closed[i]), -np.min(l_y_open[i]) + np.max(l_y_open[i])))
        else:
            epm_length.append(max(-np.min(l_x_closed[i]) + np.max(l_x_closed[i]), -np.min(l_x_open[i]) + np.max(l_x_open[i])))
            epm_width.append(max(-np.min(l_y_closed[i]) + np.max(l_y_closed[i]), -np.min(l_y_open[i]) + np.max(l_y_open[i])))
    
    epm_length = np.max(np.array(epm_length))
    epm_width = np.max(np.array(epm_width))
    print(epm_length)
    print(epm_width)
    pixpercm_x = (epm_width)/75
    pixpercm_y = (epm_length)/75
    # can't travel more than 5cm 
    max_X = 20#pixpercm_x*maxdist
    max_Y = 20#pixpercm_y*maxdist
    X_post = []
    Y_post = []
    for j in range(len(X)):
        
        X_post.append(np.copy(X[j]))
        Y_post.append(np.copy(Y[j]))
        # check number of columns of the beh matrices
        print(X[j][1])
        h = len(X[j][1])
        if h==0:
            x_diff = np.diff(np.squeeze(X_post[j]))
            y_diff = np.diff(np.squeeze(X_post[j]))

            for i in range(len(x_diff)):
                #print(X_post[j][i]-X_post[j][i+1])
                if np.abs(x_diff[i])>max_X or np.abs(y_diff[i])>max_Y:#np.abs(X_post[j][i]-X_post[j][i+1])>max_X: #or ~(l_X[j][i]<l_X[j][i+1]<l_X[j][i+2]):
                    #X[j][i+1] = X[j][i]
                    #Y[j][i+1] = Y[j][i]

                    #interpolation: for -5 to +5 points
                    if i>5 and i<len(X[j])-5:
                        fp = X_post[j][i-5:i+5+1]
                        tp = np.delete(np.linspace(-5, 5, len(fp)), obj=5).squeeze()
                        xp = np.delete(fp, obj=5).squeeze()
                        #print(xp)
                        #print(tp)
                        #print(np.interp(0, tp, xp))
                        x_interp = np.interp(0, tp, xp)

                        fp = Y_post[j][i-5:i+5+1]
                        yp = np.delete(fp, obj=5).squeeze()
                        #print(yp)
                        y_interp = np.interp(0, tp, yp)
                        X_post[j][i+1] = x_interp
                        Y_post[j][i+1] = y_interp
                    else: 
                        X[j][i+1] = X[j][i]
                        Y[j][i+1] = Y[j][i]
                        X_post[j][i+1] = X_post[j][i]
                        Y_post[j][i+1] = Y_post[j][i]
        else:
            for k in range(h):
                x_diff = np.diff(np.squeeze(X_post[j][:,k]))
                y_diff = np.diff(np.squeeze(X_post[j][:,k]))

                for i in range(len(x_diff)):
                    #print(X_post[j][i]-X_post[j][i+1])
                    if np.abs(x_diff[i])>max_X or np.abs(y_diff[i])>max_Y:#np.abs(X_post[j][i]-X_post[j][i+1])>max_X: #or ~(l_X[j][i]<l_X[j][i+1]<l_X[j][i+2]):
                        #X[j][i+1] = X[j][i]
                        #Y[j][i+1] = Y[j][i]

                        #interpolation: for -5 to +5 points
                        if i>5 and i<len(X[j])-5:
                            fp = X_post[j][i-5:i+5+1,k]
                            tp = np.delete(np.linspace(-5, 5, len(fp)), obj=5).squeeze()
                            xp = np.delete(fp, obj=5).squeeze()
                            #print(xp)
                            #print(tp)
                            #print(np.interp(0, tp, xp))
                            x_interp = np.interp(0, tp, xp)

                            fp = Y_post[j][i-5:i+5+1, k]
                            yp = np.delete(fp, obj=5).squeeze()
                            #print(yp)
                            y_interp = np.interp(0, tp, yp)
                            X_post[j][i+1,k] = x_interp
                            Y_post[j][i+1,k] = y_interp
                        else: 
                            X[j][i+1,k] = X[j][i,k]
                            Y[j][i+1,k] = Y[j][i,k]
                            X_post[j][i+1,k] = X_post[j][i,k]
                            Y_post[j][i+1,k] = Y_post[j][i,k]
                #if # np.abs(Y_post[j][i]-Y_post[j][i+1])>max_Y:# or ~(l_Y[j][i]<l_Y[j][i+1]<l_Y[j][i+2]):
                    #X[j][i+1] = X[j][i]
                    #Y[j][i+1] = Y[j][i]
                    #X_post[j][i+1] = X_post[j][i]
                    #Y_post[j][i+1] = Y_post[j][i]

    return X_post, Y_post


def check_cross_prod(A, B, C):
    #if (AxB * AxC >= 0 && BxC * BxA >= 0) - C is the vector to check for 
    return (A[0] * B[1] - A[1] * B[0]) * (A[0] * C[1] - A[1] * C[0]) >= 0 and (B[0] * C[1] - B[1] * C[0]) * (B[0] * A[1] - B[1] * A[0])>=0

def plotArm_idx(X,Y, label, ax,gray):
    # find indices for each arm tpe and assign colour. save both indices and colours
    #print(label)
    label = np.array(label)
    epm0 = label == 0
    epm1 = label == 1
    epm2 = label == 2
    epm3 = label == 3
    epm4 = label == 4
    epm5 = label == 5
    epm6 = label == 6
    epm7 = label == 7
    epm8 = label == 8
    epmneg = label == -1
    #print(epm7[epm7==True].size)
    if not gray:
        e0_cmap = 'b'
        e2_cmap = 'r'
        e1_cmap = 'g'
        e3_cmap = 'y'
        e4_cmap = 'pink'
        e5_cmap = 'cyan'
        e6_cmap = 'magenta'
        e7_cmap = 'orange'
        e8_cmap = 'dimgrey'
        eneg_cmap = 'black'
    
    else:
        e0_cmap = 'black'
        e2_cmap = 'black'
        e1_cmap = 'black'
        e3_cmap = 'black'
        e4_cmap = 'black'
        e5_cmap = 'black'
        e6_cmap = 'black'
        e7_cmap = 'black'
        e8_cmap = 'black'
        eneg_cmap = 'black'
    #print(epm0)
    e0=ax.scatter(X[epm0],Y[epm0],
               c=e0_cmap, s=0.5)
    e1=ax.scatter(X[epm1],Y[epm1],
               c=e1_cmap, s=0.5)
    e2=ax.scatter(X[epm2],Y[epm2],
               c=e2_cmap, s=0.5)  
    e3=ax.scatter(X[epm3],Y[epm3],
               c=e3_cmap, s=0.5)
    e4=ax.scatter(X[epm4],Y[epm4],
               c=e4_cmap, s=0.5) 
    e5=ax.scatter(X[epm5],Y[epm5],
               c=e5_cmap, s=0.5)  
    e6=ax.scatter(X[epm6],Y[epm6],
               c=e6_cmap, s=0.5)  
    e7=ax.scatter(X[epm7],Y[epm7],
               c=e7_cmap, s=0.5)  
    e8=ax.scatter(X[epm8],Y[epm8],
               c=e8_cmap, s=0.5)  
    eneg=ax.scatter(X[epmneg],Y[epmneg],
               c=eneg_cmap, s=0.5) 
    
    ax.grid(False)
        
    return ax

def run(partname, l_mouse_name, l_date, l_path_cdata, l_rotation, electrodetype,epmtype,flag_idxused, exptype="reg", sorter="tdc"):
    l_sample_rate_video=np.zeros(len(l_mouse_name))
    excelcolidx = find_excel_col_idx(partname)
    thresh = calc_thresh(excelcolidx, l_mouse_name, exptype)

    if not os.path.isdir('mouseDataNew'):
        os.mkdir('mouseDataNew')
    #print(flag_idxused)
    for i in range(len(l_mouse_name)):
        print(i)
        if flag_idxused == 1:
            if sorter == "kilo":
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_ALLIDX_kilo.npz'):
                    print('ALLIDX data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
            else:
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_ALLIDX.npz'):
                    print('ALLIDX data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
            
        elif flag_idxused == 2:
            if sorter == "kilo":
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_tailback_kilo.npz'):
                    print('Back-Tail data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
            else:
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_tailback.npz'):
                    print('Back-Tail data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
        else:   
            if sorter == "kilo":
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_check_kilo.npz'):
                    print('data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
            else:
                if os.path.exists('mouseDataNew/'+l_mouse_name[i]+'_'+l_date[i]+'_neuralBehdataCorrected_check.npz'):
                    print('data already present')
                    l_mouse_name[i] = np.nan
                    l_date[i] = np.nan
                    l_rotation[i] = np.nan
                    l_path_cdata[i] = np.nan
                    thresh[i] = np.nan
    

    l_mouse_name = [x for x in l_mouse_name if x == x]
    l_date =[x for x in l_date if x == x]
    l_rotation =[x for x in l_rotation if x == x]
    l_path_cdata = [x for x in l_path_cdata if x == x]
    thresh = [x for x in thresh if x == x]

    if len(l_mouse_name) == 0:
        #raise Exception('Data is already generated for the mice!')
        return 0

    for ii in range(len(l_mouse_name)):

        #edit
        media_file = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\"+l_mouse_name[ii]+"_EPM.avi"
        
        if electrodetype == 0 and epmtype == 0: # silicon electrodes
            media_file = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_SiliconProbe\\"+l_mouse_name[ii]+"_EPM.avi"
       
        if electrodetype == 1 and epmtype == 0: # InNphy electrodes
            media_file = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_InNPhy16\\"+l_mouse_name[ii]+"_EPM.avi"
       
        if exptype == "D1":
            media_file = r"Y:\Ephys_in_vivo\01_RAW_DATA\2_In_Nphy\D1-Ephys\Data\\"+str(l_mouse_name[ii])+"\\"+l_date[ii]+"\\"+l_mouse_name[ii]+"_EPM.avi"


        #l_path_cdata = "Y:\\Ephys_in_vivo\\01_RAW_DATA\2_In_Nphy\\"+l_mouse_name[ii]+"\\"+l_date[ii]+"\\"+l_mouse_name[ii]+"_2023-04-14_12-06-58_EPM\Record Node 101\experiment1\recording1\continuous\Rhythm_FPGA-100.0\continuous.dat"
        #l_sample_rate_video = ffmpeg.probe(media_file)["avg_frame_rate"]

        media = cv2.VideoCapture(media_file)
        l_sample_rate_video[ii] = media.get(5)
        print('sampling rate '+str(l_sample_rate_video[ii]))

    if exptype=="D1":
        l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center = get_coordinates_EPM_d1(l_mouse_name, l_date, l_rotation)
    else:
        l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center = get_coordinates_EPM(l_mouse_name, l_date, l_rotation)
        
    if flag_idxused !=1:
        print('widen')
        l_x_open, l_y_open = widen_EPM(l_x_open, l_y_open, thresh)

    if flag_idxused == 2:
        l_arm_idx, l_X_bef, l_Y_bef = get_arm_idx_backtail(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname, electrodetype,epmtype, exptype)
    else:
        l_arm_idx, l_X_bef, l_Y_bef = get_arm_idx(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname, electrodetype,epmtype, exptype)
    l_X, l_Y = behData_removeOutliers((l_X_bef), (l_Y_bef),l_x_closed, l_y_closed, l_x_open, l_y_open)


    for ii in range(len(l_mouse_name)):
        #print('hmm')
        fig, axs = plt.subplots(1,2)
        #print(ii)
        #axs[0].plot(np.squeeze(np.array(l_X)), np.squeeze(np.array(l_Y)))
        if l_rotation[ii] == 0:
            #print(3)
            #print((l_X_bef == l_X).maximum())
            axs[0] = plotArm_idx(np.squeeze(np.array(l_X_bef[ii])), np.squeeze(np.array(l_Y_bef[ii])), np.squeeze(l_arm_idx[ii]), axs[0], False)
            axs[0].scatter(l_x_center[ii], l_y_center[ii], c='r')
            axs[0].scatter(l_x_center[ii], np.flip(l_y_center[ii]), c='r')
            axs[0].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[0]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[0].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[1]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[0].scatter(np.squeeze(l_x_open[ii])[0]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[0].scatter(np.squeeze(l_x_open[ii])[1]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[0].set_title('without removing outliers')

            axs[1]= plotArm_idx(np.squeeze(np.array(l_X[ii])), np.squeeze(np.array(l_Y[ii])), np.squeeze(l_arm_idx[ii]), axs[1], False)
            axs[1].scatter(l_x_center[ii], l_y_center[ii], c='r')
            axs[1].scatter(l_x_center[ii], np.flip(l_y_center[ii]), c='r')
            axs[1].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[0]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[1].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[1]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[1].scatter(np.squeeze(l_x_open[ii])[0]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[1].scatter(np.squeeze(l_x_open[ii])[1]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[1].set_title('with removing outliers')
            
        else: 
            #print(2)
            axs[0] = plotArm_idx(np.squeeze(np.array(l_X_bef[ii])), np.squeeze(np.array(l_Y_bef[ii])), np.squeeze(l_arm_idx[ii]), axs[0], False)
            axs[0].scatter(l_x_center[ii], l_y_center[ii], c='r')
            axs[0].scatter(l_x_center[ii], np.flip(l_y_center[ii]), c='r')
            axs[0].scatter(l_x_open[ii], np.squeeze(l_y_open[ii])[0]*np.ones(np.squeeze(l_x_open[ii]).size), c='r')
            axs[0].scatter(l_x_open[ii], np.squeeze(l_y_open[ii])[1]*np.ones(np.squeeze(l_x_open[ii]).size), c='r')
            axs[0].scatter(np.squeeze(l_x_closed[ii])[0]*np.ones(np.squeeze(l_y_closed[ii]).size), l_y_closed[ii], c='r' )
            axs[0].scatter(np.squeeze(l_x_closed[ii])[1]*np.ones(np.squeeze(l_y_closed[ii]).size), l_y_closed[ii], c='r' )
            axs[0].set_title('without removing outliers')

            axs[1]= plotArm_idx(np.squeeze(np.array(l_X[ii])), np.squeeze(np.array(l_Y[ii])), np.squeeze(l_arm_idx[ii]), axs[1], False)
            axs[1].scatter(l_x_center[ii], l_y_center[ii], c='r')
            axs[1].scatter(l_x_center[ii], np.flip(l_y_center[ii]), c='r')
            axs[1].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[0]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[1].scatter(l_x_closed[ii], np.squeeze(l_y_closed[ii])[1]*np.ones(np.squeeze(l_x_closed[ii]).size), c='r')
            axs[1].scatter(np.squeeze(l_x_open[ii])[0]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[1].scatter(np.squeeze(l_x_open[ii])[1]*np.ones(np.squeeze(l_y_open[ii]).size), l_y_open[ii], c='r' )
            axs[1].set_title('with removing outliers')
        
        if not os.path.isdir('PreprocessingVisualisations'):
            os.mkdir('PreprocessingVisualisations')

        if flag_idxused == 1:
            plt.savefig("PreprocessingVisualisations//"+l_mouse_name[ii]+"_"+l_date[ii]+"_"+"EPM_EventSelection_ALLIDX.png")
        else:
            plt.savefig("PreprocessingVisualisations//"+l_mouse_name[ii]+"_"+l_date[ii]+"_"+partname+"EPM_EventSelection.png")
        
        plt.show()

    l_X, l_Y, origin = data_centering(l_X, l_Y, l_x_center, l_y_center)
    l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected = select_events(l_arm_idx, l_X, l_Y, l_sample_rate_video, l_mouse_name, l_date, exp_type=exptype)

    l_data, l_beh, l_data_25, cluster_id_list = get_electrophysiological_data(l_mouse_name, l_date, l_rotation, l_sample_rate_video, l_path_cdata, l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected, exptype, sorter)   
    if flag_idxused == 2:
        l_beh_lowdim = distanceGen_tailback(l_beh)
    else:
        l_beh_lowdim = distanceGen(l_beh)
    
    if flag_idxused ==0:

        for ii in range(len(l_mouse_name)):
            if sorter == "kilo":
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_check_kilo.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii],l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_check.pkl', 'wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)
            else:
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_check.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii],l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_check.pkl', 'wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)
    
    if flag_idxused==2: 

        for ii in range(len(l_mouse_name)):
            if sorter == "kilo":
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX_tailback_kilo.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii],l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX_tailback.pkl','wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)
            else:
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX_tailback.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii], l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX_tailback.pkl','wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)
    
    if flag_idxused==1:
        for ii in range(len(l_mouse_name)):
            if sorter == "kilo":
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX_kilo.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii], l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX.pkl','wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)
            else:
                np.savez('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX.npz', l_data=l_data[ii], l_beh= l_beh[ii],l_beh_lowdim=l_beh_lowdim[ii],l_x_open=l_x_open[ii], l_y_open=l_y_open[ii], l_x_closed=l_x_closed[ii], l_y_closed=l_y_closed[ii], l_x_center=l_x_center[ii], l_y_center=l_y_center[ii], origin = origin[ii], l_data_25=l_data_25[ii])
                with open('mouseDataNew/'+l_mouse_name[ii]+'_'+l_date[ii]+'_neuralBehdataCorrected_ALLIDX.pkl','wb') as f:
                    pkl.dump([l_data[ii], l_beh[ii],l_beh_lowdim[ii],l_x_open[ii], l_y_open[ii], l_x_closed[ii], l_y_closed[ii], l_x_center[ii], l_y_center[ii], origin[ii],l_event_idx[ii],l_duration_idx[ii],l_starting_idx[ii], l_end_idx[ii], l_data_25[ii]], f)


if __name__ == '__main__':

    # flag idx 1: separate headdips as idx=-1
    # flag idx 0: include headdips with OA
    partname = "snout"
    l_mouse_name = ["F2491",]
    l_date = ["20230414"]
    l_path_cdata = ["Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F2491\\20230414\\F2491_2023-04-14_12-27-12_EPM\\Record Node 101\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat"]
    l_rotation = [0]
    epmType = 1 # 0: modified / 1: classical
    electrodetype = 1 # 0: silicon / 1:inNphy
    run(partname, l_mouse_name, l_date, l_path_cdata, l_rotation, electrodetype,epmType,flag_idxused=0, exptype="reg", sorter='kilo')
