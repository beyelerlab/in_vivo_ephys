"""
Population dynamic analysis
Version: 20221122
"""

from numpy import load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'figure.max_open_warning': 0}) # remove figure warnings

#Parameters for the plots
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 12*ratio
marker_size = 15/2
marker_edge_width = 3/2
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}
sns.set(style='ticks')

#Create lists for different parameters:
#1. Mouse ID (Sex_Number):
# l_mouse_name = ['F27', 'F28', 'M29', 'M30', 'M31']
# l_mouse_name = ['F27', 'F28', 'M29', 'M30', 'M31']
l_mouse_name =['F2491']#['F27', 'M29'] ##['F27', 'M29']

# l_sample_rate_video_in_Hz = [17.61, 17.34, 16.26, 17.41, 18.11]
l_sample_rate_video = [18.898]#[17.61, 16.26]#, 18.245]#

#2. Folders corresponding to the recording dates (1 per mouse):
# l_date = ['20221103', '20221103', '20221103', '20221103', '20221103']
l_date =['20230414']# ['20221103', '20221103'] #,'20230414'] #

#3. Paths from each mouse directing to the continuous.dat file (from openephys):
# l_path_cdata = ['C:\\beyeler-yoni\\data\\raw\\F27\\20221103\\F27_2022-11-03_15-11-37_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat',
#                 'C:\\beyeler-yoni\\data\\raw\\F28\\20221103\\F28_2022-11-03_15-36-24_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat',
#                 'C:\\beyeler-yoni\\data\\raw\\M29\\20221103\\M29_2022-11-03_16-04-53_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat',
#                 'C:\\beyeler-yoni\\data\\raw\\M30\\20221103\\M30_2022-11-03_16-57-11_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat',
#                 'C:\\beyeler-yoni\\data\\raw\\M31\\20221103\\M31_2022-11-03_16-32-56_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat']
l_path_cdata = ["Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F2491\\20230414\\F2491_2023-04-14_12-27-12_EPM\\Record Node 101\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat"]#, "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F2491\\20230512\\F2491_2023-05-12_10-52-35_EPM\\Record Node 101\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat"] #["Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F27\\20221103\\F27_2022-11-03_15-11-37_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat",
               #["Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F27\\20221103\\F27_2022-11-03_15-11-37_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat",
                #"Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\M29\\20221103\\M29_2022-11-03_16-04-53_EPM\\Record Node 104\\experiment1\\recording1\\continuous\\Rhythm_FPGA-100.0\\continuous.dat"]
                #
#4. Rotation of the EPM (0: closed arms facing the walls / 1: open arms facing the walls)
# l_rotation = [1, 1, 1, 1, 1]
l_rotation = [0]#[1, 1]
#Create a function that gives the coordinates of the EPM based on 3 variables (Mouse ID, Date, Rotation):


def find_excel_col_idx(partname):
    parts = ['snout', 'leftear', 'rightear', 'shoulder', 'spine1', 'spine2', 'spine3', 'tailbase', 'tail1', 'tail2', 'tailend']
    # 0 - x, 1 - y, 2 - likelihood
    idx = parts.index(partname)
    excelcolidx = idx*3
    #print(excelcolidx)
    return excelcolidx


def get_coordinates_EPM(l_mouse_name, l_date, l_rotation):
    l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center = [], [], [], [], [], [] #Create empty lists for the 3 regions of interests in X and Y coordinates
    for mouse_idx in range(len(l_mouse_name)): #mouse_idx = mouse position in the EPM
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        # path = "C:\\beyeler-yoni\\data\\analysis\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_userlandmarks.npz" #this .npz file is given by the ephys script and gives the coordinates of each zones of the EPM
        #path = "Y:\\Ephys_in_vivo\\02_ANALYSIS\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_userlandmarks.npz" #this .npz file is given by the ephys script and gives the coordinates of each zones of the EPM
        path = "S:\___DATA\in_vivo_ePhys\BackUp_ePhys_Mice\EPM_ephys_npz_files\\" + mouse_name + "_EPM_userlandmarks.npz" 
        #this path is changed because the server is full(made on 27th June2024)
        data = load(path)
        lst = data.files #data.files = location of each zones in the EPM
        f_rotation = l_rotation[mouse_idx]
        #Initiliaze the coordinates for each zones. They are 4 coordinates in X and Y for each zones (because we delineate the zone in 4 points):
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

        # print(l_x_open)
        # print(l_y_open)
        # print(l_x_closed)
        # print(l_y_closed)
        # print(l_x_center)
        # print(l_y_center)
    return l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center

def get_arm_idxx(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center):
    l_arm_idx = []
    l_X, l_Y = [], [] #create two lists, one l_X for the coordinates in X and one l_Y for the coordinates in Y
    for mouse_idx in range(len(l_mouse_name)):
        l_arm_idx_indiv = []
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        path = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_bonsai.csv"
        mouseX, mouseY, mouseAngle, mouseMajorAxisLength, mouseMinorAxisLength, mouseArea, optoPeriod = np.genfromtxt(path, dtype=float, skip_header=1, unpack=True) #load the bonsai.csv file = coordinates of the mouse (mouseX, mouseY)
        l_X.append(mouseX) # might have nans??
        l_Y.append(mouseY)

        for i in range(len(mouseX)): # the length of the coordinates data in X and Y
            x = mouseX[i] # read the x location of the mouse
            y = mouseY[i] # read the y location of the mouse
            arm_idx = 5 #there are 5 idx (0: open arm 1 /1: open arm 2/ 2: closed arm 1 /3: closed arm 2 /4: center)
            ref_x_open = l_x_open[mouse_idx]
            ref_y_open = l_y_open[mouse_idx]
            ref_x_closed = l_x_closed[mouse_idx]
            ref_y_closed = l_y_closed[mouse_idx]
            ref_x_center = l_x_center[mouse_idx]
            ref_y_center = l_y_center[mouse_idx]
            if l_rotation[mouse_idx] == 0:
                #l_arm_idx_indiv.append(arm_idx) # why?
                if ref_y_closed[0] < y < ref_y_closed[1]: # why +5?
                    if ref_x_closed[0] < x < ref_x_closed[1] :
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_x_closed[2] < x < ref_x_closed[3]:
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
                    arm_idx = 4
                l_arm_idx_indiv.append(arm_idx)

            else:
                # check whether it is in the open arm
                if ref_y_open[0] - 5 < y < ref_y_open[1] + 5: # why +5?
                    if ref_x_open[0] < x < ref_x_open[1]:
                        arm_idx = 0
                    elif ref_x_open[2] < x < ref_x_open[3]:
                        arm_idx = 1
                    else:
                        pass
                # check whether it is in the closed arm
                if ref_x_closed[0] < x < ref_x_closed[1]:
                    if ref_y_closed[0] < y < ref_y_closed[1]:
                        arm_idx = 2
                    elif ref_y_closed[2] < y < ref_y_closed[3]:
                        arm_idx = 3
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    arm_idx = 4

                l_arm_idx_indiv.append(arm_idx)
        l_arm_idx.append(l_arm_idx_indiv)
    return l_arm_idx, l_X, l_Y


def find_excel_col_idx(partname):
    parts = ['snout', 'leftear', 'rightear', 'shoulder', 'spine1', 'spine2', 'spine3', 'tailbase', 'tail1', 'tail2', 'tailend']
    # 0 - x, 1 - y, 2 - likelihood
    idx = parts.index(partname)
    excelcolidx = idx*3
    #print(excelcolidx)
    return excelcolidx

def get_beh_dataframe(excelcolidx, l_mousename, l_date, electrodetype,epmtype):
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

        path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videos\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        if l_date[mouse_idx] == "20230512" and name == "F2491":
            path_behavior = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\F2491\\20230512\\dlc\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
        if electrodetype == 0 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_SiliconProbe\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        if electrodetype == 1 and epmtype == 0: # silicon electrodes
            path_behavior = "S:\\_Tanmai\\Python Scripts\\DLC\\ePhys_in_vivo_demo-tdhanireddy-2023-08-02\\videosModifiedEPM_InNPhy16\\"+name+"_EPMDLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000_filtered.csv"
        
        
        behavior_data = pd.read_csv(path_behavior, delimiter=',',low_memory=False)
        #pprint.pprint(behavior_data)
        beh_x = behavior_data[prefix][2:]
        #print(beh_x)

        prefix = "DLC_resnet50_ePhys_in_vivo_demoAug2shuffle1_100000"+"."+str(excelcolidx+1) # y-data
        #print(prefix)
        beh_y = behavior_data[prefix][2:]
        #print(beh_y)
        mouseX[name] = beh_x
        mouseY[name] = beh_y

    mouseX = pd.DataFrame(mouseX).to_numpy(dtype = 'float')
    mouseY = pd.DataFrame(mouseY).to_numpy(dtype = 'float')
    return mouseX, mouseY

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

        for i in range(len(mouseX)): # the length of the coordinates data in X and Y
            x = mouseX[i] # read the x location of the mouse
            y = mouseY[i] # read the y location of the mouse
            arm_idx = 5 #there are 5 idx (0: open arm 1 /1: open arm 2/ 2: closed arm 1 /3: closed arm 2 /4: center)
            ref_x_open = l_x_open[mouse_idx]
            ref_y_open = l_y_open[mouse_idx]
            ref_x_closed = l_x_closed[mouse_idx]
            ref_y_closed = l_y_closed[mouse_idx]
            ref_x_center = l_x_center[mouse_idx]
            ref_y_center = l_y_center[mouse_idx]
            if l_rotation[mouse_idx] == 0:
                #l_arm_idx_indiv.append(arm_idx) # why?
                if ref_y_closed[0] < y < ref_y_closed[1]: # why +5?
                    if ref_x_closed[0] < x < ref_x_closed[1] :
                        arm_idx = 2
                        isOAprev = 0
                    elif ref_x_closed[2] < x < ref_x_closed[3]:
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
                    arm_idx = 4
                l_arm_idx_indiv.append(arm_idx)

            else:
                # check whether it is in the open arm
                if ref_y_open[0] - 5 < y < ref_y_open[1] + 5: # why +5?
                    if ref_x_open[0] < x < ref_x_open[1]:
                        arm_idx = 0
                    elif ref_x_open[2] < x < ref_x_open[3]:
                        arm_idx = 1
                    else:
                        pass
                # check whether it is in the closed arm
                if ref_x_closed[0] < x < ref_x_closed[1]:
                    if ref_y_closed[0] < y < ref_y_closed[1]:
                        arm_idx = 2
                    elif ref_y_closed[2] < y < ref_y_closed[3]:
                        arm_idx = 3
                    else:
                        pass
                # check whether it is in the center
                if (ref_x_center[0] < x < ref_x_center[1]) and (ref_y_center[0] < y < ref_y_center[1]):
                    arm_idx = 4

                l_arm_idx_indiv.append(arm_idx)
        l_arm_idx.append(l_arm_idx_indiv)
    return l_arm_idx, l_X, l_Y

    return l_arm_idx, l_X, l_Y



#create a function that selects the events: mouse entering the open arms and closed arms
def select_events(l_arm_idx, l_X, l_Y, l_sample_rate_video):
    threshold = 40 #minimal threshold when the mouse stays in the defined region for >=1 s (1s = 20/20Hz)
    l_event_idx, l_duration_idx, l_starting_idx, l_end_idx= [], [], [], []
    l_X_selected, l_Y_selected = [], []
    for mouse_idx in range(len(l_arm_idx)):
        l_event_idx_indiv = []
        l_duration_idx_indiv = [] #indiv = for each mouse
        l_starting_idx_indiv = []
        l_end_idx_indiv = []
        l_X_indiv, l_Y_indiv = [], []
        mouse_arm_idx = l_arm_idx[mouse_idx]
        duration_counter = 1 #?
        for i in range(len(mouse_arm_idx)-1):
            if mouse_arm_idx[i+1] == mouse_arm_idx[i]:
                duration_counter+=1 #how many times we have the same arm idx in the same event
            else:
                if (duration_counter >= threshold) and (mouse_arm_idx[i]!=4) and (mouse_arm_idx[i]!=5): #do not consider idx4: center and idx:5 outside the EPM
                    l_starting_idx_indiv.append(i-duration_counter+1)
                    l_end_idx_indiv.append(i) #mouse_arm idx(i) different from mouse_arm idx(i+1): new idx (that's why we have start and end idx)
                    l_event_idx_indiv.append(mouse_arm_idx[i])
                    l_duration_idx_indiv.append(duration_counter)
                    l_X_indiv.append(l_X[mouse_idx][i-duration_counter+1:i+1]) #all of the X coordinates for the indiv selected events
                    l_Y_indiv.append(l_Y[mouse_idx][i-duration_counter+1:i+1]) #all of the Ycoordinates for the indiv selected events
                else:
                    pass
                duration_counter = 1

        l_event_idx.append(l_event_idx_indiv)
        l_duration_idx.append(l_duration_idx_indiv)
        l_starting_idx.append(l_starting_idx_indiv)
        l_end_idx.append(l_end_idx_indiv)
        l_X_selected.append(l_X_indiv)
        l_Y_selected.append(l_Y_indiv)
    return l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected

def check_csv_separator(filepath):
    separators = [',',';']
    with open(filepath) as f:
        line = f.readline()
        for sep in separators:
            if sep in line:
                return sep
    return None

def get_one_ch_data(path_cdata, total_ch_number, ch_id):
    d = np.memmap(path_cdata, dtype=np.int16) # why did we use a memmap?
    d = np.reshape(d, (int(d.shape[0] / total_ch_number), total_ch_number))
    return d[:, ch_id]

def analog_2_events(signal):
    print(np.max(signal[0:30000 * 60]))
    th = ((np.max(signal[0:30000 * 60]).astype(float) + np.min(signal[0:30000 * 60]).astype(float)) / 2 )# what
    signal = signal > th
    signal = signal.astype(int)
    diff_sig = np.diff(signal)
    onsets = np.where(diff_sig == 1)[0]
    offsets = np.where(diff_sig == -1)[0]
    print('Onset '+str(onsets[0]))
    return {'onsets': onsets, 'offsets': offsets}

def get_video_frames_onsets(path_cdata, total_ch_number=27, ch_id=19): #ch19 is the ADC1, corresponding to the TTLs of the behavior camera
    d = get_one_ch_data(path_cdata, total_ch_number=total_ch_number, ch_id=ch_id)
    TTL_camera = analog_2_events(d)
    onset_ephys_idx = TTL_camera['onsets'][0]
    offset_ephys_idx = TTL_camera['offsets'][0] # why not
    return onset_ephys_idx

def get_electrophysiological_data(l_mouse_name, l_date, l_rotation, l_sample_rate_video, l_path_cdata, l_event_idx, l_duration_idx, l_starting_idx, l_end_idx):
    l_data_PCA = []
    l_arm_idx_PCA = []
    for mouse_idx in range(len(l_mouse_name)):
        l_t_starting_video = np.asarray(l_starting_idx[mouse_idx]) / l_sample_rate_video[mouse_idx]
        l_t_end_video = np.asarray(l_end_idx[mouse_idx]) / l_sample_rate_video[mouse_idx]
        date = l_date[mouse_idx]
        mouse_name = l_mouse_name[mouse_idx]
        path_ephys = "Y:\\Ephys_in_vivo\\01_RAW_DATA\\2_In_Nphy\\" + mouse_name + "\\" + date + "\\" + mouse_name + "_EPM_spikesorting.csv"
        path_cdata = l_path_cdata[mouse_idx]
        onset_ephys_idx = get_video_frames_onsets(path_cdata, total_ch_number=27,
                                                  ch_id=19)  # to synchronize with the behavior

        separator = check_csv_separator(path_ephys)
        if separator is None:
            print('Program failed to load spikes because csv separator is not defined')
        else:
            idx, cluster_id = np.genfromtxt(path_ephys, dtype=int, skip_header=1, unpack=True,delimiter=separator)

            l_neuron_id = list(set(cluster_id))
            n_neurons = len(l_neuron_id)
            n_bins_total = 0  # ? - total num of bins across all events
            l_t_ephys = (idx - onset_ephys_idx) / 30000  # beginning of the ephys recording
            bin_size = round(1 / l_sample_rate_video[mouse_idx],6)  # 50ms, arbitrary fixed (because, frame rate is 20 Hz, so 1/20=50ms)
            for i in range(len(l_t_starting_video)):
                t_starting_bin = round(l_t_starting_video[i], 6)
                t_end_bin = round(l_t_end_video[i], 6)
                bin_vec = np.around(t_starting_bin - round(bin_size / 2, 6) + np.arange(0, round((t_end_bin + round(bin_size / 2, 6) - (t_starting_bin - round(bin_size / 2, 6))) / bin_size) + 1) * bin_size,6)  # ?? - time bins from starttime - timebin/2 to endtime+timebin/2

                for j in range(len(l_neuron_id)):
                    neuron_id = l_neuron_id[j]
                    neuron_idx_spike = np.where(cluster_id == neuron_id)[0]
                    spike_timing_neuron = l_t_ephys[neuron_idx_spike]
                    hist_neuron = np.histogram(spike_timing_neuron, bin_vec)[0]
                    if j == 0:
                        n_bins_total += len(hist_neuron)  # ?? - this is to calculate total number of bins across all events

            data_PCA = np.zeros((n_neurons, n_bins_total)) * np.nan
            arm_idx_PCA = np.zeros(n_bins_total) * np.nan
            n_bins_idx = 0
            for i in range(len(l_t_starting_video)):
                t_starting_bin = round(l_t_starting_video[i], 6)
                t_end_bin = round(l_t_end_video[i], 6)
                bin_vec = np.around(t_starting_bin - round(bin_size / 2, 6) + np.arange(0, round((t_end_bin + round(bin_size / 2, 6) - (t_starting_bin - round(bin_size / 2, 6))) / bin_size) + 1) * bin_size, 6)  # ??

                for j in range(len(l_neuron_id)):  # j = idx for access one neuron
                    neuron_id = l_neuron_id[j]
                    neuron_idx_spike = np.where(cluster_id == neuron_id)[0]
                    spike_timing_neuron = l_t_ephys[neuron_idx_spike]
                    hist_neuron = np.histogram(spike_timing_neuron, bin_vec)[0]
                    data_PCA[j, n_bins_idx:n_bins_idx + len(hist_neuron)] = hist_neuron
                    if j == len(l_neuron_id) - 1:
                        n_bins_idx += len(hist_neuron)  # swap
                        arm_idx_PCA[n_bins_idx:n_bins_idx + len(hist_neuron)] = l_event_idx[mouse_idx][i]  # swap these two lines of code ???
            plt.plot(data_PCA[0, :])
            plt.show()
            l_data_PCA.append(data_PCA)
            l_arm_idx_PCA.append(arm_idx_PCA)
    return l_data_PCA, l_arm_idx_PCA

def PCA_analysis(l_data_PCA):
    k = 3  # nb of dimensions
    l_projected_data = []
    print(len(l_data_PCA))
    for m in range(len(l_data_PCA)):
        data_mat = l_data_PCA[m]
        cov_mat = np.cov(data_mat)
        eigval, eigvec = np.linalg.eig(cov_mat)
        idx = eigval.argsort()[::-1] # ??
        eigvec_sorted = eigvec[:, idx]
        PC = eigvec_sorted[:, :k]
        projected_data = np.matmul((np.transpose(data_mat)), PC)
        l_projected_data.append(projected_data)
        print("Dimension of projected data:")
        print(projected_data.shape)
    return l_projected_data

def get_color_idx(l_event_idx, l_X_selected, l_Y_selected, l_arm_idx_PCA, l_x_open, l_y_open, l_x_closed, l_y_closed):
    n_color_bins = 256
    l_cmap_idx = []
    l_color_idx = []
    for i in range(len(l_event_idx)): # number of mouse
        l_cmap_idx_indiv = []
        l_color_idx_indiv = []
        n_data_point = 0
        rot = 1
        if len(l_x_open[0]) == 2:
            rot = 0
        for j in range(len(l_event_idx[i])): # number of events
            n_data_point += len(l_X_selected[i][j])

            for m in range(len(l_X_selected[i][j])): #m = coordinates
                if l_event_idx[i][j] == 0: # l_event_idx[i][j] # the arm idx of each event
                    l_cmap_idx_indiv.append(0)
                    if rot == 1:
                        l_color_idx_indiv.append(int(n_color_bins - np.ceil((l_X_selected[i][j][m] - l_x_open[i][0])/(l_x_open[i][1] - l_x_open[i][0]) * n_color_bins)))
                    else: 
                        l_color_idx_indiv.append(int(n_color_bins - np.ceil((l_Y_selected[i][j][m] - l_y_open[i][0])/(l_y_open[i][1] - l_y_open[i][0]) * n_color_bins)))

                    # print(int(n_color_bins - np.ceil((l_X_selected[i][j][m] - l_x_open[i][0])/(l_x_open[i][1] - l_x_open[i][0]) * n_color_bins)))
                elif l_event_idx[i][j] == 1:
                    l_cmap_idx_indiv.append(1)
                    if rot == 1:
                        l_color_idx_indiv.append(int(np.floor((l_X_selected[i][j][m] - l_x_open[i][2]) / (l_x_open[i][3] - l_x_open[i][2]) * n_color_bins)))
                    else:
                        l_color_idx_indiv.append(int(np.floor((l_Y_selected[i][j][m] - l_y_open[i][2]) / (l_y_open[i][3] - l_y_open[i][2]) * n_color_bins)))
                    # print(int(np.floor((l_X_selected[i][j][m] - l_x_open[i][2]) / (l_x_open[i][3] - l_x_open[i][2]) * n_color_bins)))
                elif l_event_idx[i][j] == 2:
                    l_cmap_idx_indiv.append(2)
                    if rot == 1:
                        l_color_idx_indiv.append(int(n_color_bins - np.ceil((l_Y_selected[i][j][m] - l_y_closed[i][0]) / (l_y_closed[i][1] - l_y_closed[i][0]) * n_color_bins)))
                    else:
                        l_color_idx_indiv.append(int(n_color_bins - np.ceil((l_X_selected[i][j][m] - l_x_closed[i][0]) / (l_x_closed[i][1] - l_x_closed[i][0]) * n_color_bins)))
                    
                    # print(int(n_color_bins - np.ceil((l_Y_selected[i][j][m] - l_y_closed[i][0]) / (l_y_closed[i][1] - l_y_closed[i][0]) * n_color_bins)))
                elif l_event_idx[i][j] == 3:
                    l_cmap_idx_indiv.append(3)
                    if rot ==1:
                        l_color_idx_indiv.append(int(np.floor((l_Y_selected[i][j][m] - l_y_closed[i][2]) / (l_y_closed[i][3] - l_y_closed[i][2]) * n_color_bins)))
                    else:
                        l_color_idx_indiv.append(int(np.floor((l_X_selected[i][j][m] - l_x_closed[i][2]) / (l_x_closed[i][3] - l_x_closed[i][2]) * n_color_bins)))
                else:
                    pass
        l_cmap_idx.append(l_cmap_idx_indiv)
        # print("Dimension of the color index")
        # print(len(l_color_idx_indiv))
        l_color_idx.append(l_color_idx_indiv)
    return l_cmap_idx, l_color_idx

def plotting_trajectory(l_projected_data, l_mouse_name, l_X_selected, l_Y_selected, l_event_idx, l_duration_idx, l_x_open, l_y_open, l_x_closed, l_y_closed, l_arm_idx_PCA, l_cmap_idx, l_color_idx):
    # define color masps
    orig_cmaps = []
    cmap = plt.cm.Oranges
    orig_cmaps.append(cmap)
    cmaplist_oranges = np.array([cmap(i) for i in range(cmap.N)])
    cmap = plt.cm.Reds
    orig_cmaps.append(cmap)
    cmaplist_reds = np.array([cmap(i) for i in range(cmap.N)])
    cmap = plt.cm.Blues
    orig_cmaps.append(cmap)
    cmaplist_blues = np.array([cmap(i) for i in range(cmap.N)])
    cmap = plt.cm.Greens
    orig_cmaps.append(cmap)
    cmaplist_greens = np.array([cmap(i) for i in range(cmap.N)])
    colors = [cmaplist_oranges, cmaplist_reds, cmaplist_blues, cmaplist_greens]

    for m in range(len(l_projected_data)):
        projected_data_indiv = l_projected_data[m]
        # plt.figure(figsize=(figure_len, figure_width))
        # ax = plt.axes(projection='3d')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(True)
        # ax.spines['left'].set_visible(True)
        
        k = 0
        for n in range(len(l_event_idx[m])):
            s_files = ''
            plt.figure(figsize=(figure_len, figure_width))
            ax = plt.axes(projection='3d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            for ii in range(l_duration_idx[m][n]-1):
                if l_event_idx[m][n] == 0: # use specific color depending on the arm index
                    ax.plot(projected_data_indiv[k+ii:k+ii+2, 0], projected_data_indiv[k+ii:k+ii+2, 1], projected_data_indiv[k+ii:k+ii+2, 2], color=colors[l_cmap_idx[m][k+ii]][l_color_idx[m][k+ii]], linewidth=plot_line_width)
                    s_files = 'open_arm_1'
                elif l_event_idx[m][n] == 1:
                    ax.plot(projected_data_indiv[k+ii:k+ii+2, 0], projected_data_indiv[k+ii:k+ii+2, 1],projected_data_indiv[k+ii:k+ii+2, 2], color=colors[l_cmap_idx[m][k+ii]][l_color_idx[m][k+ii]],linewidth=plot_line_width)
                    s_files = 'open_arm_2'
                elif l_event_idx[m][n] == 2:
                    ax.plot(projected_data_indiv[k+ii:k+ii+2, 0], projected_data_indiv[k+ii:k+ii+2, 1],projected_data_indiv[k+ii:k+ii+2, 2], color=colors[l_cmap_idx[m][k+ii]][l_color_idx[m][k+ii]],linewidth=plot_line_width)
                    s_files = 'closed_arm_1'
                elif l_event_idx[m][n] == 3:
                    ax.plot(projected_data_indiv[k+ii:k+ii+2, 0], projected_data_indiv[k+ii:k+ii+2, 1],projected_data_indiv[k+ii:k+ii+2, 2], color=colors[l_cmap_idx[m][k+ii]][l_color_idx[m][k+ii]],linewidth=plot_line_width)
                    s_files = 'closed_arm_2'
                else:
                    pass

            k += l_duration_idx[m][n]
            ax.set_xlabel('PC1', fontsize=font_size_1/ratio, **hfont)
            ax.set_ylabel('PC2', fontsize=font_size_1/ratio, **hfont)
            ax.set_zlabel('PC3', fontsize=font_size_1/ratio, **hfont)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([-20, 20])
            plt.savefig('svg/' + l_mouse_name[m] + '_event_' + str(n) + '_' + str(s_files) + '_DLC.svg')
            plt.savefig('png/' + l_mouse_name[m] + '_event_' + str(n) + '_' + str(s_files) + '_DLC.png')


def plotting_3D_points(l_projected_data, l_mouse_name, l_X_selected, l_Y_selected, l_event_idx, l_x_open, l_y_open,l_x_closed, l_y_closed, l_arm_idx_PCA, l_cmap_idx, l_color_idx):
    for m in range(len(l_projected_data)):
        projected_data_indiv = l_projected_data[m]
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.axes(projection='3d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        n_arm_idx_PCA_0 = len(np.where(l_arm_idx_PCA[m] == 0)[0]) #nb of times the location for open arms 1 occurs
        n_arm_idx_PCA_1 = len(np.where(l_arm_idx_PCA[m] == 1)[0])
        n_arm_idx_PCA_2 = len(np.where(l_arm_idx_PCA[m] == 2)[0])
        n_arm_idx_PCA_3 = len(np.where(l_arm_idx_PCA[m] == 3)[0])
        counter_2 = 0
        counter_3 = 0
        for ii in range(projected_data_indiv.shape[0]):
            if l_arm_idx_PCA[m][ii] == 0:
                ax.plot(projected_data_indiv[ii, 0], projected_data_indiv[ii, 1], projected_data_indiv[ii, 2],  linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width, markersize=marker_size, markeredgecolor='black', markerfacecolor='red', alpha=0.5)
            elif l_arm_idx_PCA[m][ii] == 1:
                ax.plot(projected_data_indiv[ii, 0], projected_data_indiv[ii, 1], projected_data_indiv[ii, 2], linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width, markersize=marker_size, markeredgecolor='black', markerfacecolor='green', alpha=0.5)
            elif l_arm_idx_PCA[m][ii] == 2:
                if counter_2 < 500: #500 first events!
                    ax.plot(projected_data_indiv[ii, 0], projected_data_indiv[ii, 1], projected_data_indiv[ii, 2],  linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width, markersize=marker_size, markeredgecolor='black', markerfacecolor='blue', alpha=0.5)
                    counter_2 +=1
            elif l_arm_idx_PCA[m][ii] == 3:
                if counter_3 < 500:
                    ax.plot(projected_data_indiv[ii, 0], projected_data_indiv[ii, 1], projected_data_indiv[ii, 2], linestyle='none', marker='o', fillstyle='full', markeredgewidth=marker_edge_width, markersize=marker_size, markeredgecolor='black', markerfacecolor='yellow', alpha=0.5)
                    counter_3 +=1
            else:
                pass
        plt.savefig(l_mouse_name[m] + '_DLC.png')


# def plotting_test():
#     # define color masps
#     orig_cmaps = []
#     cmap = plt.cm.Greys
#     orig_cmaps.append(cmap)
#     cmaplist_greys = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     cmap = plt.cm.Purples
#     orig_cmaps.append(cmap)
#     cmaplist_purples = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     cmap = plt.cm.Blues
#     orig_cmaps.append(cmap)
#     cmaplist_blues = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     cmap = plt.cm.Greens
#     orig_cmaps.append(cmap)
#     cmaplist_greens = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     cmap = plt.cm.Oranges
#     orig_cmaps.append(cmap)
#     cmaplist_oranges = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     cmap = plt.cm.Reds
#     orig_cmaps.append(cmap)
#     cmaplist_reds = np.array([cmap(i) for i in range(cmap.N)])[25:]
#     colors = [cmaplist_greys, cmaplist_purples, cmaplist_blues, cmaplist_greens, cmaplist_oranges, cmaplist_reds]
#     print(colors[0][230])

#     plt.figure(figsize=(figure_len, figure_width))
#     ax = plt.axes(projection='3d')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     # print("here")
#     # print(projected_data_indiv.shape)
#     # print(len(l_cmap_idx[m]))
#     # print(len(l_color_idx[m]))
#
#     for ii in range(projected_data_indiv.shape[0] - 2):
#         ax.plot(projected_data_indiv[ii:ii + 2, 0], projected_data_indiv[ii:ii + 2, 1],
#                 projected_data_indiv[ii:ii + 2, 2], color=colors[l_cmap_idx[m][ii]][l_color_idx[m][ii]],
#                 linewidth=plot_line_width)

if __name__ == "__main__":
    l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center = get_coordinates_EPM(l_mouse_name, l_date, l_rotation)
    l_arm_idx, l_X, l_Y = get_arm_idx(l_mouse_name, l_date, l_rotation, l_x_open, l_y_open, l_x_closed, l_y_closed, l_x_center, l_y_center, partname = 'snout', electrodetype=1, epmtype=1)
    l_event_idx, l_duration_idx, l_starting_idx, l_end_idx, l_X_selected, l_Y_selected = select_events(l_arm_idx, l_X, l_Y, l_sample_rate_video)
    l_data_PCA, l_arm_idx_PCA = get_electrophysiological_data(l_mouse_name, l_date, l_rotation, l_sample_rate_video, l_path_cdata, l_event_idx, l_duration_idx, l_starting_idx, l_end_idx)
    l_projected_data = PCA_analysis(l_data_PCA)
    l_cmap_idx, l_color_idx = get_color_idx(l_event_idx, l_X_selected, l_Y_selected, l_arm_idx_PCA, l_x_open, l_y_open, l_x_closed, l_y_closed)
    plotting_trajectory(l_projected_data, l_mouse_name, l_X_selected, l_Y_selected, l_event_idx, l_duration_idx, l_x_open, l_y_open, l_x_closed, l_y_closed, l_arm_idx_PCA, l_cmap_idx, l_color_idx)
    plotting_3D_points(l_projected_data, l_mouse_name, l_X_selected, l_Y_selected, l_event_idx, l_x_open, l_y_open, l_x_closed, l_y_closed, l_arm_idx_PCA, l_cmap_idx, l_color_idx)
    