import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import math
import csv

gait_duration = "Number of time steps for one gait duration"
subject_name = "Name_of_subject"
col_list = ["UserID", "Class", "Signal"]
df = pd.read_csv(subject_name+".csv", usecols=col_list)
All = df
sample_LW = []
sample_RA = []
sample_RD = []

for i in range(len(df)):
    if df["Class"][i] == "LW":
        sample_LW.append(df["Signal"][i])
    if df["Class"][i] == "RA":
        sample_RA.append(df["Signal"][i])
    if df["Class"][i] == "RD":
        sample_RD.append(df["Signal"][i])

print("Subject Name: ", subject_name)
print("Gait Duration: ", gait_duration)
num_LW_signals = int(len(sample_LW)/gait_duration)
print("Total LW: ", num_LW_signals)
num_RA_signals = int(len(sample_RA)/gait_duration)
print("Total RA: ", num_RA_signals)
num_RD_signals = int(len(sample_RD)/gait_duration)
print("Total RD: ", num_RD_signals)
total_signals = num_LW_signals + num_RA_signals + num_RD_signals
print("Total signals:", total_signals)

subsequence_LW = np.zeros((num_LW_signals,gait_duration))
subsequence_RA = np.zeros((num_RA_signals,gait_duration))
subsequence_RD = np.zeros((num_RD_signals,gait_duration))

interval_LW = np.zeros((num_LW_signals, 3, int(gait_duration/3)))
interval_RA = np.zeros((num_RA_signals, 3, int(gait_duration/3)))
interval_RD = np.zeros((num_RD_signals, 3, int(gait_duration/3)))

tic1 = int(gait_duration/3)
tic2 = tic1*2
tic3 = tic1*3

j = 0
for i in range(0, (num_LW_signals)):
    subsequence_LW[i] = sample_LW[j:j+gait_duration]
    interval_LW[i,0,:] = subsequence_LW[i][0:tic1]
    interval_LW[i,1,:] = subsequence_LW[i][tic1:tic2]
    interval_LW[i,2,:] = subsequence_LW[i][tic2:tic3]
    j = j+gait_duration
j = 0
for i in range(0, (num_RA_signals)):
    subsequence_RA[i] = sample_RA[j:j+gait_duration]
    interval_RA[i,0,:] = subsequence_RA[i][0:tic1]
    interval_RA[i,1,:] = subsequence_RA[i][tic1:tic2]
    interval_RA[i,2,:] = subsequence_RA[i][tic2:tic3]
    j = j+gait_duration
j = 0
for i in range(0, (num_RD_signals)): 
    subsequence_RD[i] = sample_RD[j:j+gait_duration]
    interval_RD[i,0,:] = subsequence_RD[i][0:tic1]
    interval_RD[i,1,:] = subsequence_RD[i][tic1:tic2]
    interval_RD[i,2,:] = subsequence_RD[i][tic2:tic3]
    j = j+gait_duration

grad0_LW = np.zeros((num_LW_signals,int(gait_duration/3)))
ang0_LW = np.zeros((num_LW_signals,int(gait_duration/3)))
grad1_LW = np.zeros((num_LW_signals,int(gait_duration/3)))
ang1_LW = np.zeros((num_LW_signals,int(gait_duration/3)))
grad2_LW = np.zeros((num_LW_signals,int(gait_duration/3)))
ang2_LW = np.zeros((num_LW_signals,int(gait_duration/3)))

grad0_RA = np.zeros((num_RA_signals,int(gait_duration/3)))
ang0_RA = np.zeros((num_RA_signals,int(gait_duration/3)))
grad1_RA = np.zeros((num_RA_signals,int(gait_duration/3)))
ang1_RA = np.zeros((num_RA_signals,int(gait_duration/3)))
grad2_RA = np.zeros((num_RA_signals,int(gait_duration/3)))
ang2_RA = np.zeros((num_RA_signals,int(gait_duration/3)))

grad0_RD = np.zeros((num_RD_signals,int(gait_duration/3)))
ang0_RD = np.zeros((num_RD_signals,int(gait_duration/3)))
grad1_RD = np.zeros((num_RD_signals,int(gait_duration/3)))
ang1_RD = np.zeros((num_RD_signals,int(gait_duration/3)))
grad2_RD = np.zeros((num_RD_signals,int(gait_duration/3)))
ang2_RD = np.zeros((num_RD_signals,int(gait_duration/3)))

csvrecord = "/home/ajey/Desktop/FGAHOG1D/HOG_"+subject_name+".csv"

for i in range(len(interval_LW)):
    grad0_LW[i] = np.gradient(interval_LW[i,0])
    grad1_LW[i] = np.gradient(interval_LW[i,1])
    grad2_LW[i] = np.gradient(interval_LW[i,2])
    ang0_LW[i] = np.arctan(grad0_LW[i])
    ang1_LW[i] = np.arctan(grad1_LW[i])
    ang2_LW[i] = np.arctan(grad2_LW[i])

for i in range(len(interval_RA)):
    grad0_RA[i] = np.gradient(interval_RA[i,0])
    grad1_RA[i] = np.gradient(interval_RA[i,1])
    grad2_RA[i] = np.gradient(interval_RA[i,2])
    ang0_RA[i] = np.arctan(grad0_RA[i])
    ang1_RA[i] = np.arctan(grad1_RA[i])
    ang2_RA[i] = np.arctan(grad2_RA[i])
        
for i in range(len(interval_RD)):
    grad0_RD[i] = np.gradient(interval_RD[i,0])
    grad1_RD[i] = np.gradient(interval_RD[i,1])
    grad2_RD[i] = np.gradient(interval_RD[i,2])
    ang0_RD[i] = np.arctan(grad0_RD[i])
    ang1_RD[i] = np.arctan(grad1_RD[i])
    ang2_RD[i] = np.arctan(grad2_RD[i])
    
for i in range(num_LW_signals):
    bins0, values = np.histogram(grad0_LW[i], 8)
    bins1, values = np.histogram(grad1_LW[i], 8)
    bins2, values = np.histogram(grad2_LW[i], 8)
    bins0 = list(bins0)
    bins1 = list(bins1)
    bins2 = list(bins2)
    # print(bins0+bins1+bins2)
    with open(csvrecord, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, lineterminator="\n")
                csvwriter.writerow(bins0+bins1+bins2+["LW"]+["\n"])

print("Successfully prepared HOG1D data for LW signals!")

for i in range(num_RA_signals):
    bins0, values = np.histogram(grad0_RA[i], 8)
    bins1, values = np.histogram(grad1_RA[i], 8)
    bins2, values = np.histogram(grad2_RA[i], 8)
    bins0 = list(bins0)
    bins1 = list(bins1)
    bins2 = list(bins2)
    # print(bins0+bins1+bins2)
    with open(csvrecord, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, lineterminator="\n")
                csvwriter.writerow(bins0+bins1+bins2+["RA"]+["\n"])

print("Successfully prepared HOG1D data for RA signals!")

for i in range(num_RD_signals):
    bins0, values = np.histogram(grad0_RD[i], 8)
    bins1, values = np.histogram(grad1_RD[i], 8)
    bins2, values = np.histogram(grad2_RD[i], 8)
    bins0 = list(bins0)
    bins1 = list(bins1)
    bins2 = list(bins2)
    # print(bins0+bins1+bins2)
    with open(csvrecord, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, lineterminator="\n")
                csvwriter.writerow(bins0+bins1+bins2+["RD"]+["\n"])

print("Successfully prepared HOG1D data for RD signals!")

def draw_line(x,y,angle,length, c):
    terminus_x = x + length * math.cos(angle)
    terminus_y = y + length * math.sin(angle)
    line = ax0.plot([x, terminus_x],[y,terminus_y], c)
    ar = FancyArrowPatch ((x,y),(terminus_x,terminus_y),color= c,
                              arrowstyle='->', mutation_scale=15)
    ax0.add_patch(ar)
    return line

index = np.arange(gait_duration)
fig = plt.figure(figsize=(15, 8))
ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
line0 = ax0.plot(subsequence_LW[0], 'black', label = 'One LW Gait Cycle')

for i in range(tic1):
    line1 = draw_line(index[i],interval_LW[0,0,i],ang0_LW[0,i],0.2, 'r')
ax1 = plt.subplot2grid((2, 4), (1, 0))
ax1.hist(grad0_LW[0], 8, color = 'r')
    
for i in range(tic1, tic2):
    line2  = draw_line(index[i],interval_LW[0,1,i-tic1],ang1_LW[0,i-tic1],0.2, 'b')
ax2 = plt.subplot2grid((2, 4), (1, 1))
ax2.hist(grad1_LW[0], 8, color = 'b')
ax2.set_yticklabels([])
ax2.set_yticks([])

for i in range(tic2, tic3):
    line3 = draw_line(index[i],interval_LW[0,2,i-tic2],ang2_LW[0,i-tic2],0.2, 'g')
ax3 = plt.subplot2grid((2, 4), (1, 2))
ax3.hist(grad2_LW[0], 8, color = 'g')
ax3.set_yticklabels([])
ax3.set_yticks([])

ax0.legend()
ax0.set_title("One LW Gait cycle split into three equal intervals")
ax0.set_ylabel("Normalised magnitude")
ax0.set_xlabel("sample index")
ax1.legend(['HOG in I1'],loc = 'upper right')
ax2.legend(['HOG in I2'],loc = 'upper right')
ax3.legend(['HOG in I1'],loc = 'upper right')
plt.subplots_adjust(wspace=0)
plt.show()

index = np.arange(gait_duration)
fig = plt.figure(figsize=(15, 8))
ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
line0 = ax0.plot(subsequence_RA[0], 'black', label = 'One RA Gait Cycle')

for i in range(tic1):
    line1 = draw_line(index[i],interval_RA[0,0,i],ang0_RA[0,i],0.2, 'r')
ax1 = plt.subplot2grid((2, 4), (1, 0))
ax1.hist(grad0_RA[0], 8,color = 'r')
    
for i in range(tic1,tic2):
    line2  = draw_line(index[i],interval_RA[0,1,i-tic1],ang1_RA[0,i-tic1],0.2, 'b')
ax2 = plt.subplot2grid((2, 4), (1, 1))
ax2.hist(grad1_RA[0], 8,color = 'b')
ax2.set_yticklabels([])
ax2.set_yticks([])

for i in range(tic2,tic3):
    line3 = draw_line(index[i],interval_RA[0,2,i-tic2],ang2_RA[0,i-tic2],0.2, 'g')
ax3 = plt.subplot2grid((2, 4), (1, 2))
ax3.hist(grad0_RA[0], 8,color = 'g')
ax3.set_yticklabels([])
ax3.set_yticks([])

ax0.legend()
ax0.set_title("One RA Gait cycle split into three equal intervals")
ax0.set_ylabel("Normalised magnitude")
ax0.set_xlabel("sample index")
ax1.legend(['HOG in I1'],loc = 'upper right')
ax2.legend(['HOG in I2'],loc = 'upper right')
ax3.legend(['HOG in I1'],loc = 'upper right')
plt.subplots_adjust(wspace=0)
plt.show()

index = np.arange(gait_duration)
fig = plt.figure(figsize=(15, 8))
ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
line0 = ax0.plot(subsequence_RD[0], 'black', label = 'One RD Gait Cycle')

for i in range(tic1):
    line1 = draw_line(index[i],interval_RD[0,0,i],ang0_RD[0,i],0.2, 'r')
ax1 = plt.subplot2grid((2, 4), (1, 0))
ax1.hist(grad0_RD[0], 8, color = 'r')
    
for i in range(tic1,tic2):
    line2  = draw_line(index[i],interval_RD[0,1,i-tic1],ang1_RD[0,i-tic1],0.2, 'b')
ax2 = plt.subplot2grid((2, 4), (1, 1))
ax2.hist(grad1_RD[0], 8,color = 'b')
ax2.set_yticklabels([])
ax2.set_yticks([])

for i in range(tic2,tic3):
    line3 = draw_line(index[i],interval_RD[0,2,i-tic2],ang2_RD[0,i-tic2],0.2, 'g')
ax3 = plt.subplot2grid((2, 4), (1, 2))
ax3.hist(grad0_RD[0], 8,color = 'g')
ax3.set_yticklabels([])
ax3.set_yticks([])

ax0.legend()
ax0.set_title("One RD Gait cycle split into three equal intervals")
ax0.set_ylabel("Normalised magnitude")
ax0.set_xlabel("sample index")
ax1.legend(['HOG in I1'],loc = 'upper right')
ax2.legend(['HOG in I2'],loc = 'upper right')
ax3.legend(['HOG in I1'],loc = 'upper right')
plt.subplots_adjust(wspace=0)
plt.show()