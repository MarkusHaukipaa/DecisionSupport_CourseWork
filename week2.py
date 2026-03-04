"""
Created on Thu Oct 23 13:26:02 2025

@author: mhauk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sc
import neurokit2 as nk2
import matplotlib.dates as mdates 


df1 = pd.read_csv(r"C:\Users\mhauk\Device_support\Materials\E_2\annotation1.AN1",
                 skiprows=2,
                 names=["Time","VType","Variable","Value","Status"],
                 index_col=False)

df_info = pd.read_csv(r"C:\Users\mhauk\Device_support\Materials\E_2\info.txt",
                      names = ['Fs','Start_date','Start_time','Label',
                               'Dimension','Coef1','Coef2','Coef3','Coef4',
                               'Nmb_chans','N'])

CI = df1.loc[df1['Variable'] == 30001000]    
Tp = df1.loc[df1['Variable'] == 400]
PCWP = df1.loc[df1['Variable'] == 800]  
      
print(f"CI shape: {CI.shape}, \n"
      f"Tp shape: {Tp.shape} \n"
      f"PCWP shape: {PCWP.shape}")
#The shape of Tp is greatly larger than CI or PCWP
"""CI shape: (26, 5), 
Tp shape: (699, 5) 
PCWP shape: (25, 5)"""
# The shape may differ because different frequency used for the measurements
# The frequency can be accounted for the computational load of CI, the
# invasivess of PCWP or just generally the fluctuation of Tp is greater or more
# of importance

merged_v = pd.concat([CI,Tp,PCWP])



merged_v["Cardiac_Failure"] = (((merged_v['Value'] < 2) & (merged_v['Variable'] ==30001000)) |
                     ((merged_v['Value'] < 32.5) & (merged_v['Variable'] ==400)) |
                     ((merged_v['Value'] > 10) & (merged_v['Variable'] == 800 )))

merged_v = merged_v.sort_values("Cardiac_Failure")
merged_v = merged_v.drop_duplicates(subset = "Time",keep = 'last')

merged_v = merged_v.sort_values("Time")
fig, ax = plt.subplots()


date_time_str = str(df_info['Start_date'][0][3:5]+ df_info['Start_date'][0][6:8])

merged_v['Time_formatted']=date_time_str+merged_v['Time'].astype(str) 
merged_v['Time_formatted']=pd.to_datetime((merged_v['Time_formatted']),
                                          format='%m%y%d%H%M%S') 
date_form = mdates.DateFormatter("%H:%M") 

ax.plot(merged_v['Time_formatted'], merged_v["Cardiac_Failure"], label = "AN1")
ax.xaxis.set_major_formatter(date_form)
ax.set_yticks([0,1])
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Detected Cardiac Failure: AN1', weight='bold')

df2 = pd.read_csv(r"C:\Users\mhauk\Device_support\Materials\E_2\annotation2.AN2",
                 skiprows=2,
                 names=["Time","VType","Variable","Value","Status"],
                 index_col=False)



merged_v2 = []
merged_v2 =  df2.loc[(df2['Variable'] >= 15001114) &
                     (df2['Variable'] <= 15001119)]

merged_v2 = merged_v2.sort_values("Time")
#ignore dublicates

merged_v3 =  df2.loc[((df2['Variable'] >= 82) &
                     (df2['Variable'] <= 93) )|
                     (df2['Variable'] == 116) |
                    (df2['Variable'] == 117)  ]
merged_v3['Time_formatted']=date_time_str+merged_v3['Time'].astype(str) 
merged_v3['Time_formatted']=pd.to_datetime((merged_v3['Time_formatted']),
                                          format='%m%y%d%H%M%S')
print(merged_v3.to_string(index = False))

merged_v2 = merged_v2.drop_duplicates(subset = "Time",keep = 'last')
df2["cardiac failure"] = ((df2["Time"] == 9100000) | 
                                 (df2["Time"] == 9112000)|
                                 (df2["Time"] == 9120000)|
                                  (df2["Time"] == 10050300))

df2 = df2.drop_duplicates(subset = "Time",keep = 'last')

df2['Time_formatted']=date_time_str+df2['Time'].astype(str) 
df2['Time_formatted']=pd.to_datetime((df2['Time_formatted']),
                                          format='%m%y%d%H%M%S') 

ax.plot(df2["Time_formatted"], df2["cardiac failure"], 
        color = 'r',label = "AN2")
ax.xaxis.set_major_formatter(date_form)
ax.set_yticks([0,1])
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.title('Detected Cardiac Failure: AN1 & AN2', weight='bold')

plt.show()
