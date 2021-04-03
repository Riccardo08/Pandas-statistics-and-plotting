import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime as dt
import os
import numpy as np
import matplotlib.gridspec as gridspec

# TODO Exercise 0
df = pd.read_csv('ASHRAE90.1_OfficeSmall_STD2016_NewYork.csv')
df=df.iloc[288:,:]
Date = df['Date/Time'].str.split(' ', expand=True)
Date.rename(columns={0:'nullo',1:'date',2:'null', 3:'time'},inplace=True)
Date['time'] = Date['time'].replace(to_replace='24:00:00', value= '0:00:00')
data = Date['date']+' '+Date['time']
data = pd.to_datetime(data, format = '%m/%d %H:%M:%S')

df['month']= data.apply(lambda x: x.month)
df['day']=data.apply(lambda x: x.day)
df['hour']=data.apply(lambda x: x.hour)
df['dn']=data.apply(lambda x: x.day_name())
df['mn']=data.apply(lambda x: x.month_name())
df['data']=Date.date

#TODO Exercise 1

matplotlib.rcParams['font.size']=14
# matplotlib.rcParams['figure.facecolor'] = '#00000000'
#TODO 1.Grafico Outdoor e indoor temperature con legenda e titolo

df.plot(x='data', y=['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)', 'CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'], lw=0.5)
plt.ylabel('Temperature [°C]')
plt.legend(['Outdoor Air Drybulb Temperature', 'Zone Mean Air Temperature'], loc = 'lower right')
plt.title('Outdoor and Indoor Temperature')
plt.show()

matplotlib.rcParams['font.size']=12
sns.set_style("whitegrid")
#TODO 2.Grafico relative humidity, cambia il fontsize di tutto (tick, label, titolo, legenda)

df.plot(x='data', y=['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)', 'CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'], lw=0.5)
plt.ylabel('Humidity [%]')
plt.legend(['Outdoor Air Humidity', 'Zone Air Humidity'], loc = 'lower right')
plt.title('Outdoor and Indoor Relative Humidity')
plt.show()

matplotlib.rcParams['font.size']=10
sns.set_style("darkgrid")
# plt.style.use(['dark_background'])
#TODO 3.Grafico mean radiant temperature

df.plot(x='data', y=['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'], lw=0.5)
plt.ylabel('Temperature [°C]')
plt.legend('', frameon=False)
plt.title('Mean Radiant Temperature')
plt.show()

#TODO 4.Grafico thermal comfort PPD/PMV fai un subplot mettendo le immagini una sotto l’altra
fig, axes = plt.subplots(2,figsize=(14,10))
fig.suptitle('Zone PMV and PPD')
fig.subplots_adjust(hspace=0.5)
# plt.tight_layout(pad=2)
df.plot(ax=axes[0], x='data', y='CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)', label='PMV')
axes[0].legend(loc='upper right')
axes[0].set_title('PMV')
df.plot(ax=axes[1], x='data', y='CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)', label='PPD', c ='r')
axes[1].legend(loc='upper right')
axes[1].set_title('PPD [%]')
plt.show()



#TODO 5.Grafico cooling coil, inserisci dei rettangoli blu se sei nella cooling season
# (cioè quando il cooling coil è elevato) e rosso nella heating season. I rettangoli devono
# essere semitrasparenti.
df.plot(x='data', y=['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)', 'PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'])
plt.legend(['Cooling coil cooling rate [W]', 'Heating coil heating rate [W]'])
plt.title('Heat pump Heating and Cooling coil rate')
plt.show()

#TODO 6.Grafico People occupant count
df.plot(x='data', y = 'CORE_ZN:Zone People Occupant Count [](TimeStep)', lw=0.3, figsize=(30,8))
plt.suptitle('Zone People Occupant Count')
plt.legend('', frameon=False)
plt.show()


#TODO 7.Grafico diffuse solar radiation/direct solar radiation:
# fai subplot verticali con legenda per ognuno e leggenda globale
fig, axes =plt.subplots(2, figsize=(6,8))
fig.subplots_adjust(hspace=0.3)
fig.suptitle('Diffuse and direct solar radiation rate per area [W/m2]')
p1=df.plot(ax=axes[0], x='data', y='Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)', lw=0.5, label='Diffuse solar radiation')
#axes[0].legend(['Diffuse'])
p2=df.plot(ax=axes[1], x='data', y='Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)', c='r', lw=0.5, label='Direct solar radiation')
#axes[1].legend(['Direct'])
fig.legend([p1, p2], labels=['Diffuse', 'Direct'], ncol=2, loc='lower center')
#plt.legend( handles=[p1,p2], loc='center right', ncol=2, shadow=True, title='Legend', fancybox='True')
plt.show()

#TODO Exercise 2:usando il pacchetto pandas estrarre statistiche base dalle colonne del dataset
# (media,mediana, massimo, minimo, deviazione standard).Ripetere questo passaggio per 4 aggregazioni:
# 1.	Tutto l’anno
# 2.	per mese
# 3.	per giorno della settimana
# 4.	per ora del giorno

# PER MESE
Tout_m = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Tout_m['mean+std']=Tout_m['mean']+Tout_m['std']
Tout_m['mean-std']=Tout_m['mean']-Tout_m['std']
Tout_m['month']=range(1,13)
Hu_out_m = df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_out_m['mean+std']=Hu_out_m['mean']+Hu_out_m['std']
Hu_out_m['mean-std']=Hu_out_m['mean']-Hu_out_m['std']
Hu_out_m['month']=range(1,13)
Wind_m = df['Environment:Site Wind Speed [m/s](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Wind_m['mean+std']=Wind_m['mean']+Wind_m['std']
Wind_m['mean-std']=Wind_m['mean']-Wind_m['std']
Wind_m['month']=range(1,13)
Diff_rad_m = df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Diff_rad_m['mean+std']=Diff_rad_m['mean']+Diff_rad_m['std']
Diff_rad_m['mean-std']=Diff_rad_m['mean']-Diff_rad_m['std']
Diff_rad_m['month']=range(1,13)
Dir_rad_m = df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Dir_rad_m['mean+std']=Dir_rad_m['mean']+Dir_rad_m['std']
Dir_rad_m['mean-std']=Dir_rad_m['mean']-Dir_rad_m['std']
Dir_rad_m['month']=range(1,13)
People_m = df['CORE_ZN:Zone People Occupant Count [](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
People_m['mean+std']=People_m['mean']+People_m['std']
People_m['mean-std']=People_m['mean']-People_m['std']
People_m['month']=range(1,13)
Rad_temp_m = df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Rad_temp_m['mean+std']=Rad_temp_m['mean']+Rad_temp_m['std']
Rad_temp_m['mean-std']=Rad_temp_m['mean']-Rad_temp_m['std']
Rad_temp_m['month']=range(1,13)
Tint_m = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Tint_m['mean+std']=Tint_m['mean']+Tint_m['std']
Tint_m['mean-std']=Tint_m['mean']-Tint_m['std']
Tint_m['month']=range(1,13)
Hu_int_m = df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_int_m['mean+std']=Hu_int_m['mean']+Hu_int_m['std']
Hu_int_m['mean-std']=Hu_int_m['mean']-Hu_int_m['std']
Hu_int_m['month']=range(1,13)
PMV_m = df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
PMV_m['mean+std']=PMV_m['mean']+PMV_m['std']
PMV_m['mean-std']=PMV_m['mean']-PMV_m['std']
PMV_m['month']=range(1,13)
PPD_m = df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
PPD_m['mean+std']=PPD_m['mean']+PPD_m['std']
PPD_m['mean-std']=PPD_m['mean']-PPD_m['std']
PPD_m['month']=range(1,13)
Cooling_m = df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Cooling_m['mean+std']=Cooling_m['mean']+Cooling_m['std']
Cooling_m['mean-std']=Cooling_m['mean']-Cooling_m['std']
Cooling_m['month']=range(1,13)
Heating_m = df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Heating_m['mean+std']=Heating_m['mean']+Heating_m['std']
Heating_m['mean-std']=Heating_m['mean']-Heating_m['std']
Heating_m['month']=range(1,13)
Supp_m = df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).agg(['max', 'min', 'median', 'std', 'mean'])
Supp_m['mean+std']=Supp_m['mean']+Supp_m['std']
Supp_m['mean-std']=Supp_m['mean']-Supp_m['std']
Supp_m['month']=range(1,13)

# PER GIORNO DELLA SETTIMANA
Tout_d = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Tout_d['mean+std']=Tout_d['mean']+Tout_d['std']
Tout_d['mean-std']=Tout_d['mean']-Tout_d['std']
Tout_d['day']=range(1,31+1)
Hu_out_d = df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_out_d['mean+std']=Hu_out_d['mean']+Hu_out_d['std']
Hu_out_d['mean-std']=Hu_out_d['mean']-Hu_out_d['std']
Hu_out_d['day']=range(1,31+1)
Wind_d = df['Environment:Site Wind Speed [m/s](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Wind_d['mean+std']=Wind_d['mean']+Wind_d['std']
Wind_d['mean-std']=Wind_d['mean']-Wind_d['std']
Wind_d['day']=range(1,31+1)
Diff_rad_d = df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Diff_rad_d['mean+std']=Diff_rad_d['mean']+Diff_rad_d['std']
Diff_rad_d['mean-std']=Diff_rad_d['mean']-Diff_rad_d['std']
Diff_rad_d['day']=range(1,31+1)
Dir_rad_d = df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Dir_rad_d['mean+std']=Dir_rad_d['mean']+Dir_rad_d['std']
Dir_rad_d['mean-std']=Dir_rad_d['mean']-Dir_rad_d['std']
Dir_rad_d['day']=range(1,31+1)
People_d = df['CORE_ZN:Zone People Occupant Count [](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
People_d['mean+std']=People_d['mean']+People_d['std']
People_d['mean-std']=People_d['mean']-People_d['std']
People_d['day']=range(1,31+1)
Rad_temp_d = df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Rad_temp_d['mean+std']=Rad_temp_d['mean']+Rad_temp_d['std']
Rad_temp_d['mean-std']=Rad_temp_d['mean']-Rad_temp_d['std']
Rad_temp_d['day']=range(1,31+1)
Tint_d = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Tint_d['mean+std']=Tint_d['mean']+Tint_d['std']
Tint_d['mean-std']=Tint_d['mean']-Tint_d['std']
Tint_d['day']=range(1,31+1)
Hu_int_d = df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_int_d['mean+std']=Hu_int_d['mean']+Hu_int_d['std']
Hu_int_d['mean-std']=Hu_int_d['mean']-Hu_int_d['std']
Hu_int_d['day']=range(1,31+1)
PMV_d = df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
PMV_d['mean+std']=PMV_d['mean']+PMV_d['std']
PMV_d['mean-std']=PMV_d['mean']-PMV_d['std']
PMV_d['day']=range(1,31+1)
PPD_d = df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
PPD_d['mean+std']=PPD_d['mean']+PPD_d['std']
PPD_d['mean-std']=PPD_d['mean']-PPD_d['std']
PPD_d['day']=range(1,31+1)
Cooling_d = df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Cooling_d['mean+std']=Cooling_d['mean']+Cooling_d['std']
Cooling_d['mean-std']=Cooling_d['mean']-Cooling_d['std']
Cooling_d['day']=range(1,31+1)
Heating_d = df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Heating_d['mean+std']=Heating_d['mean']+Heating_d['std']
Heating_d['mean-std']=Heating_d['mean']-Heating_d['std']
Heating_d['day']=range(1,31+1)
Supp_d = df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).agg(['max', 'min', 'median', 'std', 'mean'])
Supp_d['mean+std']=Supp_d['mean']+Supp_d['std']
Supp_d['mean-std']=Supp_d['mean']-Supp_d['std']
Supp_d['day']=range(1,31+1)

# PER ORA DEL GIORNO
Tout_h = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Tout_h['mean+std']=Tout_h['mean']+Tout_h['std']
Tout_h['mean-std']=Tout_h['mean']-Tout_h['std']
Tout_h['hour']=range(0,24)
Hu_out_h = df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_out_h['mean+std']=Hu_out_h['mean']+Hu_out_h['std']
Hu_out_h['mean-std']=Hu_out_h['mean']-Hu_out_h['std']
Hu_out_h['hour']=range(0,24)
Wind_h = df['Environment:Site Wind Speed [m/s](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Wind_h['mean+std']=Wind_h['mean']+Wind_h['std']
Wind_h['mean-std']=Wind_h['mean']-Wind_h['std']
Wind_h['hour']=range(0,24)
Diff_rad_h = df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Diff_rad_h['mean+std']=Diff_rad_h['mean']+Diff_rad_h['std']
Diff_rad_h['mean-std']=Diff_rad_h['mean']-Diff_rad_h['std']
Diff_rad_h['hour']=range(0,24)
Dir_rad_h = df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Dir_rad_h['mean+std']=Dir_rad_h['mean']+Dir_rad_h['std']
Dir_rad_h['mean-std']=Dir_rad_h['mean']-Dir_rad_h['std']
Dir_rad_h['hour']=range(0,24)
People_h = df['CORE_ZN:Zone People Occupant Count [](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
People_h['mean+std']=People_h['mean']+People_h['std']
People_h['mean-std']=People_h['mean']-People_h['std']
People_h['hour']=range(0,24)
Rad_temp_h = df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Rad_temp_h['mean+std']=Rad_temp_h['mean']+Rad_temp_h['std']
Rad_temp_h['mean-std']=Rad_temp_h['mean']-Rad_temp_h['std']
Rad_temp_h['hour']=range(0,24)
Tint_h = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Tint_h['mean+std']=Tint_h['mean']+Tint_h['std']
Tint_h['mean-std']=Tint_h['mean']-Tint_h['std']
Tint_h['hour']=range(0,24)
Hu_int_h = df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Hu_int_h['mean+std']=Hu_int_h['mean']+Hu_int_h['std']
Hu_int_h['mean-std']=Hu_int_h['mean']-Hu_int_h['std']
Hu_int_h['hour']=range(0,24)
PMV_h = df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
PMV_h['mean+std']=PMV_h['mean']+PMV_h['std']
PMV_h['mean-std']=PMV_h['mean']-PMV_h['std']
PMV_h['hour']=range(0,24)
PPD_h = df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
PPD_h['mean+std']=PPD_h['mean']+PPD_h['std']
PPD_h['mean-std']=PPD_h['mean']-PPD_h['std']
PPD_h['hour']=range(0,24)
Cooling_h = df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Cooling_h['mean+std']=Cooling_h['mean']+Cooling_h['std']
Cooling_h['mean-std']=Cooling_h['mean']-Cooling_h['std']
Cooling_h['hour']=range(0,24)
Heating_h = df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Heating_h['mean+std']=Heating_h['mean']+Heating_h['std']
Heating_h['mean-std']=Heating_h['mean']-Heating_h['std']
Heating_h['hour']=range(0,24)
Supp_h = df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).agg(['max', 'min', 'median', 'std', 'mean'])
Supp_h['mean+std']=Supp_h['mean']+Supp_h['std']
Supp_h['mean-std']=Supp_h['mean']-Supp_h['std']
Supp_h['hour']=range(0,24)



#TODO Exercise 3:plotta grafici delle statistiche calcolate prima
# (esempio: istogramma della temperatura media per mese durante l’anno)
# PER MESE
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(20,25))
fig.suptitle('Statistics variables per month', fontsize=50)
fig.subplots_adjust(hspace=0.5)

Tout_m.iloc[:, :3].plot(ax=axes[0,0])
Tout_m['mean'].plot(kind ='bar', ax=axes[0,0])
axes[0,0].set_title('Site Outdoor Air Drybulb Temperature [C]')
axes[0,0].fill_between(Tout_m['month'], Tout_m['mean+std'], Tout_m['mean-std'], facecolor='blue', alpha=0.5)

Tint_m.iloc[:, :3].plot(ax=axes[1,0])
Tint_m['mean'].plot(kind ='bar', ax=axes[1,0])
axes[1,0].set_title('Zone Mean Air Temperature [C]')
axes[1,0].fill_between(Tint_m['month'], Tint_m['mean+std'], Tint_m['mean-std'], facecolor='blue', alpha=0.5)

Hu_out_m.iloc[:, :3].plot(ax=axes[0,1])
Hu_out_m['mean'].plot(kind ='bar', ax=axes[0,1])
axes[0,1].set_title('Site outdoor Air Relative Humidity [%]')
axes[0,1].fill_between(Hu_out_m['month'], Hu_out_m['mean+std'], Hu_out_m['mean-std'], facecolor='blue', alpha=0.5)

Hu_int_m.iloc[:, :3].plot(ax=axes[1,1])
Hu_int_m['mean'].plot(kind ='bar', ax=axes[1,1])
axes[1,1].set_title('Zone Air Relative Humidity [%]')
axes[1,1].fill_between(Hu_int_m['month'], Hu_int_m['mean+std'], Hu_int_m['mean-std'], facecolor='blue', alpha=0.5)

Diff_rad_m.iloc[:, :3].plot(ax=axes[2,0])
Diff_rad_m['mean'].plot(kind ='bar', ax=axes[2,0])
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
axes[2,0].fill_between(Diff_rad_m['month'], Diff_rad_m['mean+std'], Diff_rad_m['mean-std'], facecolor='blue', alpha=0.5)

Dir_rad_m.iloc[:, :3].plot(ax=axes[2,1])
Dir_rad_m['mean'].plot(kind ='bar', ax=axes[2,1])
axes[2,1].set_title('Direct Solar Radiation Rate per Area [W/m2]')
axes[2,1].fill_between(Dir_rad_m['month'], Dir_rad_m['mean+std'], Dir_rad_m['mean-std'], facecolor='blue', alpha=0.5)

PMV_m.iloc[:, :3].plot(ax=axes[3,0])
PMV_m['mean'].plot(kind ='bar', ax=axes[3,0])
axes[3,0].set_title('PMV')
axes[3,0].fill_between(PMV_m['month'], PMV_m['mean+std'], PMV_m['mean-std'], facecolor='blue', alpha=0.5)

PPD_m.iloc[:, :3].plot(ax=axes[3,1])
PPD_m['mean'].plot(kind ='bar', ax=axes[3,1])
axes[3,1].set_title('PPD')
axes[3,1].fill_between(PPD_m['month'], PPD_m['mean+std'], PPD_m['mean-std'], facecolor='blue', alpha=0.5)

Cooling_m.iloc[:, 1:3].plot(ax=axes[4,0])
Cooling_m['mean'].plot(kind ='bar', ax=axes[4,0])
axes[4,0].set_title('Cooling Coil Total Cooling Rate [W]')
axes[4,0].fill_between(Cooling_m['month'], Cooling_m['mean+std'], Cooling_m['mean-std'], facecolor='blue', alpha=0.5)

Heating_m.iloc[:, 1:3].plot(ax=axes[4,1])
Heating_m['mean'].plot(kind ='bar', ax=axes[4,1])
axes[4,1].set_title('Heating Coil Heating Rate [W]')
axes[4,1].fill_between(Heating_m['month'], Heating_m['mean+std'], Heating_m['mean-std'], facecolor='blue', alpha=0.5)

People_m.iloc[:, :3].plot(ax=axes[5,0])
People_m['mean'].plot(kind ='bar', ax=axes[5,0])
axes[5,0].set_title('Zone People Occupant Count')
axes[5,0].fill_between(People_m['month'], People_m['mean+std'], People_m['mean-std'], facecolor='blue', alpha=0.5)

Wind_m.iloc[:, :3].plot(ax=axes[5,1])
Wind_m['mean'].plot(kind ='bar', ax=axes[5,1])
axes[5,1].set_title('Site Wind Speed [m/s]')
axes[5,1].fill_between(Wind_m['month'], Wind_m['mean+std'], Wind_m['mean-std'], facecolor='blue', alpha=0.5)
plt.show()

# PER GIORNO DELLA SETTIMANA
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(20,25))
fig.suptitle('Statistics variables per day', fontsize=50)
fig.subplots_adjust(hspace=0.5)
Tout_d.iloc[:, :3].plot(ax=axes[0,0])
Tout_d['mean'].plot(kind ='bar', ax=axes[0,0])
axes[0,0].set_title('Site Outdoor Air Drybulb Temperature [C]')
axes[0,0].fill_between(Tout_d['day'], Tout_d['mean+std'], Tout_d['mean-std'], facecolor='blue', alpha=0.5)

Tint_d.iloc[:, :3].plot(ax=axes[1,0])
Tint_d['mean'].plot(kind ='bar', ax=axes[1,0])
axes[1,0].set_title('Zone Mean Air Temperature [C]')
axes[1,0].fill_between(Tint_d['day'], Tint_d['mean+std'], Tint_d['mean-std'], facecolor='blue', alpha=0.5)

Hu_out_d.iloc[:, :3].plot(ax=axes[0,1])
Hu_out_d['mean'].plot(kind ='bar', ax=axes[0,1])
axes[0,1].set_title('Site outdoor Air Relative Humidity [%]')
axes[0,1].fill_between(Hu_out_d['day'], Hu_out_d['mean+std'], Hu_out_d['mean-std'], facecolor='blue', alpha=0.5)

Hu_int_d.iloc[:, :3].plot(ax=axes[1,1])
Hu_int_d['mean'].plot(kind ='bar', ax=axes[1,1])
axes[1,1].set_title('Zone Air Relative Humidity [%]')
axes[1,1].fill_between(Hu_int_d['day'], Hu_int_d['mean+std'], Hu_int_d['mean-std'], facecolor='blue', alpha=0.5)

Diff_rad_d.iloc[:, :3].plot(ax=axes[2,0])
Diff_rad_d['mean'].plot(kind ='bar', ax=axes[2,0])
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
axes[2,0].fill_between(Diff_rad_d['day'], Diff_rad_d['mean+std'], Diff_rad_d['mean-std'], facecolor='blue', alpha=0.5)

Dir_rad_d.iloc[:, :3].plot(ax=axes[2,1])
Dir_rad_d['mean'].plot(kind ='bar', ax=axes[2,1])
axes[2,1].set_title('Direct Solar Radiation Rate per Area [W/m2]')
axes[2,1].fill_between(Dir_rad_d['day'], Dir_rad_d['mean+std'], Dir_rad_d['mean-std'], facecolor='blue', alpha=0.5)

PMV_d.iloc[:, :3].plot(ax=axes[3,0])
PMV_d['mean'].plot(kind ='bar', ax=axes[3,0])
axes[3,0].set_title('PMV')
axes[3,0].fill_between(PMV_d['day'], PMV_d['mean+std'], PMV_d['mean-std'], facecolor='blue', alpha=0.5)

PPD_d.iloc[:, :3].plot(ax=axes[3,1])
PPD_d['mean'].plot(kind ='bar', ax=axes[3,1])
axes[3,1].set_title('PPD')
axes[3,1].fill_between(PPD_d['day'], PPD_d['mean+std'], PPD_d['mean-std'], facecolor='blue', alpha=0.5)

Cooling_d.iloc[:, 1:3].plot(ax=axes[4,0])
Cooling_d['mean'].plot(kind ='bar', ax=axes[4,0])
axes[4,0].set_title('Cooling Coil Total Cooling Rate [W]')
axes[4,0].fill_between(Cooling_d['day'], Cooling_d['mean+std'], Cooling_d['mean-std'], facecolor='blue', alpha=0.5)

Heating_d.iloc[:, 1:3].plot(ax=axes[4,1])
Heating_d['mean'].plot(kind ='bar', ax=axes[4,1])
axes[4,1].set_title('Heating Coil Heating Rate [W]')
axes[4,1].fill_between(Heating_d['day'], Heating_d['mean+std'], Heating_d['mean-std'], facecolor='blue', alpha=0.5)

People_d.iloc[:, :3].plot(ax=axes[5,0])
People_d['mean'].plot(kind ='bar', ax=axes[5,0])
axes[5,0].set_title('Zone People Occupant Count')
axes[5,0].fill_between(People_d['day'], People_d['mean+std'], People_d['mean-std'], facecolor='blue', alpha=0.5)

Wind_d.iloc[:, :3].plot(ax=axes[5,1])
Wind_d['mean'].plot(kind ='bar', ax=axes[5,1])
axes[5,1].set_title('Site Wind Speed [m/s]')
axes[5,1].fill_between(Wind_d['day'], Wind_d['mean+std'], Wind_d['mean-std'], facecolor='blue', alpha=0.5)
plt.show()

# PER ORA DEL GIORNO
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(20,25))
fig.suptitle('Statistics variables per hour', fontsize=50)
fig.subplots_adjust(hspace=0.5)
Tout_h.iloc[:, :3].plot(ax=axes[0,0])
Tout_h['mean'].plot(kind ='bar', ax=axes[0,0])
axes[0,0].set_title('Site Outdoor Air Drybulb Temperature [C]')
axes[0,0].fill_between(Tout_h['hour'], Tout_h['mean+std'], Tout_h['mean-std'], facecolor='blue', alpha=0.5)

Tint_h.iloc[:, :3].plot(ax=axes[1,0])
Tint_h['mean'].plot(kind ='bar', ax=axes[1,0])
axes[1,0].set_title('Zone Mean Air Temperature [C]')
axes[1,0].fill_between(Tint_h['hour'], Tint_h['mean+std'], Tint_h['mean-std'], facecolor='blue', alpha=0.5)

Hu_out_h.iloc[:, :3].plot(ax=axes[0,1])
Hu_out_h['mean'].plot(kind ='bar', ax=axes[0,1])
axes[0,1].set_title('Site outdoor Air Relative Humidity [%]')
axes[0,1].fill_between(Hu_out_h['hour'], Hu_out_h['mean+std'], Hu_out_h['mean-std'], facecolor='blue', alpha=0.5)

Hu_int_h.iloc[:, :3].plot(ax=axes[1,1])
Hu_int_h['mean'].plot(kind ='bar', ax=axes[1,1])
axes[1,1].set_title('Zone Air Relative Humidity [%]')
axes[1,1].fill_between(Hu_int_h['hour'], Hu_int_h['mean+std'], Hu_int_h['mean-std'], facecolor='blue', alpha=0.5)

Diff_rad_h.iloc[:, :3].plot(ax=axes[2,0])
Diff_rad_h['mean'].plot(kind ='bar', ax=axes[2,0])
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
axes[2,0].fill_between(Diff_rad_h['hour'], Diff_rad_h['mean+std'], Diff_rad_h['mean-std'], facecolor='blue', alpha=0.5)

Dir_rad_h.iloc[:, :3].plot(ax=axes[2,1])
Dir_rad_h['mean'].plot(kind ='bar', ax=axes[2,1])
axes[2,1].set_title('Direct Solar Radiation Rate per Area [W/m2]')
axes[2,1].fill_between(Dir_rad_h['hour'], Dir_rad_h['mean+std'], Dir_rad_h['mean-std'], facecolor='blue', alpha=0.5)

PMV_h.iloc[:, :3].plot(ax=axes[3,0])
PMV_h['mean'].plot(kind ='bar', ax=axes[3,0])
axes[3,0].set_title('PMV')
axes[3,0].fill_between(PMV_h['hour'], PMV_h['mean+std'], PMV_h['mean-std'], facecolor='blue', alpha=0.5)

PPD_h.iloc[:, :3].plot(ax=axes[3,1])
PPD_h['mean'].plot(kind ='bar', ax=axes[3,1])
axes[3,1].set_title('PPD')
axes[3,1].fill_between(PPD_h['hour'], PPD_h['mean+std'], PPD_h['mean-std'], facecolor='blue', alpha=0.5)

Cooling_h.iloc[:, 1:3].plot(ax=axes[4,0])
Cooling_h['mean'].plot(kind ='bar', ax=axes[4,0])
axes[4,0].set_title('Cooling Coil Total Cooling Rate [W]')
axes[4,0].fill_between(Cooling_h['hour'], Cooling_h['mean+std'], Cooling_h['mean-std'], facecolor='blue', alpha=0.5)

Heating_h.iloc[:, 1:3].plot(ax=axes[4,1])
Heating_h['mean'].plot(kind ='bar', ax=axes[4,1])
axes[4,1].set_title('Heating Coil Heating Rate [W]')
axes[4,1].fill_between(Heating_h['hour'], Heating_h['mean+std'], Heating_h['mean-std'], facecolor='blue', alpha=0.5)

People_h.iloc[:, :3].plot(ax=axes[5,0])
People_h['mean'].plot(kind ='bar', ax=axes[5,0])
axes[5,0].set_title('Zone People Occupant Count')
axes[5,0].fill_between(People_h['hour'], People_h['mean+std'], People_h['mean-std'], facecolor='blue', alpha=0.5)

Wind_h.iloc[:, :3].plot(ax=axes[5,1])
Wind_h['mean'].plot(kind ='bar', ax=axes[5,1])
axes[5,1].set_title('Site Wind Speed [m/s]')
axes[5,1].fill_between(Wind_h['hour'], Wind_h['mean+std'], Wind_h['mean-std'], facecolor='blue', alpha=0.5)
plt.show()


"""
# Media per mese
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,22))
fig.suptitle('Statistics variables per month', fontsize=50)
fig.subplots_adjust(hspace=0.4)
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[0,0], label='')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.month).max().plot(ax=axes[0,0],c='r', label='Max')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.month).min().plot(ax=axes[0,0],c='g', label='Min')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.month).median().plot(ax=axes[0,0],c ='y', label='Median')
axes[0,0].set_title('Outdoor Air Drybulb Temperature [C]')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[1,0], label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.month).max().plot(ax=axes[1,0],c='r', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.month).min().plot(ax=axes[1,0],c='g', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.month).median().plot(ax=axes[1,0],c ='y', label='')
axes[1,0].set_title('Outdoor Air Relative Humidity [%]')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[2,0], label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).max().plot(ax=axes[2,0],c='r', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).min().plot(ax=axes[2,0],c='g', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).median().plot(ax=axes[2,0],c ='y', label='')
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[3,0], label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).max().plot(ax=axes[3,0],c='r', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).min().plot(ax=axes[3,0],c='g', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.month).median().plot(ax=axes[3,0],c ='y', label='')
axes[3,0].set_title('Direct Solar Radiation Rate per Area [W/m2]')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[4,0], label='')
# df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).max().plot(ax=axes[4,0],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).min().plot(ax=axes[4,0],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).median().plot(ax=axes[4,0],c ='y', label='')
axes[4,0].set_title('Heating Coil Heating Rate [W]')
#df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[4,0])
#axes[4,0].set_title('Zone Mean Radiant Temperature [°C]')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[0,1], label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.month).max().plot(ax=axes[0,1],c='r', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.month).min().plot(ax=axes[0,1],c='g', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.month).median().plot(ax=axes[0,1],c ='y', label='')
axes[0,1].set_title('Zone Mean Air Temperature [°C]')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[1,1], label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.month).max().plot(ax=axes[1,1],c='r', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.month).min().plot(ax=axes[1,1],c='g', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.month).median().plot(ax=axes[1,1],c ='y', label='')
axes[1,1].set_title('Zone Air Relative Humidity [%]')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[2,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.month).max().plot(ax=axes[2,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.month).min().plot(ax=axes[2,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.month).median().plot(ax=axes[2,1],c ='y', label='')
axes[2,1].set_title('Thermal Comfort Fanger Model PMV')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[3,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.month).max().plot(ax=axes[3,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.month).min().plot(ax=axes[3,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.month).median().plot(ax=axes[3,1],c ='y', label='')
axes[3,1].set_title('Zone Thermal Comfort Fanger Model PPD')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[4,1], label='')
# df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.month).max().plot(ax=axes[4,1],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.month).min().plot(ax=axes[4,1],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.month).median().plot(ax=axes[4,1],c ='y', label='')
axes[4,1].set_title('Cooling Coil Total Cooling Rate [W]')
#df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[4,1])
#axes[5,0].set_title('Heating supply Coil Heating Rate [W]')
fig.legend(loc='upper right')
plt.show()


# Medie per ora
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,22))
fig.suptitle('Statistics variables per hour', fontsize=50)
fig.subplots_adjust(hspace=0.4)
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[0,0], label='')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.hour).max().plot(ax=axes[0,0],c='r', label='Max')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.hour).min().plot(ax=axes[0,0],c='g', label='Min')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.hour).median().plot(ax=axes[0,0],c ='y', label='Median')
axes[0,0].set_title('Outdoor Air Drybulb Temperature [C]')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[1,0], label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.hour).max().plot(ax=axes[1,0],c='r', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.hour).min().plot(ax=axes[1,0],c='g', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.hour).median().plot(ax=axes[1,0],c ='y', label='')
axes[1,0].set_title('Outdoor Air Relative Humidity [%]')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[2,0], label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).max().plot(ax=axes[2,0],c='r', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).min().plot(ax=axes[2,0],c='g', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).median().plot(ax=axes[2,0],c ='y', label='')
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[3,0], label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).max().plot(ax=axes[3,0],c='r', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).min().plot(ax=axes[3,0],c='g', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.hour).median().plot(ax=axes[3,0],c ='y', label='')
axes[3,0].set_title('Direct Solar Radiation Rate per Area [W/m2]')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[4,0], label='')
# df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).max().plot(ax=axes[4,0],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).min().plot(ax=axes[4,0],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.hour).median().plot(ax=axes[4,0],c ='y', label='')
axes[4,0].set_title('Heating Coil Heating Rate [W]')
#df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[4,0])
#axes[4,0].set_title('Zone Mean Radiant Temperature [°C]')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[0,1], label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.hour).max().plot(ax=axes[0,1],c='r', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.hour).min().plot(ax=axes[0,1],c='g', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.hour).median().plot(ax=axes[0,1],c ='y', label='')
axes[0,1].set_title('Zone Mean Air Temperature [°C]')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[1,1], label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.hour).max().plot(ax=axes[1,1],c='r', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.hour).min().plot(ax=axes[1,1],c='g', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.hour).median().plot(ax=axes[1,1],c ='y', label='')
axes[1,1].set_title('Zone Air Relative Humidity [%]')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[2,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.hour).max().plot(ax=axes[2,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.hour).min().plot(ax=axes[2,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.hour).median().plot(ax=axes[2,1],c ='y', label='')
axes[2,1].set_title('Thermal Comfort Fanger Model PMV')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[3,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.hour).max().plot(ax=axes[3,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.hour).min().plot(ax=axes[3,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.hour).median().plot(ax=axes[3,1],c ='y', label='')
axes[3,1].set_title('Zone Thermal Comfort Fanger Model PPD')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.hour).mean().plot.bar(ax=axes[4,1], label='')
# df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.hour).max().plot(ax=axes[4,1],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.hour).min().plot(ax=axes[4,1],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.hour).median().plot(ax=axes[4,1],c ='y', label='')
axes[4,1].set_title('Cooling Coil Total Cooling Rate [W]')
#df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.month).mean().plot.bar(ax=axes[4,1])
#axes[5,0].set_title('Heating supply Coil Heating Rate [W]')
fig.legend(loc='upper right')
plt.show()

# Medie per giorno
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,22))
fig.suptitle('Statistics variables per hour', fontsize=50)
fig.subplots_adjust(hspace=0.4)
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[0,0], label='')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.day).max().plot(ax=axes[0,0],c='r', label='Max')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.day).min().plot(ax=axes[0,0],c='g', label='Min')
df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].groupby(df.day).median().plot(ax=axes[0,0],c ='y', label='Median')
axes[0,0].set_title('Outdoor Air Drybulb Temperature [C]')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[1,0], label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.day).max().plot(ax=axes[1,0],c='r', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.day).min().plot(ax=axes[1,0],c='g', label='')
df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'].groupby(df.day).median().plot(ax=axes[1,0],c ='y', label='')
axes[1,0].set_title('Outdoor Air Relative Humidity [%]')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[2,0], label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).max().plot(ax=axes[2,0],c='r', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).min().plot(ax=axes[2,0],c='g', label='')
df['Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).median().plot(ax=axes[2,0],c ='y', label='')
axes[2,0].set_title('Diffuse Solar Radiation Rate per Area [W/m2]')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[3,0], label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).max().plot(ax=axes[3,0],c='r', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).min().plot(ax=axes[3,0],c='g', label='')
df['Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)'].groupby(df.day).median().plot(ax=axes[3,0],c ='y', label='')
axes[3,0].set_title('Direct Solar Radiation Rate per Area [W/m2]')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[4,0], label='')
# df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).max().plot(ax=axes[4,0],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).min().plot(ax=axes[4,0],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).median().plot(ax=axes[4,0],c ='y', label='')
axes[4,0].set_title('Heating Coil Heating Rate [W]')
#df['CORE_ZN:Zone Mean Radiant Temperature [C](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[4,0])
#axes[4,0].set_title('Zone Mean Radiant Temperature [°C]')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[0,1], label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.day).max().plot(ax=axes[0,1],c='r', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.day).min().plot(ax=axes[0,1],c='g', label='')
df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].groupby(df.day).median().plot(ax=axes[0,1],c ='y', label='')
axes[0,1].set_title('Zone Mean Air Temperature [°C]')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[1,1], label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.day).max().plot(ax=axes[1,1],c='r', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.day).min().plot(ax=axes[1,1],c='g', label='')
df['CORE_ZN:Zone Air Relative Humidity [%](TimeStep)'].groupby(df.day).median().plot(ax=axes[1,1],c ='y', label='')
axes[1,1].set_title('Zone Air Relative Humidity [%]')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[2,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.day).max().plot(ax=axes[2,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.day).min().plot(ax=axes[2,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].groupby(df.day).median().plot(ax=axes[2,1],c ='y', label='')
axes[2,1].set_title('Thermal Comfort Fanger Model PMV')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[3,1], label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.day).max().plot(ax=axes[3,1],c='r', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.day).min().plot(ax=axes[3,1],c='g', label='')
df['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'].groupby(df.day).median().plot(ax=axes[3,1],c ='y', label='')
axes[3,1].set_title('Zone Thermal Comfort Fanger Model PPD')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[4,1], label='')
# df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.day).max().plot(ax=axes[4,1],c='r', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.day).min().plot(ax=axes[4,1],c='g', label='')
df['PSZ-AC:1 HEAT PUMP DX COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)'].groupby(df.day).median().plot(ax=axes[4,1],c ='y', label='')
axes[4,1].set_title('Cooling Coil Total Cooling Rate [W]')
#df['PSZ-AC:1 HEAT PUMP DX SUPP HEATING COIL:Heating Coil Heating Rate [W](TimeStep)'].groupby(df.day).mean().plot.bar(ax=axes[4,1])
#axes[5,0].set_title('Heating supply Coil Heating Rate [W]')
fig.legend(loc='upper right')
plt.show()
"""

