import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import calendar
from scipy.stats import zscore


'''
##########################################################
############ read in data, clean, and format  ############
##########################################################
'''

# read in
df = pd.read_csv("./Input/Home_Energy_Usage_2.csv")

# clean and make long
df.columns = df.columns.str.replace(" kWH", "", regex=False)
df_long = df.melt(id_vars=["Date"], var_name="Time", value_name="kWh")
df_long["Timestamp"] = pd.to_datetime(df_long["Date"] + " " + df_long["Time"], format="%m/%d/%Y %I:%M %p")
df_long = df_long.drop(columns=["Date", "Time"])

# set index for tsa and resort just in case
df_long = df_long.set_index("Timestamp")
df_long = df_long.sort_index()

# print and export if needed
# print('Final DF: ', df_long.head())
# df_long.to_csv("./Cleaned_Data/cleaned_home_energy_usage2.csv")

'''
#################################################################
########################## EDA ##################################
#################################################################
'''

'''
#############################
############ Max values and experimenting with printing
#############################
'''

# find max usage for 30 min interval 
max_row = df_long.loc[df_long["kWh"].idxmax()]
print(f"Max 30 min interval usage {max_row['kWh']} kWh at {max_row.name}")

# find max usage for 1 day
daily_kwh = df_long.resample("D").sum()
print(f'Highest usage day was {daily_kwh["kWh"].idxmax().date()} with: {daily_kwh["kWh"].max()} kWh')

# find top 5 weeks of usage
weekly_kwh = df_long.resample("W").sum()
top_weeks = weekly_kwh.sort_values("kWh", ascending=False).head()
print("🔝 Top 5 Highest Usage Weeks:")
for date, kwh in top_weeks["kWh"].items():
    print(f"📅 {date.strftime('%Y-%m-%d')}⚡ {kwh:.2f} kWh")

# find max usage for 1 month
monthly_kwh = df_long.resample('M').sum()
top_months = monthly_kwh.sort_values("kWh", ascending=False).head(3)
top_months.index = top_months.index.to_period('M')
print("\n🔝 Top 3 Highest Usage Months:")
print(tabulate(top_months, headers=["Date", "kWh"], tablefmt="grid"))

'''
#############################
############ data information and statistical info
#############################
'''

### Data information
print('\n~Data Information~')

# basic sumamry
print("\n🔍 Summary Statistics:")
print(df_long.describe())
# check for missing data
print("\n❓ Missing Data Check:")
print(df_long.isnull().sum())

# there was missing data but I think it is daylight savings so lets check the date
missing_dates = df_long[df_long['kWh'].isnull()]
print("\n❓ Dates with Missing Data in 'kWh' column:")
print(missing_dates.index)

# outliers with z score
df_long_outliers = df_long.dropna().copy()
df_long_outliers['z_score'] = zscore(df_long_outliers['kWh'])
outliers = df_long_outliers[df_long_outliers['z_score'].abs() > 3]  # Threshold for outliers
print("🚨 Outliers:\n", outliers)


'''
#############################
############ simple plots to visualize data and possible trends
#############################
'''

# distribution plot
print("\n📊 Distribution of kWh Usage:")
plt.figure(figsize=(10, 6))
sns.histplot(df_long['kWh'], bins=40, kde=True, color='seagreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of kWh Usage', fontsize=14)
plt.xlabel('kWh 30min Usage', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# box plot to see outliers
print("\n📊 Boxplot of kWh Usage:")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_long['kWh'], color='seagreen', flierprops={'marker': 'o', 'markersize': 6, 'markerfacecolor': 'red'})
# plt.title('30 min kWh Usage', fontsize=14)
plt.xlabel('kWh 30min Usage', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()


# hour avg consumption
print("\n🕐 Average kWh Usage by Hour of Day:")
df_long['hour'] = df_long.index.hour
hourly_avg = df_long.groupby('hour')['kWh'].mean()
plt.figure(figsize=(10, 6))
hourly_avg.plot(marker='o', linestyle='-', color='seagreen')
plt.xticks(range(0, 24))  # Ensure all hours are shown
plt.title('Average kWh Usage by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average kWh')
plt.ylim(0, hourly_avg.max() * 1.1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# week avg consumption
print("\n🕐 Average kWh Usage by Day of Week:")
df_long['day_of_week'] = df_long.index.dayofweek
daily_avg = df_long.groupby('day_of_week')['kWh'].mean()
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 6))
daily_avg.plot(kind='bar', color='seagreen', alpha=0.8, edgecolor='black')
plt.xticks(ticks=range(7), labels=day_labels, rotation=45)  # Replace numbers with day names
plt.title('Average kWh Usage by Day')
plt.xlabel('Day of Week')
plt.ylabel('Average kWh')
plt.ylim(0, 0.75)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout() 
plt.show()


# month avg consumption
print("\n🕐 Average kWh Usage by Month:")
df_long['month'] = df_long.index.month
monthly_avg = df_long.groupby('month')['kWh'].mean()
month_labels = [calendar.month_name[i] for i in range(1, 13)]  # Full month names (January - December)
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar', color='seagreen', alpha=0.8, edgecolor='black')
plt.xticks(ticks=range(12), labels=month_labels, rotation=45)  # Replace numbers with month names
plt.title('Average kWh Usage by Month')
plt.xlabel('Month')
plt.ylabel('Average kWh')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout() 
plt.show()


# heat map
print("\n🔥 Heatmap of kWh Usage by Hour per Month:")
df_long['month'] = df_long.index.month
df_long['hour'] = df_long.index.hour
heatmap_data = df_long.pivot_table(index='hour', columns=df_long.index.to_period("M"), values='kWh', aggfunc='sum')
plt.figure(figsize=(14, 6))
ax = sns.heatmap(heatmap_data, cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'kWh Usage'}, fmt=".1f")
ax.set_xticklabels([pd.to_datetime(str(x)).strftime('%b %Y') for x in heatmap_data.columns], rotation=45, ha='right')
ax.set_yticklabels([f"{int(y)}:00" for y in heatmap_data.index], rotation=0)
plt.title("Energy Usage Heatmap", fontsize=14, pad=10)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Hour of Day", fontsize=12)
plt.tight_layout() 
plt.show()


# monthly distribution
plt.figure(figsize=(12,6))
sns.boxplot(x=df_long.index.month, y=df_long['kWh'], hue=df_long.index.month, palette="coolwarm", legend=False)
plt.title("Monthly Distribution of kWh Usage")
plt.xlabel("Month")
plt.ylabel("kWh")
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.show() # don't need for now


'''
#############################
########### export data to be used in tsa and ml models
#############################
'''
# print and export if needed
print('Final DF: ', df_long.head())
df_long.to_csv("./Cleaned_Data/modified_cleaned_home_energy_usage2.csv")