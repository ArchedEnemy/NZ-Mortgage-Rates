import pandas as pd 
import datetime as dt
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


#Directory of output
directory = r'C:\Users\User\Documents\GitHub\NZ-Mortgage-Rates'
today = date.today() 
today = today.strftime("%d/%m/%Y")

df=pd.read_csv(r'X:\InterestRates.csv', parse_dates = ['time_stamp'], dayfirst=True)
date1 = pd.to_datetime(df['time_stamp'], errors='coerce', dayfirst=True,format="%d/%m/%Y %H:%M")
date2 = pd.to_datetime(df['time_stamp'], errors='coerce',format="%Y-%m-%d %H:%M:%S")
df['date'] = date1.fillna(date2)

floating = df[(df.Floating > 0)][['Institution','Product','Floating','date']]
floating['InterestRateType']='Floating'
floating.rename(columns = {'Floating':'InterestRate'}, inplace = True)
months6 = df[(df.Months6 > 0)][['Institution','Product','Months6','date']]
months6['InterestRateType']='6 Months'
months6.rename(columns = {'Months6':'InterestRate'}, inplace = True)
year1 = df[(df.Year1 > 0)][['Institution','Product','Year1','date']]
year1['InterestRateType']='1 Year'
year1.rename(columns = {'Year1':'InterestRate'}, inplace = True)
months18 = df[(df.Months6 > 0)][['Institution','Product','Months18','date']]
months18['InterestRateType']='18 Months'
months18.rename(columns = {'Months18':'InterestRate'}, inplace = True)
year2 = df[(df.Year2 > 0)][['Institution','Product','Year2','date']]
year2['InterestRateType']='2 Year'
year2.rename(columns = {'Year2':'InterestRate'}, inplace = True)
year3 = df[(df.Year3 > 0)][['Institution','Product','Year3','date']]
year3['InterestRateType']='3 Year'
year3.rename(columns = {'Year3':'InterestRate'}, inplace = True)
year4 = df[(df.Year4 > 0)][['Institution','Product','Year4','date']]
year4['InterestRateType']='4 Year'
year4.rename(columns = {'Year4':'InterestRate'}, inplace = True)
year5 = df[(df.Year5 > 0)][['Institution','Product','Year5','date']]
year5['InterestRateType']='5 Year'
year5.rename(columns = {'Year5':'InterestRate'}, inplace = True)
df_all= pd.concat([floating, months6, year1, months18, year2, year3, year4, year5])
df_all['ReportProduct']=df_all.Institution.str.cat(df_all.Product, sep='-')
df_all['BankType'] = np.where(df_all['Institution']=='ASB', 'Major'
    ,np.where(df_all['Institution']=='ANZ', 'Major'
    ,np.where(df_all['Institution']=='BNZ', 'Major'
    ,np.where(df_all['Institution']=='Westpac', 'Major'
    ,np.where(df_all['Institution']=='Kiwibank', 'Minor'
    ,np.where(df_all['Institution']=='HSBC', 'Minor'
    ,np.where(df_all['Institution']=='TSB Bank', 'Minor'
    ,np.where(df_all['Institution']=='Heartland Bank', 'Minor'
    ,np.where(df_all['Institution']=='Co-operative Bank', 'Minor'
    ,'Other')))))))))
df_all['ProductType'] = np.where((df_all['Institution']=='ASB') & (df_all['Product']=='Standard'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='BNZ') & (df_all['Product']=='Standard'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='Bank of China') & (df_all['Product']=='Special'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='China Construction Bank') & (df_all['Product']=='Non Standard'), 'Standard'
    ,np.where((df_all['Institution']=='China Construction Bank') & (df_all['Product']=='Standard'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='Kiwibank') & (df_all['Product']=='Standard'), 'Standard'
    ,np.where((df_all['Institution']=='Kiwibank') & (df_all['Product']=='Special'), 'LVR < 80%'
  
    ,np.where((df_all['Institution']=='SBS Bank') & (df_all['Product']=='Residential'), 'Standard'
    ,np.where((df_all['Institution']=='SBS Bank') & (df_all['Product']=='Special'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='Co-operative Bank') & (df_all['Product']=='Standard'), 'Standard'
    ,np.where((df_all['Institution']=='Co-operative Bank') & (df_all['Product']=='Owner Occupied'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='Heartland Bank') & (df_all['Product']=='Residential'), 'LVR < 80%'
    ,np.where((df_all['Institution']=='TSB Bank') & (df_all['Product']=='Standard'), 'Standard'
    ,np.where((df_all['Institution']=='TSB Bank') & (df_all['Product']=='Special LVR <80%'), 'LVR < 80%'
        
             
    
    ,np.where(df_all['Product']=='Better Homes Top Up', 'Other'
    ,np.where(df_all['Product']=='Blueprint to Build', 'Construction'
    ,np.where(df_all['Product']=='Choices Everyday', 'Revolving'
    ,np.where(df_all['Product']=='Choices Offset', 'Offset'
    ,np.where(df_all['Product']=='Construction lending for FHB', 'First Home Buyer'
    ,np.where(df_all['Product']=='First Home Buyer Special', 'First Home Buyer'
    ,np.where(df_all['Product']=='First Home Combo', 'First Home Buyer'
    ,np.where(df_all['Product']=='Good Energy - Up to $80K', 'Green'
    ,np.where(df_all['Product']=='Green Home Loan top-ups', 'Green'
    ,np.where(df_all['Product']=='Non Standard', 'Special'
    ,np.where(df_all['Product']=='Offset Mortgage', 'Offset'
    ,np.where(df_all['Product']=='Owner Occupied', 'Special'
    ,np.where(df_all['Product']=='Premier', 'Standard'
    ,np.where(df_all['Product']=='Residential', 'Standard'
    ,np.where(df_all['Product']=='Reverse Mortgage', 'Reverse'
    ,np.where(df_all['Product']=='Special', 'Special'
    ,np.where(df_all['Product']=='Special - Classic', 'LVR < 80%'
    ,np.where(df_all['Product']=='Special LVR < 80%', 'LVR < 80%'
    ,np.where(df_all['Product']=='Special LVR <80%', 'LVR < 80%'
    ,np.where(df_all['Product']=='Special LVR under 80%', 'LVR < 80%'
              
    ,np.where(df_all['Product']=='Standard', 'Standard'             
    ,np.where(df_all['Product']=='Std & Flybuys', 'Standard'
    ,np.where(df_all['Product']=='TotalMoney', 'Offset'
    ,np.where(df_all['Product']=='Unwind reverse equity', 'Reverse'
    ,'Other'))))))))))))))))))))))))))))))))))))))
df_all = df_all[['ReportProduct','BankType','ProductType','Institution','Product','InterestRateType','InterestRate','date']]

df_all['LatestDateFlag'] = np.where(df_all['date']==max(df_all['date']), 'Yes', 'No')
month_1t = max(df_all['date']) - pd.DateOffset(months=1)
month_1 = min(df_all[(df_all['date']>month_1t)]['date'])
month_2t = max(df_all['date']) - pd.DateOffset(months=2)
month_2 = min(df_all[(df_all['date']>month_2t)]['date'])
month_3t = max(df_all['date']) - pd.DateOffset(months=3)
month_3 = min(df_all[(df_all['date']>month_3t)]['date'])
month_4t = max(df_all['date']) - pd.DateOffset(months=4)
month_4 = min(df_all[(df_all['date']>month_4t)]['date'])
month_5t = max(df_all['date']) - pd.DateOffset(months=5)
month_5 = min(df_all[(df_all['date']>month_5t)]['date'])
month_6t = max(df_all['date']) - pd.DateOffset(months=6)
month_6 = min(df_all[(df_all['date']>month_6t)]['date'])

df_all['Months1AgoFlag'] = np.where(df_all['LatestDateFlag']=='Yes', 'Yes'
    ,np.where(df_all['date']==month_1, 'Yes'
    ,'No'))
df_all['Months3AgoFlag'] = np.where(df_all['LatestDateFlag']=='Yes', 'Yes'
    ,np.where(df_all['date']==month_3, 'Yes'
    ,'No'))
df_all['Months6AgoFlag'] = np.where(df_all['LatestDateFlag']=='Yes', 'Yes'
    ,np.where(df_all['date']==month_6, 'Yes'
    ,'No'))
df_all['Last6MonthsFlag'] = np.where(df_all['LatestDateFlag']=='Yes', 'Yes'
    ,np.where(df_all['date']==month_1, 'Yes'
    ,np.where(df_all['date']==month_2, 'Yes'
    ,np.where(df_all['date']==month_3, 'Yes'
    ,np.where(df_all['date']==month_4, 'Yes'
    ,np.where(df_all['date']==month_5, 'Yes'
    ,np.where(df_all['date']==month_6, 'Yes'
    ,'No')))))))

df_all['date'] = pd.to_datetime(df_all['date']).dt.date
df_all.to_csv(r'C:\Users\User\Documents\NZ Mortgage Normalised.csv', index = False)


#get the last change date
latest_rates = df_all[(df_all.LatestDateFlag == 'Yes')]
previous_rates = pd.merge(df_all, latest_rates, how="left", on=['Institution','InterestRateType','ProductType','InterestRate'], indicator=True)
previous_rates = previous_rates[previous_rates['_merge'] == 'left_only'] 
previous_rates.rename(columns={'date_x': 'LastChangeDate'}, inplace=True)   
previous_rates = previous_rates.groupby(['Institution', 'ProductType', 'InterestRateType'])['LastChangeDate'].max().reset_index()

# Add one day to the last change date
previous_rates['LastChangeDate'] = pd.to_datetime(previous_rates['LastChangeDate']) + pd.DateOffset(days=1)

# Calculate the number of days passed between today and the last rate change date
previous_rates['DaysSinceLastChange'] = (pd.to_datetime(today, format="%d/%m/%Y") - pd.to_datetime(previous_rates['LastChangeDate'])).dt.days


df_mjr6 = df_all[(df_all.ProductType.isin(['Standard','LVR < 80%'])) & (df_all.Institution.isin(['ANZ','ASB','BNZ','Westpac','Kiwibank']))] [['Institution','InterestRateType','InterestRate','date','ProductType']]
df_mjr6['Institution'] = df_mjr6['Institution'] + np.where(df_mjr6['ProductType']=='LVR < 80%', '*','')
df_mjr6 = df_mjr6.loc[:, df_mjr6.columns != 'ProductType']
df_mjr6 = df_mjr6.rename(columns={'Institution': 'Bank', 'InterestRateType': 'Product', 'InterestRate': 'Rate', 'date': 'Date'})
df_mjr6['Date'] = pd.to_datetime(df_mjr6['Date'])

month_24t = max(df_mjr6['Date']) - pd.DateOffset(months=24)
month_24 = min(df_mjr6[(df_mjr6['Date']>month_24t)]['Date'])
df_mjr6 = df_mjr6[(df_mjr6.Date >= month_24t)]



# Get the unique products
products = df_mjr6['Product'].unique()

# Define the color codes for the banks
color_codes = {
    'ASB': '#F7CE42',
    'ASB*': '#F7CE42',
    'ANZ': '#007dba',
    'ANZ*': '#007dba',
    'BNZ': '#193487',
    'BNZ*': '#193487',
    'Westpac': '#ff1339',
    'Westpac*': '#ff1339',
    'Kiwibank': '#65cd32',
    'Kiwibank*': '#65cd32',
}

mnratef = min(df_mjr6[(df_mjr6.Product =='Floating')]['Rate']) - 0.1
mxratef = max(df_mjr6[(df_mjr6.Product =='Floating')]['Rate']) + 0.1

mnrate = min(df_mjr6[(df_mjr6.Product !='Floating')]['Rate']) - 0.1
mxrate = max(df_mjr6[(df_mjr6.Product !='Floating')]['Rate']) + 0.1

# For each product, filter data and create a line chart
for product in products:
    product_df = df_mjr6[df_mjr6['Product'] == product]
    product_df['Date'] = pd.to_datetime(product_df['Date'])

    # Pivot the data to have banks as columns
    pivot_df = product_df.pivot(index='Date', columns='Bank', values='Rate')

    # Create a single line chart for all banks
    plt.figure(figsize=(10, 6))
    for bank, data in pivot_df.items():
        if bank[-1]== '*':
            plt.plot(data.index, data.values, label=bank, color=color_codes.get(bank, 'black'), linestyle='dashed')
        else:
            plt.plot(data.index, data.values, label=bank, color=color_codes.get(bank, 'black'))
        
    plt.title(f'{product}', fontsize=20)
    
    
    # major tick for every second week  
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

    #format xaxis date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))

    plt.xticks(rotation=90,fontsize=8)
    
    if product == 'Floating':
        plt.ylim(mnratef, mxratef)
    else:
        plt.ylim(mnrate, mxrate)
    
    plt.ylabel('Interest Rate')
    
    plt.legend(loc='center', bbox_to_anchor=(1.05, 0.5),
              ncol=1, fancybox=True, shadow=True)
    plt.gcf().text(0.89, 0.2, '* LVR < 80%', fontsize=10, bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 4})

    # Save the plot as a PNG file in the same directory as the CSV file
    plt.savefig(os.path.join(directory, f'{product}_interest_rates.png'))


df_latest = df_all[(df_all.ProductType.isin(['Standard'])) & (df_all.LatestDateFlag == 'Yes') & (df_all.Institution.isin(['ANZ','ASB','BNZ','Westpac','Kiwibank']))][['Institution','InterestRateType','InterestRate','date','ProductType']]
pivot_data = df_latest.pivot(index='Institution', columns='InterestRateType', values='InterestRate')

# Add a new row to the DataFrame
pivot_data.loc['ASB'] = np.nan
pivot_data.loc['BNZ'] = np.nan

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data = pivot_data.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data.sort_index(inplace=True ,ascending=False)

df_latest_change = previous_rates[(previous_rates.ProductType.isin(['Standard'])) & (previous_rates.Institution.isin(['ANZ','ASB','BNZ','Westpac','Kiwibank']))][['Institution','InterestRateType','LastChangeDate','DaysSinceLastChange']]
pivot_data_change = df_latest_change.pivot(index='Institution', columns='InterestRateType', values='DaysSinceLastChange')

# Add a new row to the DataFrame
pivot_data_change.loc['ASB'] = np.nan
pivot_data_change.loc['BNZ'] = np.nan

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data_change = pivot_data_change.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data_change.sort_index(inplace=True ,ascending=False)


# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10,6), dpi=300)

# Define the number of columns and rows
ncols = len(pivot_data.columns) + 1
nrows = len(pivot_data)

ax.set_xlim(0, ncols + 1)
ax.set_ylim(0, nrows + 1)

positions = [0.25] + [i + 1.5 for i in range(ncols - 1)]
columns = ['Bank'] + list(pivot_data.columns)

# Define the colors for normal and lowest values
normal_color = 'black'
lowest_color = 'black'

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        weight = 'normal'
        if j == 0:
            ha = 'left'
            text_label = f'{pivot_data.index[i]}'
            color = normal_color
            # Add bank name
            ax.annotate(
                xy=(positions[j], i + .5),
                text=text_label,
                ha=ha,
                va='center',
                weight=weight,
                fontsize=8,
                color=color
            )
        else:
            ha = 'center'
            # Add interest rate
            rate_value = pivot_data[column].iloc[i]
            days_value = pivot_data_change[column].iloc[i]
            
            if not np.isnan(rate_value):
                # Interest rate display
                if rate_value == pivot_data[column].min():
                    color = lowest_color
                    weight = 'bold'
                else:
                    color = normal_color
                    weight = 'normal'
                
                ax.annotate(
                    xy=(positions[j], i + .6),  # Slightly higher position
                    text=f'{rate_value}',
                    ha=ha,
                    va='center',
                    weight=weight,
                    fontsize=8,
                    color=color
                )
                
                # Days since last change display
                ax.annotate(
                    xy=(positions[j], i + .3),  # Slightly lower position
                    text=f'({int(days_value)})',
                    ha=ha,
                    va='center',
                    weight='normal',
                    fontsize=7,
                    color='gray'
                )

# Add column names
for index, c in enumerate(columns):
    if index == 0:
        ha = 'left'
    else:
        ha = 'center'
    ax.annotate(
        xy=(positions[index], nrows + .25),
        text=columns[index],
        ha=ha,
        va='bottom',
        weight='bold',
        fontsize=7
    )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

plt.title('Big 5 Banks Standard Rates - ' + today) 

ax.set_axis_off()
plt.savefig(
    os.path.join(directory,'latest standard rates.png'),
    dpi=300,
    transparent=True,
    bbox_inches='tight'
)


df_latest = df_all[(df_all.ProductType.isin(['LVR < 80%'])) & (df_all.LatestDateFlag == 'Yes') & (df_all.Institution.isin(['ANZ','ASB','BNZ','Westpac','Kiwibank']))][['Institution','InterestRateType','InterestRate','date','ProductType']]
pivot_data = df_latest.pivot(index='Institution', columns='InterestRateType', values='InterestRate')

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data = pivot_data.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data.sort_index(inplace=True ,ascending=False)

df_latest_change = previous_rates[(previous_rates.ProductType.isin(['LVR < 80%'])) & (previous_rates.Institution.isin(['ANZ','ASB','BNZ','Westpac','Kiwibank']))][['Institution','InterestRateType','LastChangeDate','DaysSinceLastChange']]
pivot_data_change = df_latest_change.pivot(index='Institution', columns='InterestRateType', values='DaysSinceLastChange')

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data_change = pivot_data_change.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data_change.sort_index(inplace=True ,ascending=False)

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10,6), dpi=300)

# Define the number of columns and rows
ncols = len(pivot_data.columns) + 1
nrows = len(pivot_data)

ax.set_xlim(0, ncols + 1)
ax.set_ylim(0, nrows + 1)

positions = [0.25] + [i + 1.5 for i in range(ncols - 1)]
columns = ['Bank'] + list(pivot_data.columns)

# Define the colors for normal and lowest values
normal_color = 'black'
lowest_color = 'black'

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        weight = 'normal'
        if j == 0:
            ha = 'left'
            text_label = f'{pivot_data.index[i]}'
            color = normal_color
            # Add bank name
            ax.annotate(
                xy=(positions[j], i + .5),
                text=text_label,
                ha=ha,
                va='center',
                weight=weight,
                fontsize=8,
                color=color
            )
        else:
            ha = 'center'
            # Add interest rate
            rate_value = pivot_data[column].iloc[i]
            days_value = pivot_data_change[column].iloc[i]
            
            if not np.isnan(rate_value):
                # Interest rate display
                if rate_value == pivot_data[column].min():
                    color = lowest_color
                    weight = 'bold'
                else:
                    color = normal_color
                    weight = 'normal'
                
                ax.annotate(
                    xy=(positions[j], i + .6),  # Slightly higher position
                    text=f'{rate_value}',
                    ha=ha,
                    va='center',
                    weight=weight,
                    fontsize=8,
                    color=color
                )
                
                # Days since last change display
                ax.annotate(
                    xy=(positions[j], i + .3),  # Slightly lower position
                    text=f'({int(days_value)})',
                    ha=ha,
                    va='center',
                    weight='normal',
                    fontsize=7,
                    color='gray'
                )

# Add column names
for index, c in enumerate(columns):
    if index == 0:
        ha = 'left'
    else:
        ha = 'center'
    ax.annotate(
        xy=(positions[index], nrows + .25),
        text=columns[index],
        ha=ha,
        va='bottom',
        weight='bold',
        fontsize=7
    )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

plt.title('Big 5 Banks Rate requiring LVR < 80% - ' + today) 

ax.set_axis_off()
plt.savefig(
    os.path.join(directory,'latest LVR less 80 rates.png'),
    dpi=300,
    transparent=True,
    bbox_inches='tight'
)



df_latest = df_all[(df_all.ProductType.isin(['Standard'])) & (df_all.LatestDateFlag == 'Yes') & (df_all.Institution.isin(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank']))][['Institution','InterestRateType','InterestRate','date','ProductType']]
df_latest['Institution'] = np.where(df_latest['Institution']=='Co-operative Bank', 'Co-op Bank'
    ,np.where(df_latest['Institution']=='Heartland Bank', 'Heartland'
    ,df_latest['Institution']))
pivot_data = df_latest.pivot(index='Institution', columns='InterestRateType', values='InterestRate')


# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data = pivot_data.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data.sort_index(inplace=True ,ascending=False)

df_latest_change = previous_rates[(previous_rates.ProductType.isin(['Standard'])) & (previous_rates.Institution.isin(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank']))][['Institution','InterestRateType','LastChangeDate','DaysSinceLastChange']]
df_latest_change['Institution'] = np.where(df_latest_change['Institution']=='Co-operative Bank', 'Co-op Bank'
    ,np.where(df_latest_change['Institution']=='Heartland Bank', 'Heartland'
    ,df_latest_change['Institution']))
pivot_data_change = df_latest_change.pivot(index='Institution', columns='InterestRateType', values='DaysSinceLastChange')

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data_change = pivot_data_change.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data_change.sort_index(inplace=True ,ascending=False)

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10,6), dpi=300)

# Define the number of columns and rows
ncols = len(pivot_data.columns) + 1
nrows = len(pivot_data)

ax.set_xlim(0, ncols + 1)
ax.set_ylim(0, nrows + 1)

positions = [0.25] + [i + 1.5 for i in range(ncols - 1)]
columns = ['Bank'] + list(pivot_data.columns)

# Define the colors for normal and lowest values
normal_color = 'black'
lowest_color = 'black'

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        weight = 'normal'
        if j == 0:
            ha = 'left'
            text_label = f'{pivot_data.index[i]}'
            color = normal_color
            # Add bank name
            ax.annotate(
                xy=(positions[j], i + .5),
                text=text_label,
                ha=ha,
                va='center',
                weight=weight,
                fontsize=8,
                color=color
            )
        else:
            ha = 'center'
            # Add interest rate
            rate_value = pivot_data[column].iloc[i]
            days_value = pivot_data_change[column].iloc[i]
            
            if not np.isnan(rate_value):
                # Interest rate display
                if rate_value == pivot_data[column].min():
                    color = lowest_color
                    weight = 'bold'
                else:
                    color = normal_color
                    weight = 'normal'
                
                ax.annotate(
                    xy=(positions[j], i + .6),  # Slightly higher position
                    text=f'{rate_value}',
                    ha=ha,
                    va='center',
                    weight=weight,
                    fontsize=8,
                    color=color
                )
                
                # Days since last change display
                ax.annotate(
                    xy=(positions[j], i + .3),  # Slightly lower position
                    text=f'({int(days_value)})',
                    ha=ha,
                    va='center',
                    weight='normal',
                    fontsize=7,
                    color='gray'
                )

# Add column names
for index, c in enumerate(columns):
    if index == 0:
        ha = 'left'
    else:
        ha = 'center'
    ax.annotate(
        xy=(positions[index], nrows + .25),
        text=columns[index],
        ha=ha,
        va='bottom',
        weight='bold',
        fontsize=7
    )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

plt.title('Domestic Banks Standard Rates - ' + today) 

ax.set_axis_off()
plt.savefig(
    os.path.join(directory,'latest standard rates dom.png'),
    dpi=300,
    transparent=True,
    bbox_inches='tight'
)



df_latest = df_all[(df_all.ProductType.isin(['LVR < 80%'])) & (df_all.LatestDateFlag == 'Yes') & (df_all.Institution.isin(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank']))][['Institution','InterestRateType','InterestRate','date','ProductType']]
df_latest['Institution'] = np.where(df_latest['Institution']=='Co-operative Bank', 'Co-op Bank'
    ,np.where(df_latest['Institution']=='Heartland Bank', 'Heartland'
    ,df_latest['Institution']))
pivot_data = df_latest.pivot(index='Institution', columns='InterestRateType', values='InterestRate')


# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data = pivot_data.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data.sort_index(inplace=True ,ascending=False)

df_latest_change = previous_rates[(previous_rates.ProductType.isin(['LVR < 80%'])) & (previous_rates.Institution.isin(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank']))][['Institution','InterestRateType','LastChangeDate','DaysSinceLastChange']]
df_latest_change['Institution'] = np.where(df_latest_change['Institution']=='Co-operative Bank', 'Co-op Bank'
    ,np.where(df_latest_change['Institution']=='Heartland Bank', 'Heartland'
    ,df_latest_change['Institution']))
pivot_data_change = df_latest_change.pivot(index='Institution', columns='InterestRateType', values='DaysSinceLastChange')

# Define the new order of the columns
new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']

# Reorder the columns
pivot_data_change = pivot_data_change.reindex(new_order, axis=1)

# Sort the DataFrame by the index (Bank)
pivot_data_change.sort_index(inplace=True ,ascending=False)

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10,6), dpi=300)

# Define the number of columns and rows
ncols = len(pivot_data.columns) + 1
nrows = len(pivot_data)

ax.set_xlim(0, ncols + 1)
ax.set_ylim(0, nrows + 1)

positions = [0.25] + [i + 1.5 for i in range(ncols - 1)]
columns = ['Bank'] + list(pivot_data.columns)

# Define the colors for normal and lowest values
normal_color = 'black'
lowest_color = 'black'

# Add table's main text
for i in range(nrows):
    for j, column in enumerate(columns):
        weight = 'normal'
        if j == 0:
            ha = 'left'
            text_label = f'{pivot_data.index[i]}'
            color = normal_color
            # Add bank name
            ax.annotate(
                xy=(positions[j], i + .5),
                text=text_label,
                ha=ha,
                va='center',
                weight=weight,
                fontsize=8,
                color=color
            )
        else:
            ha = 'center'
            # Add interest rate
            rate_value = pivot_data[column].iloc[i]
            days_value = pivot_data_change[column].iloc[i]
            
            if not np.isnan(rate_value):
                # Interest rate display
                if rate_value == pivot_data[column].min():
                    color = lowest_color
                    weight = 'bold'
                else:
                    color = normal_color
                    weight = 'normal'
                
                ax.annotate(
                    xy=(positions[j], i + .6),  # Slightly higher position
                    text=f'{rate_value}',
                    ha=ha,
                    va='center',
                    weight=weight,
                    fontsize=8,
                    color=color
                )
                
                # Days since last change display
                ax.annotate(
                    xy=(positions[j], i + .3),  # Slightly lower position
                    text=f'({int(days_value)})',
                    ha=ha,
                    va='center',
                    weight='normal',
                    fontsize=7,
                    color='gray'
                )

# Add column names
for index, c in enumerate(columns):
    if index == 0:
        ha = 'left'
    else:
        ha = 'center'
    ax.annotate(
        xy=(positions[index], nrows + .25),
        text=columns[index],
        ha=ha,
        va='bottom',
        weight='bold',
        fontsize=7
    )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

plt.title('Domestic Banks Rate requiring LVR < 80% - ' + today) 

ax.set_axis_off()
plt.savefig(
    os.path.join(directory,'latest LVR less 80 rates dom.png'),
    dpi=300,
    transparent=True,
    bbox_inches='tight'
)




import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

df_other = df_all[(df_all.ProductType.isin(['Standard','LVR < 80%'])) & (df_all.Institution.isin(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank']))] [['Institution','InterestRateType','InterestRate','date','ProductType']]
df_other['Institution'] = np.where(df_other['Institution']=='Co-operative Bank', 'Co-op Bank'
    ,np.where(df_other['Institution']=='Heartland Bank', 'Heartland'
    ,df_other['Institution']))
df_other['Institution'] = df_other['Institution'] + np.where(df_other['ProductType']=='LVR < 80%', '*','')
df_other = df_other.loc[:, df_other.columns != 'ProductType']
df_other = df_other.rename(columns={'Institution': 'Bank', 'InterestRateType': 'Product', 'InterestRate': 'Rate', 'date': 'Date'})
df_other['Date'] = pd.to_datetime(df_other['Date'])

month_24t = max(df_other['Date']) - pd.DateOffset(months=24)
month_24 = min(df_other[(df_other['Date']>month_24t)]['Date'])
df_other = df_other[(df_other.Date >= month_24t)]

# Get the unique products
products = df_other['Product'].unique()

# Define the color codes for the banks
color_codes = {
    'SBS Bank': '#FF8C00',
    'SBS Bank*': '#FF8C00',
    'Co-op Bank': '#008000',
    'Co-op Bank*': '#008000',
    'Heartland': '#193487',
    'Heartland*': '#193487',
    'Westpac': '#ff1339',
    'Westpac*': '#ff1339',
    'Kiwibank': '#65cd32',
    'Kiwibank*': '#65cd32',
}

mnratef = min(df_other[(df_other.Product =='Floating')]['Rate']) - 0.1
mxratef = max(df_other[(df_other.Product =='Floating')]['Rate']) + 0.1

mnrate = min(df_other[(df_other.Product !='Floating')]['Rate']) - 0.1
mxrate = max(df_other[(df_other.Product !='Floating')]['Rate']) + 0.1

# For each product, filter data and create a line chart
for product in products:
    product_df = df_other[df_other['Product'] == product]
    product_df['Date'] = pd.to_datetime(product_df['Date'])

    # Pivot the data to have banks as columns
    pivot_df = product_df.pivot(index='Date', columns='Bank', values='Rate')

    # Create a single line chart for all banks
    plt.figure(figsize=(10, 6))
    for bank, data in pivot_df.items():
        if bank[-1]== '*':
            plt.plot(data.index, data.values, label=bank, color=color_codes.get(bank, 'black'), linestyle='dashed')
        else:
            plt.plot(data.index, data.values, label=bank, color=color_codes.get(bank, 'black'))
        
    plt.title(f'{product}', fontsize=20)
    
    
    #tick based on index
    #plt.xticks(ticks=data.index, labels=data.index.strftime('%Y-%m-%d'))
    
    # major tick for every second week  
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

    #format xaxis date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))

    plt.xticks(rotation=90,fontsize=8)
    
    if product == 'Floating':
        plt.ylim(mnratef, mxratef)
    else:
        plt.ylim(mnrate, mxrate)
    
    plt.ylabel('Interest Rate')
    
    plt.legend(loc='center', bbox_to_anchor=(1.05, 0.5),
              ncol=1, fancybox=True, shadow=True)
    plt.gcf().text(0.89, 0.2, '* LVR < 80%', fontsize=10, bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 4})

    # Save the plot as a PNG file in the same directory as the CSV file
    plt.savefig(os.path.join(directory, f'{product}_interest_rates_dom.png'))



from github import Github
import os
import base64

# First create a Github instance using an access token from environment variable
token = os.getenv('GITHUB_TOKEN')
if not token:
    raise ValueError("Please set the GITHUB_TOKEN environment variable")

g = Github(token)
directory = r'C:\Users\User\Documents\GitHub\NZ-Mortgage-Rates'

# Then get a specific repository
repo = g.get_user().get_repo("NZ-Mortgage-Rates")
message = "commit from python test1"
branch = "main"

files = [
    {"name": "latest standard rates.png"},
    {"name": "latest LVR less 80 rates.png"},
    {"name": "Floating_interest_rates.png"},
    {"name": "6 Months_interest_rates.png"},
    {"name": "1 Year_interest_rates.png"},
    {"name": "18 Months_interest_rates.png"},
    {"name": "2 Year_interest_rates.png"},
    {"name": "3 Year_interest_rates.png"},
    {"name": "4 Year_interest_rates.png"},
    {"name": "5 Year_interest_rates.png"},
    {"name": "latest standard rates dom.png"},
    {"name": "latest LVR less 80 rates dom.png"},
    {"name": "Floating_interest_rates_dom.png"},
    {"name": "6 Months_interest_rates_dom.png"},
    {"name": "1 Year_interest_rates_dom.png"},
    {"name": "18 Months_interest_rates_dom.png"},
    {"name": "2 Year_interest_rates_dom.png"},
    {"name": "3 Year_interest_rates_dom.png"},
    {"name": "4 Year_interest_rates_dom.png"},
    {"name": "5 Year_interest_rates_dom.png"}
]

for file_info in files:
    file_name = file_info["name"]
    
    # Open the file in binary mode
    f_open = os.path.join(directory, file_name)
    with open(f_open, "rb") as image:
        f = image.read()
        image_data = bytearray(f)
    
    try:
        contents = repo.get_contents(file_name)
        # Compare local file content with the content in the repository
        if base64.b64encode(image_data).decode('utf-8') != contents.content:
            repo.update_file(file_name, message, bytes(image_data), sha=contents.sha, branch=branch)
        else:
            print(f"No changes detected for {file_name}. Skipping update.")
    except Exception as e:
        print(f"Error updating {file_name}: {e}")
        # Uncomment the following lines if you want to create the file if it doesn't exist
        # try:
        #     repo.create_file(file_name, message, bytes(image_data), branch=branch)
        # except Exception as e:
        #     print(f"Error creating {file_name}: {e}")