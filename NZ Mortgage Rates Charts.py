import pandas as pd 
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Directory of output
directory = r'C:\Users\User\Documents\GitHub\NZ-Mortgage-Rates'
today = date.today().strftime("%d/%m/%Y")

df = pd.read_csv(r'X:\InterestRates.csv', parse_dates=['time_stamp'], dayfirst=True)
date1 = pd.to_datetime(df['time_stamp'], errors='coerce', dayfirst=True, format="%d/%m/%Y %H:%M")
date2 = pd.to_datetime(df['time_stamp'], errors='coerce', format="%Y-%m-%d %H:%M:%S")
df['date'] = date1.fillna(date2)

interest_rate_types = {
    'Floating': 'Floating',
    'Months6': '6 Months',
    'Year1': '1 Year',
    'Months18': '18 Months',
    'Year2': '2 Year',
    'Year3': '3 Year',
    'Year4': '4 Year',
    'Year5': '5 Year'
}

df_all = pd.DataFrame()
for col, rate_type in interest_rate_types.items():
    temp_df = df[df[col] > 0][['Institution', 'Product', col, 'date']]
    temp_df['InterestRateType'] = rate_type
    temp_df.rename(columns={col: 'InterestRate'}, inplace=True)
    df_all = pd.concat([df_all, temp_df])

df_all['ReportProduct'] = df_all['Institution'].str.cat(df_all['Product'], sep='-')

bank_type_mapping = {
    'ASB': 'Major', 'ANZ': 'Major', 'BNZ': 'Major', 'Westpac': 'Major',
    'Kiwibank': 'Minor', 'HSBC': 'Minor', 'TSB Bank': 'Minor',
    'Heartland Bank': 'Minor', 'Co-operative Bank': 'Minor'
}
df_all['BankType'] = df_all['Institution'].map(bank_type_mapping).fillna('Other')

product_type_mapping = {
    ('ASB', 'Standard'): 'LVR < 80%', ('BNZ', 'Standard'): 'LVR < 80%',
    ('Bank of China', 'Special'): 'LVR < 80%', ('China Construction Bank', 'Non Standard'): 'Standard',
    ('China Construction Bank', 'Standard'): 'LVR < 80%', ('Kiwibank', 'Standard'): 'Standard',
    ('Kiwibank', 'Special'): 'LVR < 80%', ('SBS Bank', 'Residential'): 'Standard',
    ('SBS Bank', 'Special'): 'LVR < 80%', ('Co-operative Bank', 'Standard'): 'Standard',
    ('Co-operative Bank', 'Owner Occupied'): 'LVR < 80%', ('Heartland Bank', 'Residential'): 'LVR < 80%',
    ('TSB Bank', 'Standard'): 'Standard', ('TSB Bank', 'Special LVR <80%'): 'LVR < 80%',
    'Better Homes Top Up': 'Other', 'Blueprint to Build': 'Construction', 'Choices Everyday': 'Revolving',
    'Choices Offset': 'Offset', 'Construction lending for FHB': 'First Home Buyer',
    'First Home Buyer Special': 'First Home Buyer', 'First Home Combo': 'First Home Buyer',
    'Good Energy - Up to $80K': 'Green', 'Green Home Loan top-ups': 'Green', 'Non Standard': 'Special',
    'Offset Mortgage': 'Offset', 'Owner Occupied': 'Special', 'Premier': 'Standard',
    'Residential': 'Standard', 'Reverse Mortgage': 'Reverse', 'Special': 'Special',
    'Special - Classic': 'LVR < 80%', 'Special LVR < 80%': 'LVR < 80%', 'Special LVR <80%': 'LVR < 80%',
    'Special LVR under 80%': 'LVR < 80%', 'Standard': 'Standard', 'Std & Flybuys': 'Standard',
    'TotalMoney': 'Offset', 'Unwind reverse equity': 'Reverse'
}
df_all['ProductType'] = df_all.apply(lambda row: product_type_mapping.get((row['Institution'], row['Product']), product_type_mapping.get(row['Product'], 'Other')), axis=1)

df_all = df_all[['ReportProduct', 'BankType', 'ProductType', 'Institution', 'Product', 'InterestRateType', 'InterestRate', 'date']]

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


import pandas as pd 
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


def create_rates_table_image(
    institutions,
    product_type,
    output_filename,
    logo_paths=None
):
    """
    Generate a styled rates table image for a given list of institutions and product type.
    Args:
        institutions (list): List of bank names (e.g., ['ANZ', 'ASB', ...])
        product_type (str or list): ProductType(s) to filter (e.g., 'Standard' or ['Standard', 'LVR < 80%'])
        output_filename (str): Output PNG file name (full path or relative to directory)
        logo_paths (dict): Optional. Dict mapping bank name to {'path': ..., 'zoom': ...}
    """
    import matplotlib.patches as patches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg
    import numpy as np
    import os

    out_path = output_filename if os.path.isabs(output_filename) else os.path.join(directory, output_filename)

    default_logo_paths = {
        'ANZ':      {'path': os.path.join(directory, 'logos', 'ANZ.png'),      'zoom': 0.07},
        'ASB':      {'path': os.path.join(directory, 'logos', 'ASB.png'),      'zoom': 0.07},
        'BNZ':      {'path': os.path.join(directory, 'logos', 'BNZ.png'),      'zoom': 0.08},
        'Kiwibank': {'path': os.path.join(directory, 'logos', 'Kiwibank.png'), 'zoom': 0.15},
        'Westpac':  {'path': os.path.join(directory, 'logos', 'Westpac.png'),  'zoom': 0.05},
        'SBS Bank':  {'path': os.path.join(directory, 'logos', 'SBS Bank.png'),  'zoom': 0.35},
        'Co-operative Bank':  {'path': os.path.join(directory, 'logos', 'Co-operative Bank.png'),  'zoom': 0.4},
        'Heartland Bank':  {'path': os.path.join(directory, 'logos', 'Heartland Bank.png'),  'zoom': 0.04},
        'TSB Bank':  {'path': os.path.join(directory, 'logos', 'TSB Bank.png'),  'zoom': 0.4}
    }
    logo_paths = logo_paths or default_logo_paths

    df_latest = df_all[(df_all.ProductType.isin([product_type] if isinstance(product_type, str) else product_type)) &
                       (df_all.LatestDateFlag == 'Yes') &
                       (df_all.Institution.isin(institutions))][['Institution','InterestRateType','InterestRate','date','ProductType']]
    pivot_data = df_latest.pivot(index='Institution', columns='InterestRateType', values='InterestRate')

    new_order = ['6 Months', '1 Year', '18 Months', '2 Year', '3 Year', '4 Year', '5 Year', 'Floating']
    # Add missing institutions as rows of NaN
    for inst in institutions:
        if inst not in pivot_data.index:
            pivot_data.loc[inst] = np.nan
    # Sort by bank name ascending (A-Z)
    pivot_data = pivot_data.sort_index(ascending=False)

    df_latest_change = previous_rates[(previous_rates.ProductType.isin([product_type] if isinstance(product_type, str) else product_type)) &
                                      (previous_rates.Institution.isin(institutions))][['Institution','InterestRateType','LastChangeDate','DaysSinceLastChange']]
    pivot_data_change = df_latest_change.pivot(index='Institution', columns='InterestRateType', values='DaysSinceLastChange')
    pivot_data_change = pivot_data_change.reindex(new_order, axis=1)
    for inst in institutions:
        if inst not in pivot_data_change.index:
            pivot_data_change.loc[inst] = np.nan
    # Sort by bank name ascending (A-Z)
    pivot_data_change = pivot_data_change.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ncols = len(pivot_data.columns) + 1
    nrows = len(pivot_data)
    ax.set_xlim(0, ncols + 1)
    ax.set_ylim(0, nrows + 1)
    positions = [0.25] + [i + 1.5 for i in range(ncols - 1)]
    columns = ['Bank'] + list(pivot_data.columns)
    header_bg = "#ffffff"
    row_alt_bg = '#ffffff'
    lowest_color = '#2ecc40'
    normal_color = '#222'
    header_font = {'weight': 'bold', 'fontsize': 10, 'fontname': 'DejaVu Sans'}
    cell_font = {'fontsize': 9, 'fontname': 'DejaVu Sans'}
    ax.add_patch(patches.FancyBboxPatch((0, 0), ncols + 1, nrows + 1, boxstyle="round,pad=0.02", linewidth=0, facecolor='white', zorder=0))
    for i in range(nrows):
        if i % 2 == 1:
            ax.add_patch(patches.Rectangle((0, i), ncols + 1, 1, color=row_alt_bg, zorder=1))
    for i in range(nrows):
        for j, column in enumerate(columns):
            if j == 0:
                bank_name = str(pivot_data.index[i]).replace('*', '')
                logo_info = logo_paths.get(bank_name)
                if logo_info:
                    try:
                        arr_img = mpimg.imread(logo_info['path'])
                        imagebox = OffsetImage(arr_img, zoom=logo_info['zoom'])
                        ab = AnnotationBbox(imagebox, (positions[j], i + .5), frameon=False, box_alignment=(0,0.5))
                        ax.add_artist(ab)
                    except Exception as e:
                        ax.annotate(
                            bank_name,
                            xy=(positions[j], i + .5),
                            ha='left', va='center',
                            color=normal_color,
                            **cell_font
                        )
                else:
                    ax.annotate(
                        bank_name,
                        xy=(positions[j], i + .5),
                        ha='left', va='center',
                        color=normal_color,
                        **cell_font
                    )
            else:
                ha = 'center'
                try:
                    rate_value = pivot_data[column].iloc[i]
                except (IndexError, KeyError):
                    rate_value = np.nan
                try:
                    days_value = pivot_data_change[column].iloc[i]
                except (IndexError, KeyError):
                    days_value = np.nan
                if not np.isnan(rate_value):
                    if rate_value == pivot_data[column].min():
                        color = lowest_color
                        weight = 'bold'
                    else:
                        color = normal_color
                        weight = 'normal'
                    ax.annotate(
                        f'{rate_value}',
                        xy=(positions[j], i + .6),
                        ha=ha, va='center',
                        color=color,
                        weight=weight,
                        **cell_font
                    )
                    if not np.isnan(days_value):
                        ax.annotate(
                            f'({int(days_value)})',
                            xy=(positions[j], i + .3),
                            ha=ha, va='center',
                            color='#888',
                            fontsize=8,
                            fontname='DejaVu Sans'
                        )
    for index, c in enumerate(columns):
        ax.add_patch(patches.Rectangle((positions[index] - 0.4, nrows), 0.8, 0.7, color=header_bg, zorder=2))
        ha = 'left' if index == 0 else 'center'
        ax.annotate(
            columns[index],
            xy=(positions[index], nrows + .25),
            ha=ha, va='bottom',
            color=normal_color,
            **header_font
        )
    ax.set_axis_off()
    # Set plot title to output_filename without extension
    title_base = os.path.splitext(os.path.basename(output_filename))[0]
    plt.title(f"{title_base} - {today}", fontsize=14, fontweight='bold', fontname='DejaVu Sans', color='#222', pad=20)
    plt.savefig(
        out_path,
        dpi=300,
        transparent=True,
        bbox_inches='tight'
    )
    plt.close(fig)


create_rates_table_image(['ANZ','ASB','BNZ','Westpac','Kiwibank'], 'Standard', 'Big 5 Banks - Standard Rates.png')
create_rates_table_image(['ANZ','ASB','BNZ','Westpac','Kiwibank'], 'LVR < 80%', 'Big 5 Banks - Special Rates.png')
create_rates_table_image(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank'], 'Standard', 'Domestic Banks - Standard Rates.png')
create_rates_table_image(['Kiwibank','SBS Bank','Co-operative Bank','Heartland Bank','TSB Bank'], 'LVR < 80%', 'Domestic Banks - Special Rates.png')


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
    {"name": "Big 5 Banks - Standard Rates.png"},
    {"name": "Big 5 Banks - Special Rates.png"},
    {"name": "Floating_interest_rates.png"},
    {"name": "6 Months_interest_rates.png"},
    {"name": "1 Year_interest_rates.png"},
    {"name": "18 Months_interest_rates.png"},
    {"name": "2 Year_interest_rates.png"},
    {"name": "3 Year_interest_rates.png"},
    {"name": "4 Year_interest_rates.png"},
    {"name": "5 Year_interest_rates.png"},
    {"name": "Domestic Banks - Standard Rates.png"},
    {"name": "Domestic Banks - Special Rates.png"},
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

