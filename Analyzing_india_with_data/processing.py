import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create output directory for saving results
OUTPUT_DIR = 'telecom_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for better visualizations
try:
    plt.style.use('ggplot')
except:
    pass

# Configure seaborn
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Define colors
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"]

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load the dataset with the correct path
file_path = 'Public_Vs_Private_Yearly_And_Montly_Basis_Apr-2009_Feb-2015.csv'
df = pd.read_csv(file_path)

# Save basic data information to file
with open(os.path.join(OUTPUT_DIR, 'data_info.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Initial data shape: {df.shape}\n\n")
    f.write("Checking for missing values:\n")
    f.write(f"{df.isnull().sum()}\n\n")
    f.write("First 5 rows of the dataset:\n")
    f.write(f"{df.head().to_string()}\n\n")
    f.write("Basic statistics of the numerical columns:\n")
    f.write(f"{df.describe().to_string()}\n\n")
    f.write("Unique month values in the dataset:\n")
    f.write(f"{df['Month'].unique()}\n\n")

# Extract year from the 'Year ending 31st March' column
df['Year'] = df['Year ending\n 31st March'].str.extract(r'(\d{4})')

# Extract month names and years from the format like "January'11"
df['Month_Name'] = df['Month'].str.replace(r"'.*$", "", regex=True)  # Remove anything after the apostrophe
df['Month_Year'] = df['Month'].str.extract(r"'(\d+)", expand=False)  # Extract the year digits after the apostrophe

# Create a mapping for month names to numeric values
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 
    'May': 5, 'June': 6, 'July': 7, 'August': 8, 
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Map to get month numbers as integers (not floats)
df['Month_Num'] = df['Month_Name'].map(month_mapping).astype(int)

# Log month mapping results
with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'w', encoding='utf-8') as f:
    f.write("Month numbers after mapping:\n")
    f.write(f"{df[['Month', 'Month_Name', 'Month_Num']].head(10).to_string()}\n\n")
    f.write(f"Month number data type: {df['Month_Num'].dtype}\n\n")

# Handle the fiscal year logic
df['Year_Num'] = df['Year'].astype(int)
df['Year_Num'] = np.where(df['Month_Num'] <= 3, df['Year_Num'], df['Year_Num'] - 1)

# Create date strings
df['Date_Str'] = df['Year_Num'].astype(str) + '-' + df['Month_Num'].astype(str) + '-01'

# Log date components
with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
    f.write("Date components before datetime conversion:\n")
    f.write(f"{df[['Year_Num', 'Month_Num']].head(10).to_string()}\n\n")
    f.write("Date strings created:\n")
    f.write(f"{df['Date_Str'].head(10).to_string()}\n\n")

# Convert to datetime with error handling
try:
    df['Date'] = pd.to_datetime(df['Date_Str'])
    with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
        f.write("Date conversion successful!\n")
except Exception as e:
    with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Error in date conversion: {e}\n")
        f.write("Trying fallback approach...\n")
    try:
        df['Date'] = pd.to_datetime(df['Year_Num'].astype(str) + '-' + df['Month_Num'].astype(str).str.replace('.0', '', regex=False) + '-01')
        with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
            f.write("Fallback approach successful!\n")
    except Exception as e2:
        with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
            f.write(f"Fallback approach also failed: {e2}\n")
            f.write("Creating dates manually...\n")
        # Last resort: create dates manually
        from datetime import datetime
        dates = []
        for _, row in df.iterrows():
            try:
                dates.append(datetime(row['Year_Num'], row['Month_Num'], 1))
            except:
                dates.append(None)
        df['Date'] = dates
        with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
            f.write("Manual date creation completed.\n")

# Sort data by date
df = df.sort_values('Date')

with open(os.path.join(OUTPUT_DIR, 'date_processing.txt'), 'a', encoding='utf-8') as f:
    f.write(f"\nDate range of the dataset: {df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}\n")
    f.write("\nSuccessfully processed the dates!\n")

# Create a function for formatted numbers
def format_number(num):
    """Format large numbers for better readability"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

# 1. Time Series Analysis of Telephone Numbers
plt.figure(figsize=(16, 8))
plt.plot(df['Date'], df['Telephones - Public'] / 1e6, color=colors[0], linewidth=2.5, label='Public')
plt.plot(df['Date'], df['Telephones - Private'] / 1e6, color=colors[1], linewidth=2.5, label='Private')
plt.plot(df['Date'], df['Telephones - Total'] / 1e6, color=colors[2], linewidth=2.5, label='Total')

plt.title('Telephone Numbers Over Time (in Millions)', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=14, labelpad=10)
plt.ylabel('Number of Telephones (Millions)', fontsize=14, labelpad=10)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Annotate important points
max_private_idx = df['Telephones - Private'].idxmax()
plt.annotate(f"Peak Private: {format_number(df.loc[max_private_idx, 'Telephones - Private'])}",
             xy=(df.loc[max_private_idx, 'Date'], df.loc[max_private_idx, 'Telephones - Private'] / 1e6),
             xytext=(15, 15), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'telephone_numbers_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close instead of show

# 2. Market Share Analysis
plt.figure(figsize=(16, 8))
plt.stackplot(df['Date'], df['Share% age - Public'], df['Share% age - Private'], 
              labels=['Public Share', 'Private Share'],
              colors=[colors[0], colors[1]], alpha=0.7)

plt.title('Market Share Percentage Over Time', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=14, labelpad=10)
plt.ylabel('Share Percentage (%)', fontsize=14, labelpad=10)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Calculate the crossover point (if any)
for i in range(1, len(df)):
    if ((df['Share% age - Public'].iloc[i-1] > df['Share% age - Private'].iloc[i-1] and
         df['Share% age - Public'].iloc[i] < df['Share% age - Private'].iloc[i]) or
        (df['Share% age - Public'].iloc[i-1] < df['Share% age - Private'].iloc[i-1] and
         df['Share% age - Public'].iloc[i] > df['Share% age - Private'].iloc[i])):
        plt.axvline(x=df['Date'].iloc[i], color='red', linestyle='--', alpha=0.5)
        plt.annotate(f"Market Share Crossover: {df['Date'].iloc[i].strftime('%b %Y')}",
                     xy=(df['Date'].iloc[i], 50),
                     xytext=(15, 15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'market_share_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Growth Rate Comparison
plt.figure(figsize=(16, 8))
plt.plot(df['Date'], df['% age Growth - Public'], color=colors[0], linewidth=2.5, label='Public Growth')
plt.plot(df['Date'], df['% age Growth - Private'], color=colors[1], linewidth=2.5, label='Private Growth')

plt.title('Monthly Growth Rate Comparison', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=14, labelpad=10)
plt.ylabel('Growth Rate (%)', fontsize=14, labelpad=10)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add horizontal line at 0%
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Annotate max growth points
max_public_growth_idx = df['% age Growth - Public'].idxmax()
max_private_growth_idx = df['% age Growth - Private'].idxmax()

plt.annotate(f"Max Public Growth: {df.loc[max_public_growth_idx, '% age Growth - Public']:.1f}%",
             xy=(df.loc[max_public_growth_idx, 'Date'], df.loc[max_public_growth_idx, '% age Growth - Public']),
             xytext=(15, 15), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.annotate(f"Max Private Growth: {df.loc[max_private_growth_idx, '% age Growth - Private']:.1f}%",
             xy=(df.loc[max_private_growth_idx, 'Date'], df.loc[max_private_growth_idx, '% age Growth - Private']),
             xytext=(15, 15), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'growth_rate_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Monthly Additions Analysis
plt.figure(figsize=(16, 8))

# Use bar plot for additions to better visualize monthly changes
width = 15
plt.bar(df['Date'], df['Additions - Public'], width=width, color=colors[0], label='Public Additions')
plt.bar(df['Date'], df['Additions - Private'], width=width, color=colors[1], label='Private Additions', bottom=df['Additions - Public'])

plt.title('Monthly Additions in Telephone Numbers', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=14, labelpad=10)
plt.ylabel('Number of Additions', fontsize=14, labelpad=10)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add a horizontal line at 0
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Annotate significant points
max_total_addition_idx = (df['Additions - Public'] + df['Additions - Private']).idxmax()

plt.annotate(f"Highest Total Additions: {format_number(df.loc[max_total_addition_idx, 'Additions - Public'] + df.loc[max_total_addition_idx, 'Additions - Private'])}",
             xy=(df.loc[max_total_addition_idx, 'Date'], df.loc[max_total_addition_idx, 'Additions - Public'] + df.loc[max_total_addition_idx, 'Additions - Private']),
             xytext=(15, 15), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_additions.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Yearly Analysis
# Aggregate data by year
yearly_df = df.groupby(df['Date'].dt.year).agg({
    'Telephones - Public': 'mean',
    'Telephones - Private': 'mean',
    'Telephones - Total': 'mean',
    'Additions - Public': 'sum',
    'Additions - Private': 'sum',
    'Share% age - Public': 'mean',
    'Share% age - Private': 'mean',
    '% age Growth - Public': 'mean',
    '% age Growth - Private': 'mean'
}).reset_index()
yearly_df.rename(columns={'Date': 'Year'}, inplace=True)

# Save yearly summary to CSV
yearly_df.to_csv(os.path.join(OUTPUT_DIR, 'yearly_summary.csv'), index=False)

# Create a subplot with 2 charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Yearly telephone numbers
bar_width = 0.35
x = np.arange(len(yearly_df))
ax1.bar(x - bar_width/2, yearly_df['Telephones - Public'] / 1e6, bar_width, color=colors[0], label='Public')
ax1.bar(x + bar_width/2, yearly_df['Telephones - Private'] / 1e6, bar_width, color=colors[1], label='Private')

ax1.set_title('Yearly Average Telephone Numbers (in Millions)', fontsize=16, pad=20)
ax1.set_xlabel('Year', fontsize=14, labelpad=10)
ax1.set_ylabel('Average Telephones (Millions)', fontsize=14, labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(yearly_df['Year'])
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Yearly growth rates
ax2.bar(x - bar_width/2, yearly_df['% age Growth - Public'], bar_width, color=colors[0], label='Public Growth')
ax2.bar(x + bar_width/2, yearly_df['% age Growth - Private'], bar_width, color=colors[1], label='Private Growth')

ax2.set_title('Yearly Average Growth Rates', fontsize=16, pad=20)
ax2.set_xlabel('Year', fontsize=14, labelpad=10)
ax2.set_ylabel('Average Growth Rate (%)', fontsize=14, labelpad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(yearly_df['Year'])
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Add a horizontal line at 0% growth
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'yearly_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Correlation Analysis
correlation_columns = [
    'Telephones - Public', 'Telephones - Private', 'Telephones - Total',
    'Additions - Public', 'Additions - Private',
    'Share% age - Public', 'Share% age - Private',
    '% age Growth - Public', '% age Growth - Private'
]

correlation_matrix = df[correlation_columns].corr()
# Save correlation matrix to CSV
correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Telephone Data', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Market Share Evolution - Pie Charts at three time points
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Function to create pie chart
def create_share_pie(ax, data_row, title):
    shares = [data_row['Share% age - Public'], data_row['Share% age - Private']]
    ax.pie(shares, labels=['Public', 'Private'], autopct='%1.1f%%', startangle=90,
           colors=[colors[0], colors[1]], wedgeprops=dict(width=0.5))
    ax.set_title(title, fontsize=14, pad=20)

# Start period
start_idx = 0
create_share_pie(axes[0], df.iloc[start_idx], 
                 f"Market Share at Start\n({df['Date'].iloc[start_idx].strftime('%b %Y')})")

# Middle period
mid_idx = len(df) // 2
create_share_pie(axes[1], df.iloc[mid_idx], 
                 f"Market Share at Middle\n({df['Date'].iloc[mid_idx].strftime('%b %Y')})")

# End period
end_idx = len(df) - 1
create_share_pie(axes[2], df.iloc[end_idx], 
                 f"Market Share at End\n({df['Date'].iloc[end_idx].strftime('%b %Y')})")

plt.suptitle('Evolution of Market Share Distribution', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'market_share_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. Seasonal Analysis - Monthly Patterns
# Create a month column for grouping
df['Month_Name_Full'] = df['Date'].dt.month_name()

# Group by month name
monthly_df = df.groupby('Month_Name_Full').agg({
    'Additions - Public': 'mean',
    'Additions - Private': 'mean',
    '% age Growth - Public': 'mean',
    '% age Growth - Private': 'mean'
}).reset_index()

# Save monthly analysis to CSV
monthly_df.to_csv(os.path.join(OUTPUT_DIR, 'monthly_analysis.csv'), index=False)

# Reorder months chronologically
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_df['Month_Name_Full'] = pd.Categorical(monthly_df['Month_Name_Full'], categories=month_order, ordered=True)
monthly_df = monthly_df.sort_values('Month_Name_Full')

# Create a subplot with 2 charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Monthly average additions
ax1.bar(monthly_df['Month_Name_Full'], monthly_df['Additions - Public'], color=colors[0], label='Public')
ax1.bar(monthly_df['Month_Name_Full'], monthly_df['Additions - Private'], bottom=monthly_df['Additions - Public'],
        color=colors[1], label='Private')

ax1.set_title('Average Monthly Additions by Month', fontsize=16, pad=20)
ax1.set_xlabel('Month', fontsize=14, labelpad=10)
ax1.set_ylabel('Average Additions', fontsize=14, labelpad=10)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Monthly average growth
ax2.plot(monthly_df['Month_Name_Full'], monthly_df['% age Growth - Public'], 
         color=colors[0], marker='o', linewidth=2.5, label='Public Growth')
ax2.plot(monthly_df['Month_Name_Full'], monthly_df['% age Growth - Private'], 
         color=colors[1], marker='o', linewidth=2.5, label='Private Growth')

ax2.set_title('Average Monthly Growth Rate by Month', fontsize=16, pad=20)
ax2.set_xlabel('Month', fontsize=14, labelpad=10)
ax2.set_ylabel('Average Growth Rate (%)', fontsize=14, labelpad=10)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seasonal_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9. Interactive visualization with Plotly (time series with range slider)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Telephones - Public']/1e6,
    mode='lines',
    name='Public Telephones',
    line=dict(color=colors[0], width=2)
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Telephones - Private']/1e6,
    mode='lines',
    name='Private Telephones',
    line=dict(color=colors[1], width=2)
))

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Telephones - Total']/1e6,
    mode='lines',
    name='Total Telephones',
    line=dict(color=colors[2], width=2)
))

fig.update_layout(
    title='Interactive Telephone Numbers Over Time (Millions)',
    xaxis_title='Date',
    yaxis_title='Number of Telephones (Millions)',
    legend_title='Category',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Save interactive plot
fig.write_html(os.path.join(OUTPUT_DIR, 'interactive_time_series.html'))

# 10. Key Insights and Trend Analysis
# Calculate key metrics
start_date = df['Date'].iloc[0]
end_date = df['Date'].iloc[-1]
time_period = f"{start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}"

public_start = df['Telephones - Public'].iloc[0]
public_end = df['Telephones - Public'].iloc[-1]
public_change = public_end - public_start
public_percent_change = (public_change / public_start) * 100

private_start = df['Telephones - Private'].iloc[0]
private_end = df['Telephones - Private'].iloc[-1]
private_change = private_end - private_start
private_percent_change = (private_change / private_start) * 100

public_share_start = df['Share% age - Public'].iloc[0]
public_share_end = df['Share% age - Public'].iloc[-1]
public_share_change = public_share_end - public_share_start

private_share_start = df['Share% age - Private'].iloc[0]
private_share_end = df['Share% age - Private'].iloc[-1]
private_share_change = private_share_end - private_share_start

avg_public_growth = df['% age Growth - Public'].mean()
avg_private_growth = df['% age Growth - Private'].mean()

# Save insights to file
with open(os.path.join(OUTPUT_DIR, 'key_insights.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(" "*30 + "KEY INSIGHTS\n")
    f.write("="*80 + "\n")
    f.write(f"\nTime Period: {time_period} ({len(df)} months)\n")
    f.write("\n1. Overall Growth:\n")
    f.write(f"   - Public telephones increased from {public_start:,.0f} to {public_end:,.0f}\n")
    f.write(f"   - Change: +{public_change:,.0f} ({public_percent_change:.2f}%)\n")
    f.write(f"   - Private telephones increased from {private_start:,.0f} to {private_end:,.0f}\n")
    f.write(f"   - Change: +{private_change:,.0f} ({private_percent_change:.2f}%)\n")

    f.write("\n2. Market Share Shift:\n")
    f.write(f"   - Public sector: {public_share_start:.2f}% -> {public_share_end:.2f}% ({public_share_change:+.2f}%)\n")
    f.write(f"   - Private sector: {private_share_start:.2f}% -> {private_share_end:.2f}% ({private_share_change:+.2f}%)\n")

    f.write("\n3. Growth Rates:\n")
    f.write(f"   - Average monthly growth rate (Public): {avg_public_growth:.2f}%\n")
    f.write(f"   - Average monthly growth rate (Private): {avg_private_growth:.2f}%\n")

    f.write("\n4. Key Correlations:\n")
    f.write(f"   - Correlation between Public and Private telephones: {correlation_matrix.loc['Telephones - Public', 'Telephones - Private']:.2f}\n")
    f.write(f"   - Correlation between Public and Private growth rates: {correlation_matrix.loc['% age Growth - Public', '% age Growth - Private']:.2f}\n")

    # Identify best performing months
    best_public_month = monthly_df.loc[monthly_df['% age Growth - Public'].idxmax(), 'Month_Name_Full']
    best_private_month = monthly_df.loc[monthly_df['% age Growth - Private'].idxmax(), 'Month_Name_Full']
    f.write("\n5. Monthly Patterns:\n")
    f.write(f"   - Best month for public sector growth: {best_public_month}\n")
    f.write(f"   - Best month for private sector growth: {best_private_month}\n")

    f.write("\n" + "="*80 + "\n")
    f.write(" "*30 + "CONCLUSIONS\n")
    f.write("="*80 + "\n")

    f.write(f"\n1. {'Private' if private_percent_change > public_percent_change else 'Public'} telephones showed stronger overall growth during the analysis period.\n")
    f.write(f"   - Private growth: {private_percent_change:.2f}%\n")
    f.write(f"   - Public growth: {public_percent_change:.2f}%\n")

    f.write(f"\n2. Market share has shifted towards the {'private' if private_share_change > 0 else 'public'} sector.\n")
    f.write(f"   - Public sector share change: {public_share_change:+.2f}%\n")
    f.write(f"   - Private sector share change: {private_share_change:+.2f}%\n")

    f.write(f"\n3. Average monthly growth was {'higher' if avg_private_growth > avg_public_growth else 'lower'} in the private sector compared to the public sector.\n")

    # Identify trends and growth patterns
    if private_percent_change > public_percent_change:
        f.write("\n4. The private sector is growing faster than the public sector, suggesting a shift towards privatization.\n")
        f.write("   - This trend indicates increasing competition in the telecommunications industry.\n")
    else:
        f.write("\n4. The public sector is maintaining strong growth despite private competition.\n")
        f.write("   - This suggests effective government policies or investments in public telecommunications infrastructure.\n")

    f.write("\n5. Business Implications:\n")
    f.write("   - The telecommunications market is showing steady growth overall.\n")
    f.write(f"   - The {'private' if private_share_change > 0 else 'public'} sector is gaining market share.\n")
    f.write("   - Understanding seasonal patterns can help with strategic marketing and infrastructure planning.\n")

# Create a summary report in HTML format
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Telecom Analysis Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .highlight {{
            font-weight: bold;
            color: #0066cc;
        }}
    </style>
</head>
<body>
    <h1>Public vs Private Telephones in India (2009-2015) Analysis Report</h1>
    
    <h2>Overview</h2>
    <p>This report analyzes the trends and patterns in the Indian telecommunications sector between {time_period}, focusing on the comparison between public and private telephone services.</p>
    
    <h2>Key Findings</h2>
    <ul>
        <li><span class="highlight">Overall Growth:</span> Private telephones increased by {private_percent_change:.2f}%, while public telephones grew by {public_percent_change:.2f}%.</li>
        <li><span class="highlight">Market Share:</span> The private sector {"gained" if private_share_change > 0 else "lost"} {abs(private_share_change):.2f}% market share during this period.</li>
        <li><span class="highlight">Growth Rates:</span> Average monthly growth rates were {avg_public_growth:.2f}% for public and {avg_private_growth:.2f}% for private sectors.</li>
    </ul>
    
    <h2>Visualizations</h2>
    
    <div class="image-container">
        <h3>Telephone Numbers Over Time</h3>
        <p>This chart shows the growth in telephone numbers for both sectors.</p>
        <img src="telephone_numbers_over_time.png" alt="Telephone Numbers Over Time">
    </div>
    
    <div class="image-container">
        <h3>Market Share Analysis</h3>
        <p>This visualization illustrates how market share has evolved between public and private sectors.</p>
        <img src="market_share_analysis.png" alt="Market Share Analysis">
    </div>
    
    <div class="image-container">
        <h3>Growth Rate Comparison</h3>
        <p>Monthly growth rates for public and private sectors show interesting patterns.</p>
        <img src="growth_rate_comparison.png" alt="Growth Rate Comparison">
    </div>
    
    <div class="image-container">
        <h3>Monthly Additions</h3>
        <p>This chart displays the number of new connections added each month by sector.</p>
        <img src="monthly_additions.png" alt="Monthly Additions">
    </div>
    
    <div class="image-container">
        <h3>Yearly Analysis</h3>
        <p>Year-by-year comparison of telephone numbers and growth rates.</p>
        <img src="yearly_analysis.png" alt="Yearly Analysis">
    </div>
    
    <div class="image-container">
        <h3>Correlation Matrix</h3>
        <p>This heatmap shows correlations between different metrics in the dataset.</p>
        <img src="correlation_matrix.png" alt="Correlation Matrix">
    </div>
    
    <div class="image-container">
        <h3>Market Share Evolution</h3>
        <p>These pie charts show the market share distribution at the beginning, middle, and end of the analysis period.</p>
        <img src="market_share_evolution.png" alt="Market Share Evolution">
    </div>
    
    <div class="image-container">
        <h3>Seasonal Analysis</h3>
        <p>Monthly patterns in additions and growth rates reveal seasonal trends.</p>
        <img src="seasonal_analysis.png" alt="Seasonal Analysis">
    </div>
    
    <h2>Conclusions</h2>
    <p>The telecommunications market in India showed significant growth during the analysis period, with the private sector demonstrating stronger performance and gaining market share. This suggests a trend toward privatization and increased competition in the industry.</p>
    
    <p>The analysis reveals clear seasonal patterns in growth rates, which can be valuable for strategic planning. The correlation between public and private sector metrics indicates that while they compete for market share, they are influenced by similar external factors.</p>
    
    <p>For a more detailed analysis and interactive visualizations, refer to the additional files in the results directory.</p>
    
    <h2>Interactive Visualization</h2>
    <p>An interactive time series chart is available in the <a href="interactive_time_series.html">interactive dashboard</a>.</p>
    
    <footer>
        <p><em>Analysis completed on {datetime.now().strftime('%Y-%m-%d')}</em></p>
    </footer>
</body>
</html>
"""

# Save HTML report with UTF-8 encoding
with open(os.path.join(OUTPUT_DIR, 'telecom_analysis_report.html'), 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Analysis completed. All results saved to {os.path.abspath(OUTPUT_DIR)} directory.")