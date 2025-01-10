import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv('medical_examination.csv')

# Data Preprocessing
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 1. Draw Categorical Plot
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    print(df_cat.head())
    # 6
    df_cat = df_cat.rename(columns={'variable': 'feature', 'value': 'result'})
    
    # 7
    df_cat = df_cat.groupby(['cardio', 'feature', 'result']).size().reset_index(name='count')

    # 8
    fig = sns.catplot(x='feature', hue='result', col='cardio', data=df_cat, kind='count')

    # Set the xlabel to 'variable'
    fig.set_axis_labels('variable', 'total')
    # Save plot
    fig.savefig('catplot.png')

    return fig.fig  # Return the figure for further manipulation or tests

# 2. Draw Heatmap
def draw_heat_map():
    # Apply filtering to the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Calculate correlation matrix
    corr = df_heat.corr()

    # Create a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create the plot with seaborn heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', ax=ax)

    # Save the heatmap plot
    fig.savefig('heatmap.png')

    return fig  # Return the figure for further manipulation or tests
