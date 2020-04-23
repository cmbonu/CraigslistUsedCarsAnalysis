import pandas as pd
import math
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from eda_utils import *
#Configure pandas
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 50

## Define various processing lists
bucketize_columns = ['car_age','odometer']
cat_columns = ['manufacturer','condition', 'cylinders', 'fuel','title_status',\
               'transmission', 'drive', 'size', 'type', 'paint_color','state',\
               'car_age_bckt','odometer_bckt']
high_cardinality_cols = ['model','region']
odo_bins=[0, 5000, 30000, 60000, 100000, 150000, 200000,400000,5000000]
#grp_cols = ['fuel','transmission','type','manufacturer']
agg_list = ['count','sum','mean','std']
cat_columns_agg = ['manufacturer','condition', 'cylinders', 'fuel',\
               'transmission', 'drive','type', 'odometer_bckt',\
               ]

## Used Cars custom code
# Barplots
def create_barplot_figure(dframe,plot_columns=None):
    print(f'6. === Bar Plots : {dframe.shape}')
    plot_columns = plot_columns if plot_columns else dframe.columns
    bar_cols = set(plot_columns).intersection(cat_columns)
    #bar_cols = set(dframe.columns).intersection(cat_columns)
    fig = make_subplots(cols=1, rows=len(bar_cols),subplot_titles=list(bar_cols))
    for i,col_name in enumerate(bar_cols):
        #Do some plotting
        tmp_df = dframe[['id',col_name]].groupby(col_name).count().reset_index().sort_values(by='id',ascending=False)
        colors = ['blue'] * tmp_df.shape[0]
        colors[1] = 'crimson'
        col_bar = go.Bar(x=tmp_df[col_name],y=tmp_df['id'],marker_color=colors)
        fig.add_trace(col_bar,row=i+1,col=1)
    fig.update_layout(height=250*len(bar_cols), width=950, title_text="Category Bars",showlegend=False)
    #fig.show()
    return fig

#Box plots
def create_boxplot_figure(dframe,y_column,plot_columns=None):
    print(f'7. === Box Plots : {dframe.shape}')
    plot_columns = plot_columns if plot_columns else dframe.columns
    bar_cols = set(plot_columns).intersection(cat_columns)
    fig = make_subplots(cols=1, rows=len(bar_cols),subplot_titles=list(bar_cols))
    for i,col_name in enumerate(bar_cols):
        #Do some plotting
        for col_val in dframe[col_name].unique():
            fig.add_trace(go.Box(y=dframe[y_column][dframe[col_name] == col_val],
                            name=col_val),row=i+1,col=1)
    fig.update_layout(height=350*len(bar_cols), width=950, title_text="Category Boxplots",showlegend=False)
    #fig.show()
    return fig

def calculate_car_age(dframe):
    dframe['car_age'] = dframe['year'].map(lambda x : 2020-x+1)
    print(f'3. === Calculate Age : {dframe.shape}')
    return dframe

def custom_clean_up(dframe):
    print('   === CCLean')
    return dframe[['id']+cat_columns+['price']]


def process_raw_data(raw_car_data):
    processed_car_data = (raw_car_data.pipe(copy_dataset)
                                         .pipe(drop_columns)
                                         .pipe(fix_null_columns)
                                         .pipe(calculate_car_age)
                                         .pipe(remove_outliers,'price',upper_lim=200000,lower_lim=50)
                                         .pipe(remove_outliers,'car_age',upper_lim=61,lower_lim=2)
                                         .pipe(remove_outliers,'odometer',upper_lim=700000,lower_lim=0)
                                         .pipe(bin_column,'car_age',12)
                                         .pipe(bin_column,'odometer',odo_bins)
                                         .pipe(custom_clean_up)
                             )
    return processed_car_data

def run_processing_pipeline(run_eda = False):
    raw_car_data = pd.read_csv('vehicles.csv')
    processed_car_data = process_raw_data(raw_car_data)
    if run_eda:
        process_all_groups(processed_car_data,cat_columns_agg,'price',agg_list)
    return processed_car_data
    