import pandas as pd
import math
from itertools import combinations
from scipy import stats

## Deep EDA
def build_single_column_string(x, field_length):
    col_string = ''
    for i in range(0,field_length):
        col_string +='>'+str(x[i])
    return col_string[1:]

def process_group(dframe,grp_cols,target_col, aggregations_list):#['count','sum']
    #grp_cols = ['Pclass', 'Sex']
    tt_grp = dframe.groupby(by=grp_cols).agg({target_col:aggregations_list})
    tt_grp.columns =[f'{target_col}_{x}' for x in aggregations_list]#
    tt_grp_index = tt_grp.reset_index()
    tt_grp_index['agg_cols_data'] = tt_grp_index[tt_grp_index.columns[0:len(grp_cols)]].\
                                    apply(lambda x : build_single_column_string(x,len(grp_cols)),axis=1)
    tt_grp_index['agg_cols_cnt'] = len(grp_cols)
    tt_grp_index['agg_cols'] = '>'.join(grp_cols)
    tt_grp_index = tt_grp_index[tt_grp_index.columns[len(grp_cols):]]
    col_list = list(tt_grp_index.columns)
    col_list.reverse()
    tt_grp_index = tt_grp_index[col_list].dropna()
    return tt_grp_index

def process_all_groups(dframe,column_list,target_col,aggregation):
    print('==== Processing Groups ====')
    def all_combos(s):
        n = len(s)
        for r in range(1, n+1):
            for combo in combinations(s, r):
                yield combo
    ff = list(all_combos(column_list))
    dl = []
    for cnt, i in enumerate(ff):
        print('     == Processing Group {} of {} =='.format(cnt+1,len(ff)))
        p_grp = process_group(dframe,list(i),target_col,aggregation)
        p_grp.to_csv(f'eda/group_{cnt}.csv',index=False)
        #dl.append(p_grp)
    #result_df = pd.concat(dl,axis=0).reset_index();
    #result_df.drop('index',inplace=True,axis=1)
    #return # result_df

def compute_2tail_2sample_tscore(mean1,mean2,stdv1,stdv2,n1,n2):
    t_score = ((mean1-mean2)/math.sqrt((stdv1**2/n1)+(stdv2**2/n2)))/2.0
    df = n2-1 if n2 >= n1 else n1-1
    p_value = stats.t.sf(abs(t_score),df=df)
    est_value = abs(mean1-mean2)
    return {'t_score':abs(t_score),'p_value':p_value,'est_value':est_value, 'df':df}

## Data Processing Pipeline
def copy_dataset(dframe):
    print(f'0. === Copying Data : {dframe.shape}')
    return dframe.copy(deep=True)

def drop_columns(dframe):
    dframe.drop(columns='county',inplace=True)
    print(f'1. === Drop Columns : {dframe.shape}')
    return dframe

def fix_null_columns(dframe):
    print(f'2. === Filling Nulls : {dframe.shape}')
    df_columns  = dframe.columns
    df_types = dframe.dtypes
    for i,col_name in enumerate(df_columns):
        print(f'   2.{i+1} === {col_name}')
        col_type = df_types[col_name]
        if 'float64' == col_type or 'int64' == col_type:
            dframe[col_name] = dframe[col_name].fillna(0)
        if 'O'==col_type:
            dframe[col_name] = dframe[col_name].fillna('NA')
    return dframe

def remove_outliers(dframe,col_name,upper_lim,lower_lim):
    dframe=dframe[(dframe[col_name]<=upper_lim) & (dframe[col_name]>=lower_lim)]
    print(f'4. === Removing {col_name} Outliers : {dframe.shape}')
    return dframe

def filter_dataframe(dframe, col_filters):#{col_name : col_value}
    print(f'5. === Filtering : {dframe.shape}')
    for col_name in col_filters.keys():
        col_value = col_filters[col_name]
        dframe = dframe[dframe[col_name]==col_value]
        dframe.drop(columns=col_name,inplace=True)
        print(f'5.    === Filtered {col_name} by {col_value} : {dframe.shape}')
    print(f'5. === Done Filtering : {dframe.shape}')
    return dframe

def bin_column(dframe, col_name, nbins):
    bckt_count = len(nbins)-1 if type(nbins)==list else nbins
    dframe[f'{col_name}_bckt'] = pd.cut(dframe[col_name],bins=nbins)#,\
                                        #labels=[f'{col_name}_bckt_{i}' for i in range(bckt_count)])
    print(f'8. === Binning {col_name} : {dframe.shape}')
    return dframe

#