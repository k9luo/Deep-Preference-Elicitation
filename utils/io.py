from scipy.sparse import csr_matrix, load_npz, save_npz

import json
import pandas as pd
import time
import datetime


def save_numpy(matrix, path, model):
    save_npz('{}{}'.format(path, model), matrix)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_pandas(path, name, row_name='userId', col_name='movieId',
                value_name='rating', shape=(138494, 131263), sep=','):
    df = pd.read_csv(path + name, sep=sep)
    return df_to_sparse(df, row_name=row_name, col_name=col_name, value_name=value_name, shape=shape)


def date_to_timestamp(date):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    return time.mktime(dt.timetuple())


def get_yelp_df(path, sampling=False, top_user_num=6100, top_item_num=4000):
    with open(path) as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))

    df = pd.DataFrame(data)
    df.rename(columns={'stars': 'review_stars', 'cool': 'review_cool',
                       'funny': 'review_funny', 'useful': 'review_useful'},
              inplace=True)

    df['business_num_id'] = df.business_id.astype('category').\
        cat.rename_categories(range(0, df.business_id.nunique()))
    df['business_num_id'] = df['business_num_id'].astype('int')

    df['user_num_id'] = df.user_id.astype('category').\
    cat.rename_categories(range(0, df.user_id.nunique()))
    df['user_num_id'] = df['user_num_id'].astype('int')

    df['timestamp'] = df['date'].apply(date_to_timestamp)

    if sampling:
        df = filter_yelp_df(df, top_user_num=top_user_num, top_item_num=top_item_num)

    rating_matrix = df_to_sparse(df, row_name='user_num_id',
                                 col_name='business_num_id',
                                 value_name='review_stars',
                                 shape=None)

    timestamp_matrix = df_to_sparse(df, row_name='user_num_id',
                                    col_name='business_num_id',
                                    value_name='timestamp',
                                    shape=None)

    return rating_matrix, timestamp_matrix


def filter_yelp_df(df, top_user_num=6100, top_item_num=4000):
    df_implicit = df[df['review_stars']>3]
    frequent_user_id = df_implicit['user_num_id'].value_counts().head(top_user_num).index.values
    frequent_item_id = df_implicit['business_num_id'].value_counts().head(top_item_num).index.values
    return df.loc[(df['user_num_id'].isin(frequent_user_id)) & (df['business_num_id'].isin(frequent_item_id))]


def df_to_sparse(df, row_name='userId', col_name='movieId', value_name='rating',
                 shape=(138494, 131263)):
    rows = df[row_name]
    cols = df[col_name]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    return csr_matrix((values, (rows, cols)), shape=shape)

