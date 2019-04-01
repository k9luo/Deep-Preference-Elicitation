from scipy.sparse import csr_matrix, load_npz, save_npz

import pandas as pd


def save_numpy(matrix, path, model):
    save_npz('{}{}'.format(path, model), matrix)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_pandas(path, name, row_name='userId', col_name='movieId',
                value_name='rating', shape=(138494, 131263), sep=','):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[row_name]
    cols = df[col_name]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    return csr_matrix((values, (rows, cols)), shape=shape)
