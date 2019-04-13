from predict.predictor import sampling_predict
from scipy.sparse import csr_matrix

import numpy as np


def update_matrix(history_items, matrix_train, matrix_active, observation, train_index, iterative, sample_from_all, num_item_per_iter, iteration, gpu):
    if iterative:
        topk = num_item_per_iter
    else:
        topk = num_item_per_iter * iteration

    predict_items, history_items = sampling_predict(prediction_scores=observation,
                                                    topK=topk,
                                                    matrix_train=matrix_train[train_index:],
                                                    matrix_active=matrix_active[train_index:],
                                                    sample_from_all=sample_from_all,
                                                    iterative=iterative,
                                                    history_items=history_items,
                                                    gpu=gpu)

#    import ipdb; ipdb.set_trace()


    mask_row = train_index + np.repeat(np.arange(len(predict_items)), topk)
    mask_col = predict_items.ravel()
    mask_data = np.full(len(predict_items)*topk, True)
    mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_train.shape)
    matrix_train[mask] = matrix_active[mask]
#    import ipdb; ipdb.set_trace()

    return matrix_train, history_items
