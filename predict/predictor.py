from tqdm import tqdm

import numpy as np


def sampling_predict(prediction_scores, topK, matrix_train, matrix_active, sample_from_all, iterative, history_items, gpu=False):
    prediction = []

    for user_index in tqdm(range(len(prediction_scores))):
#        import ipdb; ipdb.set_trace()
        if history_items.size == 0:
            history_item = history_items
        else:
            history_item = history_items[user_index]

        vector_predict = sub_routine(prediction_scores[user_index],
                                     train_index=matrix_train[user_index].nonzero()[1],
                                     active_index=matrix_active[user_index].nonzero()[1],
                                     sample_from_all=sample_from_all,
                                     iterative=iterative,
                                     history_item=history_item, topK=topK,
                                     gpu=gpu)

        # Return empty list when there is a user has less than topK items to
        # recommend. The main program will stop.
        if len(vector_predict) != topK:
            raise ValueError('user {} has less than top {} items to recommend. Return empty list in this case.'.format(user_index, topK))
            return []

        prediction.append(vector_predict)

    predict_items = np.vstack(prediction)
#    import ipdb; ipdb.set_trace()
    if history_items.size == 0:
        history_items = predict_items
    else:
        history_items = np.column_stack((history_items, predict_items))
#    import ipdb; ipdb.set_trace()
    return predict_items, history_items

def sub_routine(vector_predict, train_index, active_index, sample_from_all, iterative, history_item, topK=500, gpu=False):

#    sort_length = topK + len(train_index)

#    if sort_length + 1 > len(vector_predict) or not sample_from_all:
#        sort_length = len(vector_predict) - 1

    sort_length = len(vector_predict) - 1
#    import ipdb; ipdb.set_trace()
    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, sort_length)[:sort_length]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, sort_length)[:sort_length]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
#    import ipdb; ipdb.set_trace()

    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
#    import ipdb; ipdb.set_trace()

    if history_item.size != 0 and iterative:
        vector_predict = np.delete(vector_predict, np.isin(vector_predict, history_item).nonzero()[0])
#    import ipdb; ipdb.set_trace()

    if not sample_from_all:
        vector_predict, index, _ = np.intersect1d(vector_predict, active_index, return_indices=True)
        vector_predict = vector_predict[index.argsort()]
#    import ipdb; ipdb.set_trace()

#    predict_items = vector_predict[:topK]
#    history_item = np.concatenate([history_item, predict_items])

    return vector_predict[:topK]
