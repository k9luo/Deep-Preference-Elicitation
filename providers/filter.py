import numpy as np

# Remove users who have less than 2*num_item_per_iter*al_iteration positive
# ratings in active set from active and test set.
# Remove users who have less than 2*topk positive ratings in active set
# from active and test set.
def filter_users(matrix_train, matrix_active, matrix_test, train_index, active_threshold, test_threshold):
    active_user_num_nonzero = np.array(matrix_active.sum(axis=1)).ravel()
    active_users = np.where(active_user_num_nonzero >= active_threshold)[0]

    users = np.concatenate([np.arange(train_index), active_users])

    return matrix_train[users,:], matrix_active[users,:], matrix_test[users,:], users

#    test_user_num_nonzero = np.array(matrix_test.sum(axis=1)).ravel()
#    test_users = np.where(test_user_num_nonzero >= test_threshold)[0]

#    test_users = np.intersect1d(active_users, test_users)
#    users = np.concatenate([np.arange(train_index), test_users])
#    return matrix_train[users,:], matrix_active[users,:], matrix_test[users,:], users
