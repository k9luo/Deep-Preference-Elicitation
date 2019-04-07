from predict.predictor import sampling_predict
from providers.filter import filter_users
from providers.update_matrix import update_matrix
from utils.argcheck import check_float_positive, check_int_positive, ratio
from utils.io import load_numpy
from utils.modelnames import active_models, rec_models
from utils.progress import inhour, WorkSplitter
from utils.regularizers import Regularizer

import argparse
import time


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.path))
    print("Active Learning Algorithm: {}".format(args.active_model))
    print("Recommendation Algorithm: {}".format(args.rec_model))
    print("GPU: {}".format(args.gpu))
    print("Iterative: {}".format(args.iterative))
    print("Sample From All: {}".format(args.sample_from_all))
    print("Train Valid Test Split Ratio: {}".format(args.ratio))
    print("Learning Rate: {}".format(args.learning_rate))
    print("Rank: {}".format(args.rank))
    print("Lambda: {}".format(args.lamb))
    print("Epoch: {}".format(args.epoch))
    print("Active Learning Iteration: {}".format(args.iter))
    print("Evaluation Ranking Topk: {}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_numpy(path=args.path, name=args.train)
    print("Train U-I Dimensions: {}".format(R_train.shape))

    R_active = load_numpy(path=args.path, name=args.active)
    print("Active U-I Dimensions: {}".format(R_active.shape))

    R_test = load_numpy(path=args.path, name=args.test)
    print("Test U-I Dimensions: {}".format(R_test.shape))

    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    train_index = int(R_test.shape[0]*args.ratio[0])

    progress.section("Preparing Data")
    matrix_train, matrix_active, matrix_test, _ = filter_users(R_train,
                                                               R_active,
                                                               R_test,
                                                               train_index=train_index,
                                                               active_threshold=2*args.num_item_per_iter*args.iter,
                                                               test_threshold=2*args.topk)
    # TODO: After this point
    import ipdb; ipdb.set_trace()

    m, n = matrix_train.shape

    metrics_result = []
    history_items = np.array([])

    model = rec_models[rec_model](observation_dim=n, latent_dim=args.rank,
                                  batch_size=128, lamb=args.lamb,
                                  learning_rate=args.learning_rate,
                                  optimizer=Regularizer[args.optimizer])

    progress.section("Training")
    model.train_model(matrix_train[:train_index], args.corruption, args.epoch)

    for i in range(args.iter):
        print('This is step {} \n'.format(i))
        print('The number of ones in train set is {}'.format(len(matrix_train[train_index:].nonzero()[0])))
        print('The number of ones in active set is {}'.format(len(matrix_active[train_index:].nonzero()[0])))

        progress.section("Predicting")
        observation = active_models[active_model](model, matrix_train[train_index:].A, args.ci, args.num_latent_sampling)

        progress.section("Update Train Set")
        matrix_train, history_items = update_matrix(history_items, matrix_train,
                                                    matrix_active, observation,
                                                    train_index, args.iterative,
                                                    args.sample_from_all,
                                                    args.num_item_per_iter,
                                                    args.iter, args.gpu)

        if not args.iterative:
            break

    print('The number of ones in train set is {}'.format(len(matrix_train[train_index:].nonzero()[0])))

    progress.section("Re-Training")
    model.train_model(matrix_train, args.corruption, args.epoch)

    progress.section("Re-Predicting")
    observation = active_models['Greedy'](model, matrix_train.A)

    result = {}
    for topk in [1, 5, 10, 50]:
        predict_items, _ = sampling_predict(prediction_scores=observation[train_index:],
                                            topK=topk,
                                            matrix_train=matrix_train[train_index:],
                                            matrix_active=matrix_active[train_index:],
                                            sample_from_all=True,
                                            iterative=False,
                                            history_items=np.array([]),
                                            gpu=gpu)

        progress.section("Create Metrics")
        result.update(eval(matrix_test[train_index:], topk, predict_items))

    metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="DeepPreferenceElicitation")

    parser.add_argument('--disable-gpu', dest='gpu', action='store_false') #print
    parser.add_argument('--disable-iterative', dest='iterative', action='store_false') #print
    parser.add_argument('--disable-sample-from-all', dest='sample_from_all', action='store_false') #print
    parser.add_argument('-a', dest='active', default='Ractive.npz')
    parser.add_argument('-active-model', dest='active_model', default="ThompsonSampling") #print
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.5)
    parser.add_argument('-ci', dest='confidence', type=check_int_positive, default=1)
    parser.add_argument('-d', dest='path', default="data/") #print
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1) #print
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1) #print
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50) #print
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100) #print
    parser.add_argument('-learning-rate', dest='learning_rate', type=check_float_positive, default=100.0) #print
    parser.add_argument('-num-item-per-iter', dest='num_item_per_iter', type=check_int_positive, default=1)
    parser.add_argument('-num-latent-sampling', dest='num_latent_sampling', type=check_int_positive, default=5)
    parser.add_argument('-optimizer', dest='optimizer', default="RMSProp")
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100) #print
    parser.add_argument('-ratio', dest='ratio', type=ratio, default='0.5, 0.0, 0.5') #print
    parser.add_argument('-rec-model', dest='rec_model', default="VAE-CF") #print
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=8292)
    parser.add_argument('-te', dest='test', default='Rtest.npz')
    parser.add_argument('-tr', dest='train', default='Rtrain.npz')
    args = parser.parse_args()

    main(args)
