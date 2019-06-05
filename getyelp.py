from utils.argcheck import ratio, shape, check_int_positive
from utils.io import load_pandas, save_numpy
from utils.progress import WorkSplitter
from providers.split import split_user_randomly, time_ordered_split
from utils.io import get_yelp_df

import argparse


def main(args):
    progress = WorkSplitter()

    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.path))
    print("Validation: {}".format(args.validation))
    print("Implicit: {}".format(args.implicit))

    progress.section("Load Raw Data")
    rating_matrix, timestamp_matrix = get_yelp_df(args.path+args.name,
                                                  sampling=True,
                                                  top_user_num=args.top_user_num,
                                                  top_item_num=args.top_item_num)

    progress.section("Split CSR Matrices")
    rtrain, rvalid, rtest, _, _, rtime = split_user_randomly(rating_matrix=rating_matrix,
                                                             timestamp_matrix=timestamp_matrix,
                                                             ratio=args.split_user_ratio,
                                                             implicit=args.implicit)

    if args.validation:
        rtrain, rvalid, _, _, _ = time_ordered_split(rating_matrix=rtrain,
                                                     timestamp_matrix=rtime,
                                                     ratio=args.split_train_valid_ratio,
                                                     implicit=False,
                                                     remove_empty=False)

    ractive, rtest, _, _, _ = time_ordered_split(rating_matrix=rtest,
                                                 timestamp_matrix=rtime,
                                                 ratio=args.split_active_test_ratio,
                                                 implicit=False,
                                                 remove_empty=False)

    progress.section("Save NPZ")
    save_numpy(rtrain, args.path, "Rtrain")
    save_numpy(rvalid, args.path, "Rvalid")
    save_numpy(ractive, args.path, "Ractive")
    save_numpy(rtest, args.path, "Rtest")
    save_numpy(rtime, args.path, "Rtime")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="GetYelp")

    parser.add_argument('--disable_validation', dest='validation', action='store_false')
    parser.add_argument('--enable_implicit', dest='implicit', action='store_true')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    parser.add_argument('--split_active_test_ratio', dest='split_active_test_ratio', type=ratio, default='0.7, 0.3, 0.0')
    parser.add_argument('--path', dest='path', default="data/")
    parser.add_argument('--name', dest='name', default='ml-1m/ratings.csv')
    parser.add_argument('--split_train_valid_ratio', dest='split_train_valid_ratio', type=ratio, default='0.5, 0.5, 0.0')
    parser.add_argument('--split_user_ratio', dest='split_user_ratio', type=ratio, default='0.5, 0.0, 0.5')
    parser.add_argument('--top_item_num', dest='top_item_num', default=4000, type=check_int_positive)
    parser.add_argument('--top_user_num', dest='top_user_num', default=6100, type=check_int_positive)

    args = parser.parse_args()

    main(args)
