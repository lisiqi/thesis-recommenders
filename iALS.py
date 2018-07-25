""" Apply implicit package to calculate related apps
from the Frappe dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/

The dataset can be found from
http://baltrunas.info/research-menu/frappe
"""

from __future__ import print_function

import logging
import time

import numpy
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.annoy_als import AnnoyAlternatingLeastSquares
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from measure import MAP
from measure import MP
from measure import NDCG
from measure import ERR

import matplotlib.pyplot as plt

# train_file = "train_rel_cold_0.5.csv"
# test_file = "test_warm.csv"

# train_file = "train80_3.csv"
# test_file = "test20_3.csv"


def read_data(filename):
    """ Reads in the frappe dataset, and returns a tuple of a pandas dataframe
    and a sparse matrix of item/user/cnt """

    # read in triples of user/item/cnt from the input dataset
    #data = pandas.read_csv(filename, sep="\t")

    # Read in triples of user/item/cnt from the input training set:
    # data_cols = ['user_id', 'item_id', 'cnt', 'daytime', 'weekday','isweekend', 'homework', 'cost', 'weather', 'country', 'city']

    data = pandas.read_csv(filename, sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
    print("original data")
    print(data)

    # sum up the counts for a given user and a given item
    data = data.groupby(["user","item"], as_index = False).sum()
    print("summed data")
    print(data)

    # map each item and user to a unique numeric value
    # data['user'] = data['user'].astype("category")
    # data['item'] = data['item'].astype("category")
    #
    # # create a sparse matrix of all the users/cnts
    # cnts = coo_matrix((data['cnt'].astype(float),
    #                    (data['item'].cat.codes.copy(),
    #                     data['user'].cat.codes.copy())))

    cnts = coo_matrix((data['cnt'].astype(float),
                       (data['item'],
                        data['user'])))

    print("conts")
    print(cnts)

    return data, cnts


def calculate_recommendations(train_filename, test_filename, output_filename, dir,
                              model_name="als",
                              factors=80, regularization=0.8,
                              iterations=10,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False):
    logging.debug("Calculating similar items. This might take a while")

    # read in the input data file
    logging.debug("reading data from %s", dir+train_filename)
    start = time.time()
    df, cnts = read_data(dir+train_filename)
    logging.debug("read data file in %s", time.time() - start)

    # generate a recommender model based off the input params
    if model_name == "als":
        if exact:
            model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                            use_native=use_native, use_cg=cg, iterations= iterations,
                                            dtype=dtype)
        else:
            model = AnnoyAlternatingLeastSquares(factors=factors, regularization=regularization,
                                                 use_native=use_native, use_cg=cg, iterations= iterations,
                                                 dtype=dtype)

        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        cnts = bm25_weight(cnts, K1=100, B=0.8)

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender(K1=100, B=0.5)

    else:
        raise NotImplementedError("TODO: model %s" % model_name)

    # train the model
    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(cnts)
    logging.debug("trained model '%s' in %s", model_name, time.time() - start)

    #
    test_data = pandas.read_csv(test_filename, sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
    test_data = test_data.groupby(["user", "item"], as_index=False).sum()
    users_test = set(test_data['user'])
    users_train = set(df['user'])

    # position is important for recommendation list and actual list
    dict_actual = {}
    for user in users_test:
        if user not in users_train:
            continue
        matched_df = test_data.loc[test_data["user"]==user]
        matched_df.sort(["cnt"], ascending=[False], inplace=True)
        dict_actual[user] = list(matched_df["item"])


    user_items = cnts.T.tocsr()
    print(user_items)
    # recommend items for a user
    dict_recommended = {} # for computing MAP and MP
    dict_recommended_df = {} # for computing NDCG
    for user in users_test:

        # if(user in users_train):
        #     print(user)
        #
        #     recommendations = model.recommend(user, user_items)
        #     print(recommendations)
        # else:
        #     continue
        if user not in users_train:
            continue
        print(user)
        recommendations = model.recommend(user, user_items)
        df = pandas.DataFrame(recommendations, columns =["item","score"])
        print(recommendations)
        print(df["item"])
        dict_recommended[user] = list(df["item"])
        dict_recommended_df[user] = df

    # print(dict_actual)
    # print(dict_recommended)
    # print("MAP")
    # print(MAP(dict_actual,dict_recommended))
    # print("MP")
    # print(MP(dict_actual, dict_recommended))
    # print("NDCG")
    # print(NDCG(dict_actual, dict_recommended))
    # print("ERR")
    # print(ERR(dict_actual, dict_recommended))

    ndcg = NDCG(dict_actual, dict_recommended)

    err = ERR(dict_actual, dict_recommended)

    map = MAP(dict_actual, dict_recommended)

    mp = MP(dict_actual, dict_recommended)

    with open("%siALS_result_%s.txt" % (dir, train_filename), "w") as o:
        o.write("NDCG\tERR\tMAP\tMP\n")
        o.write("%s\t%s\t%s\t%s\n" % (ndcg, err, map, mp))


    return (ndcg, err, map, mp)

    # # write out similar items by popularity
    # logging.debug("calculating top items")
    # user_count = df.groupby('item').size()
    # items = dict(enumerate(df['item'].cat.categories))
    # to_generate = sorted(list(items), key=lambda x: -user_count[x])
    #
    # # write out as a TSV of itemid, otheritemid, score
    # with open(output_filename, "w") as o:
    #     for itemid in to_generate:
    #         item = items[itemid]
    #         for other, score in model.similar_items(itemid, 11):
    #             o.write("%s\t%s\t%s\n" % (item, items[other], score))

def plot_with_K():
    factors = range(20, 210, 20)
    NDCG = [0.032872852,0.038897608,0.041286546,0.038957409,0.037351761,0.035125123,0.033588317,0.033338131,0.029356787,0.030264973]
    ERR = [0.044196744,0.054217954,0.057415726,0.054844445,0.052973375,0.051182259,0.047826587,0.046193844,0.041942951,0.041110595]
    MAP = [0.020140165,0.024489127,0.026530105,0.025304609,0.024318702,0.022545666,0.021392688,0.020821876,0.018174161,0.018572511]
    MP = [0.026551027,0.033600632,0.035614236,0.033345026,0.031732662,0.030642174,0.030038913,0.02967693,0.026530307,0.026626405]
    e_NDCG = [0.012040283,0.014828075,0.015457538,0.014706136,0.013943027,0.012747552,0.012345287,0.011825532,0.010467743,0.010789756]
    e_ERR = [0.01530382,0.019667428,0.021187786	,0.02056635,0.019177662	,0.018237284,0.017221768,0.015376224,0.014678835,0.013031837]
    e_MAP = [0.008060732,0.010139151,0.010803105,0.010648667,0.01000284,0.008920935,0.00847198,0.00792969,0.006911961,0.007042457]
    e_MP = [0.010196979,0.013076933,0.013574634,0.011928375,0.01143605,	0.011144217,0.011254088,0.01124475,0.009773981,0.010223195]
    # for k in range(20, 210, 20):
    #     factors.append(k)
    #     (ndcg, err, map, mp) = calculate_recommendations(train_file, test_filename, None, '',
    #                           model_name="als",
    #                           factors=k, regularization=re,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    #     NDCG.append(ndcg)
    #     ERR.append(err)
    #     MAP.append(map)
    #     MP.append(mp)

    # plot
    plt.title('Performance of iALS with regard to K')
    plt.xlabel('K (dimensionality)')
    plt.ylim([0, 0.26])
    plt.xticks(factors, factors)
    plt.errorbar(factors, NDCG, e_NDCG, fmt = 'r-', label="NDCG")
    plt.errorbar(factors, ERR,e_ERR, fmt = 'b-', label="ERR")
    plt.errorbar(factors, MAP, e_MAP, fmt =  'g-', label="MAP")
    plt.errorbar(factors, MP, e_MP, fmt = 'y-', label="MP")
    plt.grid('on')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, prop={'size': 10})
    plt.savefig('iALS_K.png', bbox_inches='tight')
    return (factors, NDCG, ERR, MAP, MP)

# plot_with_K()

def iALS_K(train_file, test_filename):
    # factors = []
    NDCG = []
    ERR = []
    MAP = []
    MP = []
    for k in range(20, 210, 20):
        # factors.append(k)
        (ndcg, err, map, mp) = calculate_recommendations(train_file, test_filename, None, '',
                              model_name="als",
                              factors=k, regularization=0.8,
                              iterations=10,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False)
        NDCG.append(ndcg)
        ERR.append(err)
        MAP.append(map)
        MP.append(mp)
    return (NDCG, ERR, MAP, MP)

def hyperK():
    with open("iALS_K_cross_validation.txt", "w") as o:

        for i in [1,2,3,4,5]:
            (NDCG, ERR, MAP, MP) = iALS_K("train80_fold%s.csv" % i, "test20_fold%s.csv" % i)
            o.write("fold%s\n" % i)
            o.write("NDCG:\n")
            for item in NDCG:
                o.write("%s\t" % item)
            o.write("\n")

            o.write("ERR:\n")
            for item in ERR:
                o.write("%s\t" % item)
            o.write("\n")

            o.write("MAP:\n")
            for item in MAP:
                o.write("%s\t" % item)
            o.write("\n")

            o.write("MP:\n")
            for item in MP:
                o.write("%s\t" % item)
            o.write("\n")

# hyperK()


def plot_with_regular(f):
    factors = []
    NDCG = []
    ERR = []
    MAP = []
    MP = []
    for k in numpy.arange(0.1, 1.0, 0.1):
        factors.append(k)
        (ndcg, err, map, mp) = calculate_recommendations(train_file, None,
                              model_name="als",
                              factors=f, regularization=k,
                              iterations=10,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False)
        NDCG.append(ndcg)
        ERR.append(err)
        MAP.append(map)
        MP.append(mp)

    # plot
    plt.xlabel('regularization')
    plt.xticks(factors, factors)
    plt.plot(factors, NDCG, 'r', label="NDCG")
    plt.plot(factors, ERR, 'b', label="ERR")
    plt.plot(factors, MAP, 'g', label="MAP")
    plt.plot(factors, MP, 'y', label="MP")
    plt.grid('on')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, prop={'size': 10})
    plt.savefig('iALS_regular_f%s.png' % f, bbox_inches='tight')
    return (factors, NDCG, ERR, MAP, MP)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generates related items on the last.fm dataset",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#     parser.add_argument('--input', type=str, default=train_file,
#                         dest='train_file', help='frappy dataset file')
#     parser.add_argument('--output', type=str, default='similar-apps.tsv',
#                         dest='outputfile', help='output file name')
#     parser.add_argument('--model', type=str, default='als',
#                         dest='model', help='model to calculate (als/bm25/tfidf/cosine)')
#     parser.add_argument('--factors', type=int, default=80, dest='factors',
#                         help='Number of factors to calculate')
#     parser.add_argument('--reg', type=float, default=0.8, dest='regularization',
#                         help='regularization weight')
#     parser.add_argument('--iter', type=int, default=10, dest='iterations',
#                         help='Number of ALS iterations')
#     parser.add_argument('--exact', help='compute exact distances (slow)', action="store_true")
#     parser.add_argument('--purepython',
#                         help='dont use cython extension (slow)',
#                         action="store_true")
#     parser.add_argument('--float32',
#                         help='use 32 bit floating point numbers',
#                         action="store_true")
#     parser.add_argument('--cg',
#                         help='use CG optimizer',
#                         action="store_true")
#     args = parser.parse_args()
#
#     logging.basicConfig(level=logging.DEBUG)
#
#     calculate_recommendations(args.train_file, args.outputfile,
#                               model_name=args.model,
#                               factors=args.factors,
#                               regularization=args.regularization,
#                               exact=args.exact,
#                               iterations=args.iterations,
#                               use_native=not args.purepython,
#                               dtype=numpy.float32 if args.float32 else numpy.float64,
#                               cg=args.cg)

# plot_with_K("train80_3.csv", "test20_3.csv",0.8)
# plot_with_K(0.2)
# plot_with_regular(80)
# plot_with_regular(90)

def e2(dir):
    num = [1,2,5,10,15,20,40,60,80,100,200,300,500]
    for i in num:
        calculate_recommendations("train_user_%s.csv" % i, "test20.csv", None, dir,
                              model_name="als",
                              factors=80, regularization=0.8,
                              iterations=10,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False)

def e2_1_fix():
    num = [1, 2, 5, 8, 10, 15, 20, 40, 70, 100, 200]
    for i in num:
        for k in [1, 3, 4, 5]:
            calculate_recommendations("train80_%s.csv_item_%s.csv" % (k,i), "test20_%s.csv" % k, None, "e2-fix-lightfm/",
                                  model_name="als",
                                  factors=80, regularization=0.8,
                                  iterations=10,
                                  exact=False,
                                  use_native=True,
                                  dtype=numpy.float64,
                                  cg=False)

e2_1_fix()

# for i in [1, 2, 5, 10, 15, 20, 40, 60, 80, 100, 200, 300, 500]:
#     for k in [2,3,4,5]:
#         calculate_recommendations("train80_%s.csv_user_%s.csv" % (k,i), "test20_%s.csv" % k, None, "e2/",
#                                       model_name="als",
#                                       factors=80, regularization=0.8,
#                                       iterations=10,
#                                       exact=False,
#                                       use_native=True,
#                                       dtype=numpy.float64,
#                                       cg=False)
    # calculate_recommendations("train_warm.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_abs_cold_item.csv", "test_abs_cold_item.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_abs_cold_user.csv", "test_abs_cold_user.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.05.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.1.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.2.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.3.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.4.csv", "test_warm.csv", None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)
    # calculate_recommendations("train_rel_cold_0.5.csv", "test_warm.csv" , None, dir,
    #                           model_name="als",
    #                           factors=80, regularization=0.8,
    #                           iterations=10,
    #                           exact=False,
    #                           use_native=True,
    #                           dtype=numpy.float64,
    #                           cg=False)


# e2("e2-2/")
# e2("e2-3/")
# e2("e2-4/")
# e2("e2-5/")

def e2_2(dir):
    num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in num:
        calculate_recommendations("train_rel_cold_%s.csv" % i, "test20_2.csv", None, dir,
                              model_name="als",
                              factors=80, regularization=0.8,
                              iterations=10,
                              exact=False,
                              use_native=True,
                              dtype=numpy.float64,
                              cg=False)
# e2_2("e2-2/2/")

########## test

# data = pandas.read_csv("e2-4/train_abs_cold_item.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(data["user"])))
#
# test_data = pandas.read_csv("e2-4/test_abs_cold_item.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(test_data["user"])))
#
# data = pandas.read_csv("e2-4/train_abs_cold_user.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(data["user"])))
#
# test_data = pandas.read_csv("e2-4/test_abs_cold_user.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(test_data["user"])))
#
# data = pandas.read_csv("e2-4/train_warm.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(data["user"])))
#
# test_data = pandas.read_csv("e2-4/test_warm.csv", sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
# print(len(set(test_data["user"])))

