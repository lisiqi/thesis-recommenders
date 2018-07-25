import pandas
import random
import scipy.sparse as sp
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import auc_score
from measure import MAP
from measure import MP
from measure import NDCG
from measure import ERR
import matplotlib.pyplot as plt

def build_interaction_matrix(df, users, items):
    #df["cnt"] = 1

    users_not_in_df = users - set(df["user"])
    for user in users_not_in_df:
        df.loc[df.shape[0]] = [user, random.choice(tuple(items)), None]

    items_not_in_df = items - set(df["item"])
    for item in items_not_in_df:
        df.loc[df.shape[0]] = [random.choice(tuple(users)), item, None]

    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)

    matrix = sp.coo_matrix((df['cnt'].astype(float),
                           (df["user"],
                            df["item"])))
    return matrix

def fetch_frappe(original_data_path, train_data_path, test_data_path, item_metedata_path):

    # Load raw data
    data_raw = pandas.read_csv(original_data_path, sep="\t")
    data_raw.head()

    train_raw = pandas.read_csv(train_data_path, sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])
    test_raw = pandas.read_csv(test_data_path, sep="\t", usecols=[0, 1, 2], names=['user', 'item', 'cnt'])

    # sum up the counts for a given user-item
    train_raw = train_raw.groupby(["user", "item"], as_index=False).sum()
    test_raw = test_raw.groupby(["user", "item"], as_index=False).sum()

    # no need to binarize the cnt

    # Figure out the dimensions
    users = set(data_raw["user"])
    items = set(data_raw["item"])

    num_users = len(users)
    num_items = len(items)

    train = build_interaction_matrix(train_raw, users, items)
    test = build_interaction_matrix(test_raw, users, items)
    # print(train)
    # print(test)

    assert train.shape == test.shape

    # identity matrix
    id_features = sp.identity(num_items,
                              format='csr',
                              dtype=np.float32)

    (item_features, id_feature_labels) = parse_item_metadata(items, item_metedata_path)

    return {'train': train,
            'test': test,
            'item_features': item_features,
            'id_features': id_features,
            'item_labels': id_feature_labels,
            'items': items,
            'users_test': set(test_raw["user"]),
            'test_data': test_raw}

def recommendations_lightFM(original_data_path, train_data_path, test_data_path, item_metedata_path, d):

    data = fetch_frappe(original_data_path, train_data_path, test_data_path, item_metedata_path)
    train = data['train']
    test = data['test']
    item_features = data['item_features']
    id_features = data['id_features']
    items = data["items"]
    users_test = data["users_test"]
    test_data = data["test_data"]

    print('The dataset has %s users and %s items, '
          'with %s interactions in the test and %s interactions in the training set.'
          % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))


    # ####### only CF
    # model = LightFM(loss='warp')
    #
    # model = model.fit(train, epochs=10)
    #
    # train_auc = auc_score(model, train).mean()
    # print('Collaborative filtering train AUC: %s' % train_auc)
    #
    # test_auc = auc_score(model, test, train_interactions=train).mean()
    # print('Collaborative filtering test AUC: %s' % test_auc)
    #
    # model.item_biases *= 0.0
    #
    # test_auc = auc_score(model, test, train_interactions=train).mean()
    # print('Collaborative filtering test AUC: %s' % test_auc)
    #
    #
    # ####### including metadata with only item_features
    # print("with only item_features:")
    # model = LightFM(loss='warp')
    #
    # model = model.fit(train, item_features=item_features, epochs=10)
    #
    # train_auc = auc_score(model, train, item_features=item_features).mean()
    # print('Hybrid train AUC: %s' % train_auc)
    #
    # test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
    # print('Hybrid test AUC: %s' % test_auc)
    #
    # model.item_biases *= 0.0
    #
    # test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
    # print('Hybrid test AUC: %s' % test_auc)
    #
    # ######## including metadata with only id_features
    # print("with only id_features:")
    # model = LightFM(loss='warp')
    #
    # model = model.fit(train, item_features=id_features, epochs=10)
    #
    # train_auc = auc_score(model, train, item_features=id_features).mean()
    # print('Hybrid train AUC: %s' % train_auc)
    #
    # test_auc = auc_score(model, test, train_interactions=train, item_features=id_features).mean()
    # print('Hybrid test AUC: %s' % test_auc)
    #
    # model.item_biases *= 0.0
    #
    # test_auc = auc_score(model, test, train_interactions=train, item_features=id_features).mean()
    # print('Hybrid test AUC: %s' % test_auc)

    ######### including metadata with item_features + id_features

#### 4M
    # item_features = sp.hstack([id_features, item_features]).tocsr()
    #
    # model = LightFM(loss='warp', no_components=d)
    #
    # model = model.fit(train, item_features=item_features, epochs=10)

##### 0M
    item_features = sp.hstack([id_features]).tocsr()

    model = LightFM(loss='warp', no_components=d)

    model = model.fit(train, item_features=item_features, epochs=10)


    # actual dictionary of test set
    dict_actual = {}
    for user in users_test:
        matched_df = test_data.loc[test_data["user"] == user]
        matched_df.sort(["cnt"], ascending=[False], inplace=True)
        dict_actual[user] = list(matched_df["item"])

    # recommend items for each user in test set
    dict_recommended = {}
    for user in users_test:
        scores = model.predict(user, np.array(list(items)), item_features=item_features)
        top_items = data['item_labels'][np.argsort(-scores)]

        dict_recommended[user] = list(top_items[:10])

    return (dict_actual, dict_recommended)


def parse_item_metadata(items, item_metedata_path):

    item_metadata_raw = pandas.read_csv(item_metedata_path, sep="\t", usecols=[0, 2, 3, 6, 9, 10])
    item_metadata_raw.head()

    id_feature_labels = np.empty(len(items), dtype=np.object)
    #names=['item', 'category', 'downloads', ('developer'), 'language', 'price', 'rating'])

    # category 32
    # print("category")
    # print(len(set(item_metadata_raw["category"])))
    category = sorted(set(item_metadata_raw["category"]))

    # downloads 16
    # print("downloads")
    # print(len(set(item_metadata_raw["downloads"])))
    downloads = sorted(set(item_metadata_raw["downloads"]))
    # print(downloads)

    # developer 2809, don't user this attribute
    # print("developer")
    # print(len(set(item_metadata_raw["developer"])))
    # developer = sorted(set(item_metadata_raw["developer"]))
    # print(developer)


    # language 29
    # print("language")
    # print(len(set(item_metadata_raw["language"])))
    language = sorted(set(item_metadata_raw["language"]))
    # print(language)

    # price 3: Free, Paid, unknown
    # modify price to paid
    # print("price")
    item_metadata_raw.loc[(item_metadata_raw["price"] !='Free') & (item_metadata_raw["price"] !='unknown'), 'price'] = 'Paid'
    # print(len(set(item_metadata_raw["price"])))
    price = sorted(set(item_metadata_raw["price"]))
    # print(price)

    # rating 9, [0, 3, 4, 5, 6, 7, 8, 9, 10]
    # modify rating to 10 degrees, use 0.5 as one interval
    # print("rating")
    item_metadata_raw.loc[item_metadata_raw["rating"] == 'unknown', 'rating'] = 0
    item_metadata_raw.loc[('0' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '0.5'), 'rating'] = 1
    item_metadata_raw.loc[('0.5' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '1'), 'rating'] = 2
    item_metadata_raw.loc[('1' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '1.5'), 'rating'] = 3
    item_metadata_raw.loc[('1.5' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '2'), 'rating'] = 4
    item_metadata_raw.loc[('2' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '2.5'), 'rating'] = 5
    item_metadata_raw.loc[('2.5' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '3'), 'rating'] = 6
    item_metadata_raw.loc[('3' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '3.5'), 'rating'] = 7
    item_metadata_raw.loc[('3.5' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '4'), 'rating'] = 8
    item_metadata_raw.loc[('4' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '4.5'), 'rating'] = 9
    item_metadata_raw.loc[('4.5' < item_metadata_raw["rating"]) & (item_metadata_raw["rating"] <= '5.0'), 'rating'] = 10

    # print(len(set(item_metadata_raw["rating"])))
    rating = sorted(set(item_metadata_raw["rating"]))
    # print(rating)

    # print(item_metadata_raw)

    # create the matrix of metadata
    # flatten the value of different metadata attribute

    category_features = sp.lil_matrix(
        (len(items), len(category)),
        dtype=np.float32)

    downloads_features = sp.lil_matrix(
        (len(items), len(downloads)),
        dtype=np.float32)

    language_features = sp.lil_matrix(
        (len(items), len(language)),
        dtype=np.float32)

    price_features = sp.lil_matrix(
        (len(items), len(price)),
        dtype=np.float32)

    rating_features = sp.lil_matrix(
        (len(items), len(rating)),
        dtype=np.float32)

    # traverse each row of item_metadata_raw
    for index, row in item_metadata_raw.iterrows():
        # print(index)

        category_features[row["item"], category.index(row["category"])] = 1.0
        downloads_features[row["item"], downloads.index(row["downloads"])] = 1.0
        language_features[row["item"], language.index(row["language"])] = 1.0
        price_features[row["item"], price.index(row["price"])] = 1.0
        rating_features[row["item"], rating.index(row["rating"])] = 1.0

        id_feature_labels[row["item"]] = row["item"]

    # item_features = sp.hstack([category_features, downloads_features, language_features, price_features, rating_features]).tocsr()
    # item_features = sp.hstack([category_features, language_features, price_features, rating_features]).tocsr()
    # item_features = sp.hstack([category_features, language_features, price_features]).tocsr()
    item_features = sp.hstack([price_features, rating_features, downloads_features, language_features, category_features]).tocsr()
    # item_features = sp.hstack([price_features]).tocsr()
    # print(item_features)


    return (item_features, id_feature_labels)


# d : #latent_factor_dimensions
def measure_lightFM(original_data_path, train_data_path, test_data_path, item_metedata_path, d, dir):
    (dict_actual, dict_recommended) = recommendations_lightFM(original_data_path, dir+train_data_path, test_data_path, item_metedata_path, d)

    ndcg = NDCG(dict_actual, dict_recommended)

    err = ERR(dict_actual, dict_recommended)

    map = MAP(dict_actual, dict_recommended)

    mp = MP(dict_actual, dict_recommended)

    with open("%slightFM_result_%s.txt" % (dir, train_data_path), "w") as o:
        o.write("NDCG\tERR\tMAP\tMP\n" )
        o.write("%s\t%s\t%s\t%s\n" % (ndcg,err,map,mp))

    return (ndcg, err, map, mp)


def plot_LightFM_K():
    factors = range(20, 210, 20)
    NDCG = [0.176916616,0.180271308,0.18830333,0.191032428,0.194497001,0.197169258,0.202235005,0.205602832,0.20800753,0.206222085]
    ERR = [0.230901367,0.233369628,0.237706523,0.239715389,0.240283247,0.24006665,0.243951663,0.246751924,0.2494684,0.243494622]
    MAP = [0.136346727,0.138633736,0.145747524,0.147710001,0.149346068,0.152628683,0.15519109,0.157653475,0.161392498,0.157731574]
    MP = [0.174545455,0.178787879,0.184346917,0.188359457,0.190950888,0.194712644,0.198140021,0.199038662,0.200292581,0.200125392]
    e_NDCG = [0.003880676,0.002393242,0.006161338,0.006526891,0.004418513,0.009530948,0.008873828,0.00998723,0.008032579,0.010456542]
    e_ERR = [0.002405796,0.003140883,0.006041138,0.008387184,0.003917406,0.008616275,0.008108721,0.007413949,0.009889407,0.007624779]
    e_MAP = [0.002718,0.001170621,0.004374829,0.004962179,0.002976229,0.007011341,0.006067949,0.007160642,0.005226265,0.006764631]
    e_MP = [0.00328913,0.003489979,0.003912151,0.004469208,0.002457924,0.005177214,0.006080122,0.00701172,0.005692914,0.007953642]
    # for k in range(20, 210, 20):
    #     factors.append(k)
    #     (ndcg, err, map, mp) = measure_lightFM("frappe.csv", "train80.csv", "test20.csv", "meta.csv", k, '')
    #     NDCG.append(ndcg)
    #     ERR.append(err)
    #     MAP.append(map)
    #     MP.append(mp)
    # plot
    plt.title('Performance of LightFM with regard to K')
    plt.xlabel('K (dimensionality)')
    plt.xticks(factors, factors)
    plt.ylim([0,0.26])
    # plt.plot(factors, NDCG, 'r', label="NDCG")
    # plt.plot(factors, ERR, 'b', label="ERR")
    # plt.plot(factors, MAP, 'g', label="MAP")
    # plt.plot(factors, MP, 'y', label="MP")
    plt.errorbar(factors, NDCG, e_NDCG, fmt = 'r-', label="NDCG")
    plt.errorbar(factors, ERR,e_ERR, fmt = 'b-', label="ERR")
    plt.errorbar(factors, MAP, e_MAP, fmt =  'g-', label="MAP")
    plt.errorbar(factors, MP, e_MP, fmt = 'y-', label="MP")
    plt.grid('on')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, prop={'size': 10})
    plt.savefig('LightFM_K.pdf', bbox_inches='tight')

plot_LightFM_K()

def lightFM_K(trainfile, testfile):
    # factors = []
    NDCG = []
    ERR = []
    MAP = []
    MP = []
    for k in range(20, 210, 20):
        # factors.append(k)
        (ndcg, err, map, mp) = measure_lightFM("frappe.csv", trainfile, testfile, "meta.csv", k, '')
        NDCG.append(ndcg)
        ERR.append(err)
        MAP.append(map)
        MP.append(mp)
    return (NDCG, ERR, MAP, MP)

def hyperK():
    with open("lightFM_K_cross_validation.txt", "w") as o:

        for i in [1,2,3,4,5]:
            (NDCG, ERR, MAP, MP) = lightFM_K("train80_fold%s.csv" % i, "test20_fold%s.csv" % i)
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

def measure_multiple_times(original_data_path, train_data_path, test_data_path, item_metedata_path, times, d):
    NDCG = []
    ERR = []
    MAP = []
    MP = []
    for i in range(0, times):
        print(i)
        (ndcg, err, map, mp) = measure_lightFM(original_data_path, train_data_path, test_data_path, item_metedata_path,d)
        NDCG.append(ndcg)
        ERR.append(err)
        MAP.append(map)
        MP.append(mp)
    print("NDCG")
    print(np.mean(NDCG))
    print("ERR")
    print(np.mean(ERR))
    print("MAP")
    print(np.mean(MAP))
    print("MP")
    print(np.mean(MP))


def recommendations_e1(method, i):
    d = 80
    data = fetch_frappe("frappe.csv", "train80_%s.csv" % i, "test20_%s.csv" % i, "meta.csv")
    train = data['train']
    test = data['test']
    item_features = data['item_features']
    id_features = data['id_features']
    items = data["items"]
    users_test = data["users_test"]
    test_data = data["test_data"]

    print(method)

    dict_recommended = {}

    if method=="cf+warp":
        model = LightFM(loss='warp', no_components=d)
        model = model.fit(train, epochs=10)
        train_auc = auc_score(model, train).mean()
        test_auc = auc_score(model, test, train_interactions=train).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)))

            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    elif method =="cf+bpr":
        model = LightFM(loss='bpr', no_components=d)
        model = model.fit(train, epochs=10)
        train_auc = auc_score(model, train).mean()
        test_auc = auc_score(model, test, train_interactions=train).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)))

            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    elif method=="id+warp":
        model = LightFM(loss='warp', no_components=d)
        model = model.fit(train, item_features=id_features, epochs=10)
        train_auc = auc_score(model, train, item_features=id_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=id_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=id_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])


    elif method=="id+bpr":
        model = LightFM(loss='bpr', no_components=d)
        model = model.fit(train, item_features=id_features, epochs=10)
        train_auc = auc_score(model, train, item_features=id_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=id_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=id_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    elif method=="item+warp":
        model = LightFM(loss='warp', no_components=d)
        model = model.fit(train, item_features=item_features, epochs=10)
        train_auc = auc_score(model, train, item_features=item_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=item_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    elif method=="item+bpr":
        model = LightFM(loss='bpr', no_components=d)
        model = model.fit(train, item_features=item_features, epochs=10)
        train_auc = auc_score(model, train, item_features=item_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=item_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    elif method=="id+item+warp":
        item_features = sp.hstack([id_features, item_features]).tocsr()
        model = LightFM(loss='warp', no_components=d)
        model = model.fit(train, item_features=item_features, epochs=10)
        train_auc = auc_score(model, train, item_features=item_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=item_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    else : # method=="id+item+bpr"
        item_features = sp.hstack([id_features, item_features]).tocsr()
        model = LightFM(loss='bpr', no_components=d)
        model = model.fit(train, item_features=item_features, epochs=10)
        train_auc = auc_score(model, train, item_features=item_features).mean()
        test_auc = auc_score(model, test, train_interactions=train, item_features=item_features).mean()
        for user in users_test:
            scores = model.predict(user, np.array(list(items)), item_features=item_features)
            # scores = model.predict(user, np.array(data['item_labels']))
            top_items = data['item_labels'][np.argsort(-scores)]

            dict_recommended[user] = list(top_items[:10])

    # actual dictionary of test set
    dict_actual = {}
    for user in users_test:
        matched_df = test_data.loc[test_data["user"] == user]
        matched_df.sort(["cnt"], ascending=[False], inplace=True)
        dict_actual[user] = list(matched_df["item"])


    ndcg = NDCG(dict_actual, dict_recommended)
    err = ERR(dict_actual, dict_recommended)
    map = MAP(dict_actual, dict_recommended)
    mp = MP(dict_actual, dict_recommended)
    return (train_auc, test_auc, ndcg, err, map, mp)

def measure_e1():
    # with open("e1/lightFM_result_1m_1.txt", "w") as o:
    #     o.write("id+item+warp\n")
    #     o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
    #     o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("id+item+warp", 1))
    #     o.write("\n")

    for i in range(1,6):
        with open("e1/lightFM_result_5m_%s.txt" % i, "w") as o:
        # o.write("cf+warp\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("cf+warp"))
        # o.write("\n")
        #
        # o.write("cf+bpr\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("cf+bpr"))
        # o.write("\n")
        #
        # o.write("id+warp\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("id+warp"))
        # o.write("\n")
        #
        # o.write("id+bpr\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("id+bpr"))
        # o.write("\n")
        #
        # o.write("item+warp\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("item+warp"))
        # o.write("\n")
        #
        # o.write("item+bpr\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("item+bpr"))
        # o.write("\n")

            o.write("id+item+warp\n")
            o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
            o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("id+item+warp", i))
            o.write("\n")

        # o.write("id+item+bpr\n")
        # o.write("train AUC\ttest AUC\tNDCG\tERR\tMAP\tMP\n")
        # o.write("%s\t%s\t%s\t%s\t%s\t%s\n" % recommendations_e1("id+item+bpr"))

# test area
# measure_lightFM("frappe.csv", "train80.csv", "test20.csv", "meta.csv")

# measure_multiple_times("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 10)
# measure_multiple_times("frappe.csv", "train_rel_cold_0.2.csv", "test_warm.csv", "meta.csv", 5)

# plot_LightFM_K()

# measure_e1()



def measure_e2_1():
    num = [1,2,5,10,15,20,40,60,80,100,200,300,500]
    for i in num:
        measure_lightFM("frappe.csv", "train_user_%s.csv" % i, "test20.csv", "meta.csv", 80, "e2/")

# measure_e2_1()
def measure_e2_1_fix():
    num = [1,2,5,8,10,15,20,40,70,100,200]
    for i in num:
        for k in [1,3,4,5]:
            measure_lightFM("frappe.csv", "train80_%s.csv_item_%s.csv" % (k,i), "test20_%s.csv" % k, "meta.csv", 80, "e2-fix-lightfm/")

measure_e2_1_fix()


def measure_e2_2():
    num = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        measure_lightFM("frappe.csv", "train_rel_cold_%s.csv" % i, "test20_2.csv", "meta.csv", 80, "e2-2/2/")

# measure_e2_2()

# for i in [2, 60, 80, 100, 200, 300, 500]:
#     for k in [2,3,4,5]:
#         measure_lightFM("frappe.csv", "train80_%s.csv_user_%s.csv" % (k,i), "test20_%s.csv" % k, "meta.csv", 80, "e2/")
# measure_lightFM("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 80, "e2-1/")
# measure_lightFM("frappe.csv", "train_abs_cold_item.csv", "test_abs_cold_item.csv", "meta.csv", 80, "e2-1/")
# measure_lightFM("frappe.csv", "train_abs_cold_user.csv", "test_abs_cold_user.csv", "meta.csv", 80, "e2-1/")
# measure_lightFM("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 80, "e2-2/")
# measure_lightFM("frappe.csv", "train_abs_cold_item.csv", "test_abs_cold_item.csv", "meta.csv", 80, "e2-2/")
# measure_lightFM("frappe.csv", "train_abs_cold_user.csv", "test_abs_cold_user.csv", "meta.csv", 80, "e2-2/")
# measure_lightFM("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 80, "e2-3/")
# measure_lightFM("frappe.csv", "train_abs_cold_item.csv", "test_abs_cold_item.csv", "meta.csv", 80, "e2-3/")
# measure_lightFM("frappe.csv", "train_abs_cold_user.csv", "test_abs_cold_user.csv", "meta.csv", 80, "e2-3/")
# measure_lightFM("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 80, "e2-4/")
# measure_lightFM("frappe.csv", "train_abs_cold_item.csv", "test_abs_cold_item.csv", "meta.csv", 80, "e2-4/")
# measure_lightFM("frappe.csv", "train_abs_cold_user.csv", "test_abs_cold_user.csv", "meta.csv", 80, "e2-4/")
# measure_lightFM("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_abs_cold_item.csv", "test_abs_cold_item.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_abs_cold_user.csv", "test_abs_cold_user.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.05.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.1.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.2.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.3.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.4.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")
# measure_lightFM("frappe.csv", "train_rel_cold_0.5.csv", "test_warm.csv", "meta.csv", 80, "e2-5/")


# measure_e1()

# recommendations_lightFM("frappe.csv", "train80.csv", "test20.csv", "meta.csv", 80)


# data = fetch_frappe("frappe.csv", "train_warm.csv", "test_warm.csv", "meta.csv")
# print(type(data["item_features"]))
# print(data["id_features"].shape)
# print(data["item_features"])
# print(data["item_labels"])
# print(len(data["items"]))
