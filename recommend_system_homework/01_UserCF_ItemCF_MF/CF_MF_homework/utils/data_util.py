import numpy as np
import pandas as pd
import tqdm


def load_ml_1m(path, path_name='ratings.dat'):
    data_col = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_table(path + 'ratings.dat', sep='::', header=None, names=data_col, engine='python')
    data.drop('timestamp', axis=1, inplace=True)  # 去除最后一列
    return data


def split_dataframe(df):
    user_ids = df['user_id'].unique()
    user_n = len(user_ids)
    train_df = pd.DataFrame(columns=('user_id', 'item_id', 'rating'))
    test_df = pd.DataFrame(columns=('user_id', 'item_id', 'rating'))
    # 按照8：2划分训练集、测试集，这里随机拿出每个用户的80%评分记录组合成训练集，剩余20%是测试集
    for tmp_id in tqdm(range(1, user_n + 1)):
        tmp_df = df[df.user_id == tmp_id]
        tmp_train = tmp_df.sample(frac=0.8)
        tmp_test = tmp_df.drop(tmp_train.index)
        train_df = train_df.append(tmp_train, ignore_index=True)
        test_df = test_df.append(tmp_test, ignore_index=True)
    return train_df, test_df


# build user-item dic，  user_id:item_list
# build item-user dic,  item_id:user_list
def build_dictionary(train_df):
    user_item_dic = {}
    item_user_dic = {}
    for row_record in train_df.itertuples():
        user_id = getattr(row_record, 'user_id')
        item_id = getattr(row_record, 'item_id')
        rating = getattr(row_record, 'rating')
        # build user-->item dictionary
        if user_id in user_item_dic:
            user_item_dic.get(user_id).append(item_id)
        else:
            user_item_dic[user_id] = [item_id]
        # build item-user dictionary
        if item_id in item_user_dic:
            item_user_dic.get(item_id).append(user_id)
        else:
            item_user_dic[item_id] = [user_id]

    return user_item_dic, item_user_dic


def user_item_score(df):
    row_num = df.shape[0]
    user_names = df['user_id'].unique()
    item_names = df['item_id'].unique()
    user_n = len(user_names)
    item_n = len(item_names)
    user_item_matrix = pd.DataFrame(np.zeros((user_n, item_n)), index=user_names, columns=item_names)
    iterable = iter(df.itertuples())
    for num in tqdm(range(1, row_num + 1)):
        i = next(iterable)
        user_item_matrix.loc[getattr(i, 'user_id'), getattr(i, 'item_id')] = getattr(i, 'rating')
    return user_item_matrix


# 将测试数据转换为3个列表
def extract_test_df(df):
    test = df.values
    test_user_id_list = []
    test_item_id_list = []
    test_rating_list = []
    for i in range(test.shape[0]):
        test_user_id_list.append(test[i][0])
        test_item_id_list.append(test[i][1])
        test_rating_list.append(test[i][2])
    return test_user_id_list, test_item_id_list, test_rating_list


def extract_df(df):
    data = df.values
    user_id_list = []
    item_id_list = []
    rating_list = []
    for i in range(data.shape[0]):
        user_id_list.append(data[i][0])
        item_id_list.append(data[i][1])
        rating_list.append(data[i][2])
    return user_id_list, item_id_list, rating_list
