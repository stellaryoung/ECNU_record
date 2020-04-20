from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import math
import numpy as np


def UserCF_prediction(user_id, item_id, item_user_dic, user_item_dic, item_map_dic,
                      user_similarity,
                      user_item_matrix_np,
                      K_num=3):
    if item_id in item_user_dic:  # 判断物品是否有评分记录
        # 用户,物品在相似度矩阵的索引值
        item_index = item_map_dic[item_id]
        user_index = user_id - 1

        user_list = item_user_dic[item_id]  # 给物品评过分的用户列表
        # 按照相似度大小给用户排序
        user_sim_list = []  # (id,similiarity)
        my_func = lambda e: e[1]
        for like_user_id in user_list:
            tmp_s = user_similarity[like_user_id - 1][user_index]
            user_sim_list.append((like_user_id, tmp_s))
        user_sim_list.sort(key=my_func, reverse=True)
        # print(id_sim_list)
        count = 0
        pred = 0
        deno = 0
        for tmp_id, tmp_sim in user_sim_list:
            pred += user_item_matrix_np[tmp_id - 1, item_index] * tmp_sim
            deno += tmp_sim
            count += 1
            if count == K_num:  # 只取最为相似的K个物品
                break
        if deno == 0: return 3
        return pred / deno  # 加权求和得到最终评分
    else:
        return 3


def ItemCF_prediction(user_id, item_id,
                      item_user_dic,
                      user_item_dic,
                      item_map_dic,
                      item_similarity,
                      user_item_matrix_np,
                      K_num=3):
    if item_id in item_user_dic:  # 判断物品是否有评分记录
        # 物品在相似度矩阵的索引值
        item_index = item_map_dic[item_id]
        like_list = user_item_dic[user_id]  # 获得用户已经评分过的物品列表

        # 找出最为相似的K个物品
        id_sim_list = []  # (id,similiarity)
        my_func = lambda e: e[1]
        for like_item_id in like_list:
            tmp_s = item_similarity[item_map_dic[like_item_id]][item_index]
            id_sim_list.append((like_item_id, tmp_s))
        id_sim_list.sort(key=my_func, reverse=True)
        # print(id_sim_list)
        count = 0
        pred = 0
        deno = 0
        for tmp_id, tmp_sim in id_sim_list:
            # pred += user_item_matrix.loc[user_id,tmp_id]*tmp_sim
            pred += user_item_matrix_np[user_id - 1, item_map_dic[tmp_id]] * tmp_sim
            deno += tmp_sim
            count += 1
            if count == K_num:  # 只取最为相似的K个物品
                break
        if deno == 0: return 3
        return pred / deno  # 加权求和得到最终评分，四舍五入取整
    else:
        return 3


def cal_similarity(user_item_matrix):
    data_matrix = user_item_matrix.values  # dataframe-->numpy
    # data_matrix:((n_users, n_items))              # 用户-物品矩阵(评分)矩阵
    user_similarity = pairwise_distances(data_matrix, metric='cosine')  # 用户向量的相似度计算
    item_similarity = pairwise_distances(data_matrix.T, metric='cosine')  # 物品向量之间相似度计算
    return user_similarity, item_similarity


def cal_item_similarity(user_item_dic, item_user_dic, user_item_matrix, item_map_dic):
    item_num = len(item_user_dic)
    user_num = len(user_item_dic)
    item_similarity = np.zeros((item_num, item_num))
    for user in tqdm(range(1, user_num + 1)):
        for item1 in user_item_dic[user]:
            for item2 in user_item_dic[user]:
                if item1 != item2:
                    item_similarity[item_map_dic[item1]][item_map_dic[item2]] += 1

    item_id_list = user_item_matrix.columns
    for item1 in tqdm(item_id_list):
        for item2 in item_id_list:
            item_raw1 = int(item1)
            item_raw2 = int(item2)
            item_new1 = item_map_dic[int(item1)]
            item_new2 = item_map_dic[int(item2)]
            if item_similarity[item_new1][item_new2] == 0:
                continue
            mul = math.sqrt(len(item_user_dic[item_raw1]) * len(item_user_dic[item_raw2]))
            if item1 != item2:
                item_similarity[item_new1][item_new2] = \
                    item_similarity[item_new1][item_new2] / mul
    return item_similarity


def cal_user_similarity(user_item_dic, item_user_dic, user_item_matrix):
    item_num = len(item_user_dic)
    user_num = len(user_item_dic)
    tmp_list = user_item_matrix.columns
    item_id_list = []
    for tmp in tmp_list:
        item_id_list.append(int(tmp))
    user_similarity = np.zeros((user_num, user_num))
    for item in tqdm(item_id_list):
        this_item_num = len(item_user_dic[item])
        for user1 in item_user_dic[item]:
            for user2 in item_user_dic[item]:
                if user1 != user2:
                    user_similarity[user1 - 1][user2 - 1] += 1 / math.log(1 + this_item_num)

    for user1 in tqdm(range(1, user_num + 1)):
        for user2 in range(1, user_num + 1):
            if user_similarity[user1 - 1][user2 - 1] == 0:
                continue
            mul = math.sqrt(len(user_item_dic[user1]) * len(user_item_dic[user2]))
            if user1 != user2:
                user_similarity[user1 - 1][user2 - 1] = \
                    user_similarity[user1 - 1][user2 - 1] / mul
    return user_similarity
