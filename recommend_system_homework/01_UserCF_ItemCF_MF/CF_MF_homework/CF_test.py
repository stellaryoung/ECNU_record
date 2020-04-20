from utils.data_util import load_ml_1m
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.data_util import build_dictionary, extract_test_df
from CF.CF_model import cal_item_similarity,cal_user_similarity,UserCF_prediction,ItemCF_prediction
from sklearn.metrics import mean_squared_error


def predict(test_user_id_list, test_item_id_list, test_rating_list, CF_F,
            item_user_dic, user_item_dic, item_map_dic,similarity,user_item_matrix_np,k_num_list):
    loss_list = []
    for K in tqdm(k_num_list):
        pred_rating_list = []
        for num in range(len(test_user_id_list)):  # (2:41,k=3) (1:11与1：03)
            pred_rating_list.append(CF_F(test_user_id_list[num],
                                         test_item_id_list[num],
                                         item_user_dic, user_item_dic,
                                         item_map_dic,
                                         similarity,
                                         user_item_matrix_np,
                                         K_num=K))
        MSE_loss = mean_squared_error(test_rating_list, pred_rating_list)
        loss_list.append(MSE_loss)
    return loss_list


if __name__ == "__main__":
    path = './data/'
    train_df = pd.read_csv(path + 'train_df.csv')
    test_df = pd.read_csv(path + 'test_df.csv')
    user_item_matrix = pd.read_csv(path + 'user_item_matrix.csv')
    user_item_matrix_np = user_item_matrix.values
    user_item_dic, item_user_dic = build_dictionary(train_df)
    test_user_id_list, test_item_id_list, test_rating_list = extract_test_df(test_df)
    item_id_list = user_item_matrix.columns
    item_map_dic = {}  # 用于用户评分矩阵是无序的，建立item_id-->index的映射
    for i, col_id in enumerate(item_id_list):
        item_map_dic[int(col_id)] = i

    user_similarity = cal_user_similarity(user_item_dic,item_user_dic,user_item_matrix)
    item_similarity = cal_item_similarity(user_item_dic, item_user_dic, user_item_matrix, item_map_dic)
    print(user_similarity.shape)
    print(item_similarity.shape)
    k_num_list = [5,10,15,20,25,30,45]     # 最为相似的用户/物品数目K
    MSE_list_user = predict(test_user_id_list, test_item_id_list, test_rating_list, UserCF_prediction,
                            item_user_dic, user_item_dic, item_map_dic,
                            user_similarity, user_item_matrix_np,
                            k_num_list)
    MSE_list_item = predict(test_user_id_list, test_item_id_list, test_rating_list, ItemCF_prediction,
                            item_user_dic, user_item_dic, item_map_dic,
                            item_similarity, user_item_matrix_np,
                            k_num_list)
    print("User CF MSE:")
    print(MSE_list_user)
    print("Item CF MSE:")
    print(MSE_list_item)