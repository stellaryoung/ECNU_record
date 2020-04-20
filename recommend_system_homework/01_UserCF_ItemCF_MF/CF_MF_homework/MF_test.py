from utils.data_util import build_dictionary, extract_df
import pandas as pd
import torch
import os
import glob   # lists of files matching given patterns, just like the Unix shell
import torch.nn as nn
from MF.MF_model import LFM
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

# item_id,user_id list ----> item_index,user_index list
def id_to_embed_index(user_test_list, item_test_list, item_user_dic, user_item_matrix):
    itemID_train_list = user_item_matrix.columns
    print(len(itemID_train_list))
    # -------------未在训练集中出现的测试集物品id编号-----------------
    undefined_index = len(itemID_train_list)
    # -------------在训练集中出现的物品id编号-------------------------
    item_map_dic = {}  # 用于用户评分矩阵是无序的，建立item_id-->index的映射
    for i, col_id in enumerate(itemID_train_list):
        item_map_dic[int(col_id)] = i

    # -------------获取在测试集出现，未在训练集中出现的物品id----------
    rare_item_id_list = []
    for item_id in item_test_list:
        if item_id not in item_user_dic:
            rare_item_id_list.append(item_id)
    print(rare_item_id_list)

    # -------------将测试集中的id转换为编号----------------------
    user_embed_index = []
    item_embed_index = []
    for user_id in user_test_list:  # user_id = index+1
        user_embed_index.append(user_id - 1)

    for item_id in item_test_list:
        if item_id in item_user_dic:
            item_embed_index.append(item_map_dic[item_id])
        else:
            item_embed_index.append(undefined_index)
    return user_embed_index, item_embed_index, item_map_dic


def id_to_embed_index2(user_list, item_list, item_map_dic):
    user_embed_index = []
    item_embed_index = []
    for user_id in user_list:  # user_id = index+1
        user_embed_index.append(user_id - 1)
    for item_id in item_list:
        item_embed_index.append(item_map_dic[item_id])
    return user_embed_index, item_embed_index


if __name__ == "__main__":
    print("MF_bias model!")
    path = './data/'
    train_df = pd.read_csv(path + 'train_df.csv')
    test_df = pd.read_csv(path + 'test_df.csv')
    user_item_matrix = pd.read_csv(path + 'user_item_matrix.csv')
    user_item_matrix_np = user_item_matrix.values

    test_user_id_list, test_item_id_list, test_rating_list = extract_df(test_df)
    train_user_id_list, train_item_id_list, train_rating_list = extract_df(train_df)
    user_item_dic, item_user_dic = build_dictionary(train_df)
    test_user_embed_index, test_item_embed_index, item_map_dic = id_to_embed_index(test_user_id_list, test_item_id_list,
                                                                                   item_user_dic, user_item_matrix)
    train_user_embed_index, train_item_embed_index = id_to_embed_index2(train_user_id_list, train_item_id_list,
                                                                        item_map_dic)

    trian_u = torch.LongTensor(train_user_embed_index)
    train_i = torch.LongTensor(train_item_embed_index)
    train_r = torch.FloatTensor(train_rating_list)

    test_u = torch.LongTensor(test_user_embed_index)
    test_i = torch.LongTensor(test_item_embed_index)
    test_r = torch.FloatTensor(test_rating_list)

    batch_size_tr = 10000
    train_dataset = RateDataset(trian_u, train_i, train_r)
    train_data = DataLoader(train_dataset, batch_size=batch_size_tr)

    batch_size_te = 5000
    test_dataset = RateDataset(test_u, test_i, test_r)
    test_data = DataLoader(test_dataset, batch_size=batch_size_te)

    num_user = 6040
    num_item = len(item_user_dic) + 1
    hidden = 10
    mu = 3
    lr = 0.1
    # weight_decay = 0.01
    epoch_num = 500
    early_stop = 5
    train_flag = False

    model_dir = './model_dir/'
    model = LFM(num_user, num_item, hidden, mu)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if train_flag == True:
        loss_list = []
        best_epoch = 0
        bad_count = 0
        # 模型每次训练前没有进行梯度更新zero_grad，导致模型使用了已有参数进行了训练
        best_loss = float('inf')
        model.train()
        for epoch in range(epoch_num):
            loss_sum = 0
            for bid, batch in tqdm(enumerate(train_data)):
                u, i, r = batch[0], batch[1], batch[2]
                r = r.float()
                # forward pass
                preds = model(u, i)
                loss = criterion(preds.squeeze(dim=1), r)
                loss_sum += loss.item()
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print('Epoch [{}/30], Loss: {:.4f}'.format(epoch + 1, loss.item()))
            if bad_count == early_stop:
                print("early stop at {}".format(epoch))
                break
            bad_count += 1
            tmp_loss = loss_sum / (801)
            print(tmp_loss)
            loss_list.append(tmp_loss)
            if tmp_loss < best_loss:
                bad_count = 0
                print("New record!!!!")
                best_loss = tmp_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_dir + '{}.pkl'.format(epoch))
                files = glob.glob(model_dir + '*.pkl')
                for file in files:
                    tmp = file.split('/')[-1]
                    tmp = tmp.split('.')[0]
                    epoch_nb = int(tmp)
                    if epoch_nb < best_epoch:
                        os.remove(file)
    else:
        best_epoch = 1093
        print('Loading model!!!')
        model.load_state_dict(torch.load(model_dir + '{}.pkl'.format(best_epoch)))
        model.eval()
        pred_list = []
        for bid, batch in tqdm(enumerate(test_data)):
            u, i, r = batch[0], batch[1], batch[2]
            preds = model(u, i)
            pred_list.extend(preds.squeeze(dim=1).detach().numpy().tolist())
        MSE_loss = mean_squared_error(pred_list, test_rating_list)
        print("MSE of test data: {}".format(MSE_loss))