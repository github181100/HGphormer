import time
import math
import torch
import numpy as np
from config import Config
from loader import Loader
from read_graph import load_dataset
from lr import PolynomialDecayLR
# from model_ import TransformerModel
from model import TransformerModel
from utils import laplacian_positional_encoding, cal_accuracy, multi_hop_adj, hierarchy_sampling, hop_nodes_dict, re_features, cal_average_time, load_sp
from early_stop import EarlyStopping, Stop_args
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

"""
模型训练主程序
"""

# Training Config
dataset_name = 'Citeseer'
config_file = "./config/" + str(dataset_name) + ".ini"
config = Config(config_file)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
train_split = 0.48
valid_split = 0.32
test_split = 0.2

# Load and Pre-treatment
adj, features, labels, nclass = load_dataset(dataset_name)

hop_adjs = multi_hop_adj(adj, config.sample_hop)

multi_hop_list = list(map(hop_nodes_dict, hop_adjs))
features = re_features(features)  # padding

laplace_emb = laplacian_positional_encoding(adj, config.laplace_dim)
config.n_class = nclass
config.input_dim = features.shape[1]

full_nodes = torch.arange(labels.shape[0])
full_dataset = TensorDataset(full_nodes, labels)
train_size = math.ceil(labels.shape[0] * train_split)
valid_size = math.ceil(labels.shape[0] * valid_split)
test_size = labels.shape[0] - (train_size + valid_size)

best_acc_list = []
epoch_time_list = []
seed_set = load_sp(dataset_name)

for split_index in range(10):
    train_dataloader, valid_dataloader, test_dataloader = Loader(full_dataset, train_size, valid_size, test_size, config.batch_size,seed_set[split_index])
    model = TransformerModel(config).to(device)

    # print(model)
    # print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=config.warmup_updates,
                    tot_updates=config.tot_updates,
                    lr=config.peak_lr,
                    end_lr=config.end_lr,
                    power=1.0,
                )

    def train_valid_epoch(epoch):
        model.train()
        loss_train_b = []
        acc_train_b = 0
        epoch_time = 0
        for _, item in enumerate(train_dataloader):
            nodes= item[0]
            labels = item[1].to(device)
            b_data = hierarchy_sampling(nodes, multi_hop_list, config.sample_num)
            src_mask = (b_data != -1).unsqueeze(-2)
            fea_data = features[b_data].to(device)
            pe_data = laplace_emb[b_data].to(device)
            optimizer.zero_grad()
            start_time = time.time()
            loss_train, output = model(fea_data, pe_data, labels)
            loss_train_b.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            epoch_time += time.time() - start_time
            lr_scheduler.step()
            acc_train = cal_accuracy(output, labels)
            acc_train_b += acc_train

        model.eval()
        loss_val_b = []
        acc_val = 0
        for _, item in enumerate(valid_dataloader):
            nodes = item[0]
            labels = item[1].to(device)
            b_data = hierarchy_sampling(nodes, multi_hop_list, config.sample_num)
            fea_data = features[b_data].to(device)
            pe_data = laplace_emb[b_data].to(device)
            loss_val, output = model(fea_data, pe_data, labels)
            loss_val_b.append(loss_val.item())
            acc_val += cal_accuracy(output, labels)

        loss_test_b = []
        acc_test = 0
        test_output  = []
        test_labels = []
        for _, item in enumerate(test_dataloader):
            nodes = item[0]
            b_labels = item[1].to(device)
            b_data = hierarchy_sampling(nodes, multi_hop_list, config.sample_num)
            fea_data = features[b_data].to(device)
            pe_data = laplace_emb[b_data].to(device)
            loss_test, b_output = model(fea_data, pe_data, b_labels)
            loss_test_b.append(loss_test.item())
            acc_test += cal_accuracy(b_output, b_labels)
            test_output.append(b_output)
            test_labels.append(b_labels)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(np.mean(loss_train_b)),
              'acc_train: {:.4f}'.format(acc_train_b / train_size),
              'epoch_time: {:.4f}'.format(epoch_time)
              )

        return (acc_val / valid_size), (acc_test / test_size), epoch_time

    # stopping_args = Stop_args(patience=config.patience, max_epochs=config.epochs)
    # early_stopping = EarlyStopping(model, **stopping_args)
    best_val_acc = 0
    best_test_acc = 0
    best_acc = 0
    total_time = []
    bad_counter = 0
    for epoch in range(config.epochs):
        val_acc, test_acc, e_time = train_valid_epoch(epoch)
        total_time.append(e_time)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if max(best_val_acc, best_test_acc) > best_acc:
            best_acc = max(best_val_acc, best_test_acc)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 50:
            break
        # if early_stopping.check([acc_val, loss_val], epoch):
        #     break

    mean_epoch_time = cal_average_time(total_time)
    best_acc_list.append(best_acc)
    epoch_time_list.append(mean_epoch_time)
    print("Optimization Finished!")
    print(f"Split{split_index + 1}‘s best acc : {best_acc_list[split_index]}, average time:{mean_epoch_time}")

    # # Restore best model
    # print('Loading {}th epoch'.format(early_stopping.best_epoch + 1))
    # model.load_state_dict(early_stopping.best_state)

    # test()

print(f'best_acc_list:{best_acc_list}')
print(f'Acc: {100 * np.mean(best_acc_list)} ± {100 * np.std(best_acc_list)}')
print(f'epoch_time_list:{epoch_time_list}')
print(f'Time: {np.mean(epoch_time_list)} ± {np.std(epoch_time_list)}')

