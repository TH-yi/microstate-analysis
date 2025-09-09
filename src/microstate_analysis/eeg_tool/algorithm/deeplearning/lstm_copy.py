# import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from scipy.io import loadmat
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, write_append_info
import numpy as np
from eeg_tool.algorithm.deeplearning.dataset import MSeqData
import time
import math
import os
import random
import shutil

def load_config_linux():
    subjects = ['april_02(3)', 'april_08', 'april_15', 'april_16(1)', 'april_16(3)', 'april_18(1)', 'april_18(2)', 'april_22', 'july_30', 'sep_12', 'sep_13', 'eeg_sep_18', 'Feb_18(1)_2014', 'Feb_19(2)_2014', 'Feb_20(2)_2014', 'Mar_12_2014', 'Mar_14(2)_2014', 'april_2(1)', 'april_19(1)', 'april_19(2)', 'april_24', 'Feb_07(1)_2014', 'Feb_18(2)_2014', 'Feb_28(1)_2014', 'Feb_28(2)_2014', 'april_30_2014', 'april_04(1)']
    tasks = ['rest_1', 'rest_2']
    conditions = ['rest', 'pu', 'ig', 'rig', 'ie', 'rie']
    res = {}
    for j in range(1, len(conditions)):
        for i in range(1, 7):
            tasks.append(conditions[j]+"_"+str(i))
    for condition in conditions:
        res[condition] = []
        for task in tasks:
            if task.startswith(condition):
                res[condition].append(task)
    return subjects, tasks, conditions, res

def load_config():
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    res = {}
    for condition in conditions:
        res[condition] = []
        for task in tasks:
            if task.startswith(condition):
                res[condition].append(task)
    return subjects, tasks, conditions, res

def create_single_condition_data(dir_path, condition_name,seq_len, lag, offset):
    data = []
    label = []
    seq_lag_o = []
    seq_label_o = []
    for condition in condition_name:
        #path = dir_path + "\\" + condition + "_seq.mat"
        path = os.path.join(dir_path, condition+"_seq.mat")
        seq_total = loadmat(path)['EEG'].flatten().tolist()
        # lag
        seq_lag = seq_total[:len(seq_total)-lag]
        seq_label = seq_total[lag:]
        seq_size = len(seq_lag)

        # offset
        index = 0
        for i in range(1, seq_size, offset):
            seq_offset = []
            label_offset = []
            if index*offset + seq_len < seq_size:
                seq_offset = seq_lag[index*offset:index*offset+seq_len]
                label_offset = seq_label[index*offset:index*offset+seq_len]
                seq_lag_o.extend(seq_offset)
                seq_label_o.extend(label_offset)
                index = index + 1

        if len(seq_lag_o) % seq_len != 0:
            num = len(seq_lag_o)//seq_len
            seq_lag_o = seq_lag_o[:num*seq_len]
            seq_label_o = seq_label_o[:num*seq_len]
        data.extend(seq_lag_o)
        label.extend(seq_label_o)
        #data.extend(loadmat(path)['EEG'].flatten().tolist())
    return torch.tensor(data), torch.tensor(label)


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # batch_size, seq_len, n_features
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, inputs):
        outputs, (hidden, cell) = self.rnn(inputs)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        # batch_size, seq_len, n_features
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, inputs, hidden, cell):
        inputs = inputs.unsqueeze(1)
        output, (hidden, cell) = self.rnn(inputs, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #src = [batch size, seq_len, feature_size]
        #trg = [batch size, seq_len, feature_size]

        batch_size = trg.shape[0]
        seq_len = trg.shape[1]
        feature_size = trg.shape[2]

        outputs = torch.zeros(batch_size, seq_len, feature_size).to(self.device)

        hidden, cell = self.encoder(src)

        inputs = trg[:, 0, :]
        for t in range(1, seq_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[:, t, :] = output
            inputs = output
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, data, label, clip):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for i, _ in enumerate(data):
        src = data[i]
        trg = label[i]
        optimizer.zero_grad()
        # batch_size, seq_len, feature_size
        output = model(src, trg)
        feature_size = output.shape[-1]
        output = output[:, 1::, :]
        trg = trg[:, 1::, :]
        loss = criterion(output.reshape(-1, feature_size), trg.reshape(-1, feature_size).argmax(1))
        accuracy = calculate_accuracy(output, trg, feature_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()
    return epoch_loss / len(data), epoch_accuracy / len(data)

def evaluate(model, data, label):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, _ in enumerate(data):
            src = data[i]
            trg = label[i]
            output = model(src, trg, 0)
            feature_size = output.shape[-1]
            output = output[:, 1::, :]
            trg = trg[:, 1::, :]
            loss = criterion(output.reshape(-1, feature_size), trg.reshape(-1, feature_size).argmax(1))
            accuracy = calculate_accuracy(output, trg, feature_size)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
    return epoch_loss / len(data), epoch_accuracy / len(data)


def calculate_accuracy(output, trg, feature_size):
    prediction = output.reshape(-1, feature_size).argmax(1)
    label = trg.reshape(-1, feature_size).argmax(1)
    return (prediction == label).sum()/prediction.shape[0]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int (elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    subjects, tasks, conditions, conditions_tasks = load_config_linux()

    #subjects, tasks, conditions, conditions_tasks = load_config()
    path_dir = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\sequences'
    # path_dir = r'/media/concordia/DA081B82081B5D37/Hongjiang/EEG/sequences'
    # path_save_dir = r'/media/concordia/DA081B82081B5D37/Hongjiang/EEG/seq_200_0.75'
    path_save_dir = r'D:\EEGdata\reconstuction_sequences\single_sub_reconstruction\lstm\seq_400_0.75'
    N_MICROSTATE = 7
    BATCH_SIZE = 32
    SEQ_LEN = 100
    LAG = 0
    RATIO = 0.25
    OFFSET = int(SEQ_LEN*RATIO)
    INPUT_DIM = N_MICROSTATE + 1
    OUTPUT_DIM = N_MICROSTATE + 1
    HID_DIM = 64
    N_LAYERS = 2
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    N_EPOCHS = 250
    CLIP = 1.0

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1')
    for subject in subjects[7::]:
        #sub_path = path_save_dir + "\\" + subject
        sub_path = os.path.join(path_save_dir, subject)
        if os.path.exists(sub_path):
            print('%s: exists'%sub_path)
            shutil.rmtree(sub_path)

        os.mkdir(sub_path)
        parameters_count = {}
        for condition in conditions:
            # parameters_count_path = sub_path + "\\" + condition + '_parameters' +'.json'
            # epochs_info_path = sub_path + "\\" + condition + '_train_epochs' +'.json'
            # models_path = sub_path + "\\" + condition + '_models' +'.pt'
            # valid_path = sub_path + "\\" + condition + '_valid_epochs' +'.json'

            parameters_count_path = os.path.join(sub_path, condition+'_parameters'+'.json')
            epochs_info_path = os.path.join(sub_path, condition+ '_train_epochs'+ '.json')
            models_path = os.path.join(sub_path, condition+ '_models'+ '.pt')
            valid_path =os.path.join(sub_path, condition+ '_valid_epochs'+'.json')

            #raw_data = create_single_condition_data(path_dir + "\\" + subject, conditions_tasks[condition])
            raw_data, raw_label = create_single_condition_data(os.path.join(path_dir, subject), conditions_tasks[condition], SEQ_LEN, LAG, OFFSET)
            data_set = MSeqData(raw_data, raw_label, INPUT_DIM, SEQ_LEN, BATCH_SIZE, device)
            best_train_loss = float('inf')
            enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
            dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
            model = Seq2Seq(enc, dec, device).to(device)
            # model = nn.DataParallel(model)
            model.apply(init_weights)
            # print(f'The model has {count_parameters(model):,} trainable parameters')
            parameters_count[condition] = count_parameters(model)
            write_append_info(sub_path, parameters_count_path, parameters_count)
            restart_time = True
            for epoch in range(N_EPOCHS):
                if restart_time:
                    start_time = time.time()
                    restart_time = False
                train_loss, train_accuracy = train(model, data_set.train_data, data_set.train_label, CLIP)
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    torch.save(model.state_dict(), models_path)
                if epoch % 10 == 0:
                    epoch_info = {'epoch_num': epoch, 'epoch_time': str(epoch_mins) + " m " + str(epoch_secs) + " s",
                                  'train_loss': train_loss, 'train_accuracy': train_accuracy}
                    print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
                    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}')
                    write_append_info(sub_path, epochs_info_path, epoch_info)
                    restart_time = True


            model.load_state_dict(torch.load(models_path))
            valid_loss, valid_accuracy = evaluate(model, data_set.test_data, data_set.test_label)
            epoch_info = {'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy}
            write_append_info(sub_path, valid_path, epoch_info)
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Accuracy: {valid_accuracy:.3f}')
