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
import torch.nn.functional as F
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
#from sklearn.tree import DecisionTreeClassifier

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
        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, inputs):
        outputs, hidden = self.rnn(inputs)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        # batch_size, seq_len, n_features
        self.rnn = nn.GRU(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, inputs, hidden):
        inputs = inputs.unsqueeze(1)
        output, hidden = self.rnn(inputs, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


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
            output, hidden  = self.decoder(inputs, hidden)
            outputs[:, t, :] = output
            inputs = output
        return outputs

# class ClassifierCNN(nn.Module):
#     def __init__(self, in_c, n_classes):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_c, 64, kernel_size= 3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(64, momentum=0.01)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size= 3, stride=1, padding=1)
#         # self.bn2 = nn.BatchNorm1d(64, momentum=0.5)
#         self.fc1 = nn.Linear(64*128,n_classes).cuda()
#
#     def forward(self,x):
#         x = self.conv1(x)
#         # x = self.bn1(x)
#         x = F.relu(x)
#
#         x = self.conv2(x)
#         # x = self.bn2(x)
#         x = F.relu(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

class ClassifierCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(64*6,512).cuda()
        self.fc2 = nn.Linear(512, 512).cuda()
        self.fc3 = nn.Linear(512, n_classes).cuda()

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        return x


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

def train_classifier(model, data, label, clip):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    output_label = []
    for i, _ in enumerate(data):
        src = data[i]
        input = torch.reshape(src,(src.shape[0],6,src.shape[1])).to(device)
        trg = torch.tensor(label[i]).to(device)
        #target = torch.reshape(trg,(src.shape[0]))
        optimizer.zero_grad()
        # batch_size, seq_len, feature_size
        output = model(input)
        output_label.append(output.argmax(1).tolist())
        loss = criterion(output, trg)
        accuracy = calculate_class_acc(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()
    return epoch_loss / len(data), epoch_accuracy / len(data), output_label

def evaluate_classifier(model, data, label):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    output_label = []
    with torch.no_grad():
        for i, _ in enumerate(data):
            src = data[i]
            input = torch.reshape(src, (src.shape[0], 6, src.shape[1])).to(device)
            trg = torch.tensor(label[i]).to(device)
            output = model(input)
            output_label.append(output.argmax(1).tolist())
            loss = criterion(output, trg)
            accuracy = calculate_class_acc(output, trg)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
    return epoch_loss / len(data), epoch_accuracy / len(data), output_label

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

def calculate_feature_target(sub_path,conditions,data,class_label,class_count,hidden_dim,batch_size):
    feature = []
    target = []
    with torch.no_grad():
        for i,_ in enumerate(data):
            src = data[i]
            hidden_all = torch.empty(batch_size,hidden_dim,class_count)
            for model_index, condition in enumerate(conditions):
                models_path = os.path.join(sub_path, condition + '_models' + '.pt')
                model.load_state_dict(torch.load(models_path))
                model.eval()
                hidden,_ = model.encoder(src)
                #hidden_layer_2 = hidden[1:,:]
                hidden_all[:,:,model_index] = hidden
            label_list = [class_label]*len(src)
            #label_en = torch.eye(class_count, dtype = torch.long)[label_list,:]
            feature.append(hidden_all)
            target.append(label_list)
    return feature, target

def calculate_class_acc(output, target):
    prediction = output.argmax(1)
    return (prediction == target).sum()/prediction.shape[0]


def calculate_accuracy(output, trg, feature_size):
    prediction = output.reshape(-1, feature_size).argmax(1)
    label = trg.reshape(-1, feature_size).argmax(1)
    return (prediction == label).sum()/prediction.shape[0]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int (elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

# def create_ml_dataset(dataset):
#     torch.stack(dataset_feature_train,0).reshape(-1,64,6)



if __name__ == '__main__':
    subjects, tasks, conditions, conditions_tasks = load_config_linux()

    #subjects, tasks, conditions, conditions_tasks = load_config()
    #path_dir = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\sequences'
    path_dir = r'/media/concordia/DA081B82081B5D37/Hongjiang/EEG/sequences'
    path_save_dir = r'/media/concordia/DA081B82081B5D37/Hongjiang/EEG/gru/seq_400_0.75'
    path_save_dir_cl = r'/media/concordia/DA081B82081B5D37/Hongjiang/EEG/gru/seq_400_0.75_cl_ml'
    #path_save_dir = r'D:\EEGdata\reconstuction_sequences\single_sub_reconstruction\seq_200ms'
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
    N_EPOCHS = 50
    CLIP = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for subject in subjects:
        #sub_path = path_save_dir + "\\" + subject
        sub_path = os.path.join(path_save_dir, subject)
        sub_path_cl = os.path.join(path_save_dir_cl, subject)
        if os.path.exists(sub_path):
            print('%s: exists'%sub_path)
            #shutil.rmtree(sub_path)
        if not os.path.exists(sub_path_cl):
            os.makedirs(sub_path_cl)
        parameters_count_path_cl = os.path.join(sub_path_cl,  'parameters' + '.json')
        epochs_info_path_cl = os.path.join(sub_path_cl,  'train_epochs' + '.json')
        models_path_cl = os.path.join(sub_path_cl,  'models' + '.pt')
        valid_path_cl = os.path.join(sub_path_cl,  'valid_epochs' + '.json')
        label_path_cl = os.path.join(sub_path_cl, 'labels' + '.json')
        parameters_count = {}
        label_save = {}
        dataset_feature_train = []
        dataset_target_train = []
        dataset_feature_test = []
        dataset_target_test = []

        for class_label, condition in enumerate(conditions):
            # parameters_count_path = sub_path + "\\" + condition + '_parameters' +'.json'
            # epochs_info_path = sub_path + "\\" + condition + '_train_epochs' +'.json'
            # models_path = sub_path + "\\" + condition + '_models' +'.pt'
            # valid_path = sub_path + "\\" + condition + '_valid_epochs' +'.json'

            parameters_count_path = os.path.join(sub_path, condition+'_parameters'+'.json')
            epochs_info_path = os.path.join(sub_path, condition+ '_train_epochs'+ '.json')

            valid_path =os.path.join(sub_path, condition+ '_valid_epochs'+'.json')

            #raw_data = create_single_condition_data(path_dir + "\\" + subject, conditions_tasks[condition])
            raw_data, raw_label = create_single_condition_data(os.path.join(path_dir, subject), conditions_tasks[condition], SEQ_LEN, LAG, OFFSET)
            data_set = MSeqData(raw_data, raw_label, INPUT_DIM, SEQ_LEN, BATCH_SIZE, device)
            best_train_loss = float('inf')
            enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
            dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
            model = Seq2Seq(enc, dec, device).to(device)
            feature_train = torch.empty(BATCH_SIZE,HID_DIM)
            feature_test = torch.empty(BATCH_SIZE, HID_DIM)

            feature_train, target_train = calculate_feature_target(sub_path, conditions, data_set.train_data,class_label,len(conditions),HID_DIM,BATCH_SIZE)
            feature_test, target_test = calculate_feature_target(sub_path, conditions, data_set.test_data, class_label,len(conditions),HID_DIM,BATCH_SIZE)
            dataset_feature_train.extend(feature_train)
            dataset_target_train.extend(target_train)
            dataset_feature_test.extend(feature_test)
            dataset_target_test.extend(target_test)

        dataset_feature_train = torch.stack(dataset_feature_train, 0).reshape(-1, 64*6).cpu().numpy()
        dataset_feature_test = torch.stack(dataset_feature_test, 0).reshape(-1, 64*6).cpu().numpy()
        dataset_target_train = np.asarray(dataset_target_train).reshape(-1)
        dataset_target_test = np.asarray(dataset_target_test).reshape(-1)

        clf_nb = GaussianNB()
        # scaler = MinMaxScaler()
        # scaler.fit(dataset_feature_train)
        # dataset_feature_train = scaler.transform(dataset_feature_train)
        # dataset_feature_test = scaler.transform(dataset_feature_test)

        clf_nb.fit(dataset_feature_train, dataset_target_train)
        prediction_nb = clf_nb.predict(dataset_feature_test)
        accuracy_nb = (prediction_nb == dataset_target_test).sum()/prediction_nb.shape[0]
        print('Accuracy_nb:\n')
        print(accuracy_nb)
        # clf_dt = DecisionTreeClassifier(random_state=0)
        # clf_dt.fit(dataset_feature_train, dataset_target_train)
        # prediction_dt = clf_dt.predict(dataset_feature_test)
        # accuracy_dt = (prediction_dt == dataset_target_test).sum()/prediction_dt.shape[0]
        # print('Accuracy_dt:\n')
        # print(accuracy_dt)

        clf_svm = svm.LinearSVC(multi_class='ovr')
        clf_svm.fit(dataset_feature_train, dataset_target_train)
        prediction_svm = clf_svm.predict(dataset_feature_test)
        accuracy_svm = (prediction_svm == dataset_target_test).sum() / prediction_svm.shape[0]
        print('Accuracy_svm:\n')
        print(accuracy_svm)

        label_save['output_test_nb'] = prediction_nb.tolist()
        label_save['label_test'] = dataset_target_test.tolist()
        label_save['output_test_svm'] = prediction_svm.tolist()

        write_append_info(sub_path_cl, label_path_cl, label_save)


        # seed = random.random()
        # random.seed(seed)
        # random.shuffle(dataset_target_train)
        # random.seed(seed)
        # random.shuffle(dataset_feature_train)
