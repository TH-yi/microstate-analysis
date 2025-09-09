import torch 
from torch.utils.data import Dataset,DataLoader


class MSeqData(Dataset):
    def __init__(self, data, label, feature_size, seq_len, batch_size, device):
        self.batch_size = batch_size
        self.data = torch.reshape(data, (-1, seq_len))
        self.label = torch.reshape(label, (-1, seq_len))
        self.train_data, self.test_data, self.train_label, self.test_label = self.split_train_test_dataset()
        self.split_batch_dataset()
        self.train_data, self.test_data, self.train_label, self.test_label = self.add_init_token(feature_size, device)
        #self.train_label = self.train_data
        #self.test_label = self.test_data

    def __getitem__(self, index):
        return self.train_data, self.test_data, self.train_label, self.test_label

    def __len__(self):
        return len(self.data)

    def split_train_test_dataset(self, train_ratio=0.8, seed=42):
        length = self.data.shape[0]
        train_size = int(train_ratio * length)
        test_size = int((1-train_ratio)*length)
        temp = torch.utils.data.random_split(self.data[0:train_size+test_size, :], [int(train_ratio*length), int((1-train_ratio)*length)], generator=torch.Generator().manual_seed(seed))
        train_indices = temp[0].indices
        test_indices = temp[1].indices
        train_data = self.data[0:train_size+test_size, :][train_indices]
        test_data = self.data[0:train_size+test_size, :][test_indices]
        train_label = self.label[0:train_size + test_size, :][train_indices]
        test_label = self.label[0:train_size + test_size, :][test_indices]
        return train_data, test_data, train_label, test_label

    def encode_one_hot(self, n_microstates, device):
        self.train_data = torch.nn.functional.one_hot(self.train_data, n_microstates).float().to(device)
        self.test_data = torch.nn.functional.one_hot(self.test_data, n_microstates).float().to(device)

    def split_batch_dataset(self):
        n_batch_train = self.train_data.shape[0] // self.batch_size
        n_batch_test = self.test_data.shape[0] // self.batch_size
        self.train_data = torch.chunk(self.train_data[0:n_batch_train*self.batch_size, :], n_batch_train)
        self.test_data = torch.chunk(self.test_data[0:n_batch_test*self.batch_size, :], n_batch_test)
        self.train_label= torch.chunk(self.train_label[0:n_batch_train*self.batch_size, :], n_batch_train)
        self.test_label = torch.chunk(self.test_label[0:n_batch_test*self.batch_size, :], n_batch_test)


    def add_init_token(self, feature_size, device):
        def init_token(data):
            res = []
            token = torch.ones(self.batch_size, 1) * (feature_size -1)
            for index, value in enumerate(data):
                temp = torch.cat((token, value), dim=1).long()
                # temp = torch.cat((temp, token), dim=1).long()
                torch_temp = torch.nn.functional.one_hot(temp, feature_size).float().to(device)
                res.append(torch_temp)
            return res
        return init_token(self.train_data), init_token(self.test_data), init_token(self.train_label), init_token(self.test_label)

if __name__ == '__main__':
    data = MSeqData(torch.arange(100), torch.arange(100), 10)
    data_loader = DataLoader(dataset=data, batch_size=2)

    for i, (data, label) in enumerate(data_loader):
        print(i, data, label)