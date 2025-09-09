def mark_bad_epochs(self, drop_epoch=0.25, n_times=2, threshold=None, filtered_data=None):
    self.start_point = []
    self.end_point = []
    self.epochs_point = []
    tasks_values = list(self.trial_info.values())
    np_tasks_values = np.array(tasks_values)
    self.tasks_start = list(np_tasks_values[:, 0])
    self.tasks_end = list(np_tasks_values[:, 1])
    self.tasks_start = list(map(int, self.tasks_start))
    self.tasks_end = list(map(int, self.tasks_end))
    epochs_info = OrderedDict()
    data = filtered_data.get_data()
    for i in range(0, len(self.tasks_start), 1):
        task_data = np.zeros((self.n_ch, 1))
        trial_name = self.trial_name[int(i / 2)]
        epochs_info[trial_name] = OrderedDict()

        start = int(self.tasks_start[i])
        end = int(
            self.tasks_end[i] - (self.tasks_end[i] - self.tasks_start[i]) % (n_times * filtered_data.info['sfreq']))
        # start = int(self.trial_duration[i])
        # end = int(self.trial_duration[i+1] - (self.trial_duration[i+1]-self.trial_duration[i]) % (n_times * self.tasks_cleaned.info['sfreq']))
        self.start_point.append(start)
        self.end_point.append(end)
        n_epochs = int((end - start) / (n_times * filtered_data.info['sfreq']))
        self.epochs_point.append(n_epochs)

        self.data_cleaned_epochs = np.asarray(np.hsplit(data[self.global_good_index, start:end], n_epochs))
        data_epochs = np.asarray(np.hsplit(data[:, start:end], n_epochs))

        self.bad_channel_epochs = self.bad_epochs_faster(data=self.data_cleaned_epochs, threshold=threshold)
        print(start)
        print(end)
        print(self.epochs_point)
        print(data_epochs)

        for j in range(n_epochs):
            ch = self.get_ch_index(self.tasks_merged.ch_names, self.global_bads, self.bad_channel_epochs[j])
            ch_bad_index = np.argwhere(ch == 1)
            ch_bad_index = ch_bad_index.reshape(ch_bad_index.shape[0]).tolist()
            ch_bad_name = [self.tasks_merged.ch_names[i] for i in range(self.n_ch) if i in ch_bad_index]
            ratio = np.sum(ch) / len(ch)
            temp = mne.io.RawArray(data_epochs[j], info=self.tasks_merged.info.copy())
            temp.info['bads'] = ch_bad_name
            if ratio < drop_epoch:
                temp = temp.interpolate_bads()
                drop = 0
                task_data = np.concatenate((task_data, temp.get_data()), axis=1)
            else:
                drop = 1
            epochs_info[trial_name][str(j)] = {'epoch_data': temp.get_data().tolist(), 'bad_channel': ch_bad_name,
                                               'interpolate_ratio': ratio, 'drop': drop}
        epochs_info[trial_name]['task_data'] = task_data[:, 1::].tolist()
    self.epochs_cleaned = epochs_info
    print(self.bad_channel_epochs)