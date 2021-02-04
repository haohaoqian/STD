import os
import numpy as np
from torch.utils.data import Dataset

class audio_class_Dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None, loader=np.load):
        super(audio_class_Dataset, self).__init__()
        label = list()
        data_path = list()
        count = 0
        for folder in os.listdir(path):
            for sample in os.listdir(os.path.join(path, folder)):
                label.append(count)
                data_path.append(os.path.join(path, folder, sample))
            count += 1

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.label = label
        self.data_path = data_path

    def __getitem__(self, index):
        label = self.label[index]
        path = self.data_path[index]
        data = self.loader(path)

        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data_path)