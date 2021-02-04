import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt

dir1 = './dataset/train'
dir2 = './audio_processed'
train_class = os.listdir(dir1)

for folder in train_class:
    os.mkdir(os.path.join(dir2, folder))
    count = 0
    plt.figure()
    for sample in os.listdir(os.path.join(dir1, folder)):
        data = np.load(os.path.join(dir1, folder, sample, 'audio_data.pkl'), allow_pickle=True)['audio']
        for i in range(4):
            S = librosa.resample(data[:, i], orig_sr=44100, target_sr=11000)
            S = np.abs(librosa.stft(S[5650:-5650], n_fft=510, hop_length=128))
            S = np.log10(S+0.0000001)
            S = np.clip(S, -5, 5)
            S -= np.min(S)
            S = 255 * (S / np.max(S))
            if S.shape[-1] < 256:
                S = np.pad(S, ((0, 0), (int(np.ceil((256 - S.shape[-1]) / 2)), int(np.floor((256 - S.shape[-1]) / 2)))))
            if S.shape[-1] > 256:
                S = S[:, int(np.ceil((S.shape[-1] - 256) / 2)): - int(np.floor((S.shape[-1] - 256) / 2))]
            if i == 0:
                feature = np.uint8(S)[:,:, np.newaxis]
            else:
                feature = np.concatenate((np.uint8(S)[:,:, np.newaxis], feature), axis=-1)
        np.save(os.path.join(dir2, folder, sample), feature)
        plt.subplot(2, 5, count + 1)
        plt.imshow(feature[:,:, 0],cmap='gray')
        count = count + 1
        if count == 10:
            plt.show()
            break