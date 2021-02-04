import numpy as np
import os
import json

dir1 = 'D:\\郝千越文件\\课程\\大三秋\\HQY大三秋\\视听信息系统导论\\大作业\\第三次大作业\\dataset\\train'
thresholds = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
#0:top,1:bottom,2:left,3:right
label_list = [1, 2, 0, 3]

with open('D:\郝千越文件\课程\大三秋\HQY大三秋\视听信息系统导论\大作业\第三次大作业\dataset\\train.json') as f:
    gt = json.load(f)

for threshold in thresholds:
    print(threshold)
    count = 0
    correct = 0
    audio_labels = list()
    video_labels = list()

    train_class = os.listdir(dir1)
    for folder in train_class:
        for sample in os.listdir(os.path.join(dir1, folder)):
            data = np.load(os.path.join(dir1, folder, sample, 'audio_data.pkl'), allow_pickle=True)['audio']
            label = [0] * 4
            for i in range(4):
                if np.max(data[:,i]) > threshold:
                    label[label_list[i]] = 1
            count += 4
            gt_temp = gt[folder + '/' + sample]
            audio_labels.append(label)
            video_labels.append(gt_temp)
            for i in range(4):
                if label[i] == gt_temp[i]:
                    correct += 1
    print(correct / count)

    count = 0
    match = 0
    for i in range(len(audio_labels)):
        for j in range(len(audio_labels)):
            if i!=j:
                count += 4
                for t in range(4):
                    if audio_labels[i][t] == video_labels[j][t]:
                        match += 1
    print(1 - match / count)