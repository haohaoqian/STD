import librosa
import numpy as np
import os
import ResNet
import torch
import torchvision.transforms as transforms
import torch.cuda
from alexnet import AlexNet
from video_process import get_video_feature
from pairing import pairing

def test_task1(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task1/test/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 1, ‘audio_0001’: 3, ...}
    class number:
        ‘061_foam_brick’: 0
        'green_basketball': 1
        'salt_cylinder': 2
        'shiny_toy_gun': 3
        'stanley_screwdriver': 4
        'strawberry': 5
        'toothpaste_box': 6
        'toy_elephant': 7
        'whiteboard_spray': 8
        'yellow_block': 9
    '''
    results = dict()

    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    audio_transforms = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_list=[9,2,3,8,7,6,5,0,4,1]

    model = ResNet.resnet18(num_classes=10)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./model/resnet18.pth'))
    model.eval()

    for sample in os.listdir(root_path):
        data = np.load(os.path.join(root_path, sample), allow_pickle=True)['audio']
        for i in range(4):
            S = librosa.resample(data[:, i], orig_sr=44100, target_sr=11000)
            S = np.abs(librosa.stft(S[5650:-5650], n_fft=510, hop_length=128))
            S = np.log10(S+0.0000001)
            S = np.clip(S, -5, 5)
            S -= np.min(S)
            S = 255 * (S / np.max(S))
            if S.shape[-1] != 256:
                S = np.pad(S, ((0, 0), (int(np.ceil((256 - S.shape[-1]) / 2)), int(np.floor((256 - S.shape[-1]) / 2)))))
            if i == 0:
                feature = np.uint8(S)[:,:, np.newaxis]
            else:
                feature = np.concatenate((np.uint8(S)[:,:, np.newaxis], feature), axis=-1)

        X = audio_transforms(feature)
        X = X.to(device)
        y_hat = torch.softmax(model(X.unsqueeze(0)), dim=-1).argmax(dim=1).cpu().item()

        results[sample]=label_list[y_hat]
    return results

def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''
    results = dict()

    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    audio_transforms = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_list=[9,2,3,8,7,6,5,0,4,1]
    audio_model = ResNet.resnet18(num_classes=10)
    audio_model = audio_model.to(device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(torch.load('./model/resnet18.pth'))
    audio_model.eval()

    audio_class_features = []
    video_class_features = []
    audio_motion_features = []
    video_motion_features = []
    Files = list(filter(lambda x: x.endswith(".pkl"), os.listdir(root_path)))
    Files.sort()
    for sample in Files:
        audio_path=root_path + '/' + sample
        data = np.load(audio_path, allow_pickle=True)['audio']
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
        X = audio_transforms(feature)
        X = X.to(device)
        class_feature_t = torch.softmax(audio_model(X.unsqueeze(0)), dim=-1).squeeze(0)
        class_feature=np.zeros(10)
        for i in range(10):
            class_feature[label_list[i]]=class_feature_t[i]
        threshold = 0.35
        label_list2 = [1, 2, 0, 3]
        label = [0] * 4
        for i in range(4):
            if np.max(data[:,i]) > threshold:
                label[label_list2[i]] = 1
        audio_class_features.append(class_feature)
        audio_motion_features.append(label)

    net = AlexNet()
    net.load_state_dict(torch.load('./model/alexnet.pt'))

    k = 0
    while os.path.exists(root_path + '/video_' + str('%04d' % k)):
        video_class_feature, video_move_feature = get_video_feature(net, root_path + '/video_' + str('%04d' % k))
        video_class_features.append(video_class_feature)
        video_motion_features.append(video_move_feature)
        k = k + 1

    indices = pairing(audio_class_features, audio_motion_features, video_class_features, video_motion_features, -100)
    j = 0
    for sample in Files:
        results[sample] = indices[j][1]
        j = j + 1
    return results

def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    results = dict()

    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    audio_transforms = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_list=[9,2,3,8,7,6,5,0,4,1]
    audio_model = ResNet.resnet18(num_classes=10)
    audio_model = audio_model.to(device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(torch.load('./model/resnet18.pth'))
    audio_model.eval()

    audio_class_features = []
    video_class_features = []
    audio_motion_features = []
    video_motion_features = []
    Files = list(filter(lambda x: x.endswith(".pkl"), os.listdir(root_path)))
    Files.sort()
    for sample in Files:
        audio_path=root_path + '/' + sample
        data = np.load(audio_path, allow_pickle=True)['audio']
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
        X = audio_transforms(feature)
        X = X.to(device)
        class_feature_t = torch.softmax(audio_model(X.unsqueeze(0)), dim=-1).squeeze(0)
        class_feature=np.zeros(10)
        for i in range(10):
            class_feature[label_list[i]]=class_feature_t[i]
        threshold = 0.35
        label_list2 = [1, 2, 0, 3]
        label = [0] * 4
        for i in range(4):
            if np.max(data[:,i]) > threshold:
                label[label_list2[i]] = 1
        audio_class_features.append(class_feature)
        audio_motion_features.append(label)

    net = AlexNet()
    net.load_state_dict(torch.load('model/alexnet.pt'))

    k = 0
    while os.path.exists(root_path + '/video_' + str('%04d' % k)):
        video_class_feature, video_move_feature = get_video_feature(net, root_path + '/video_' + str('%04d' % k))
        video_class_features.append(video_class_feature)
        video_motion_features.append(video_move_feature)
        k = k + 1

    indices = pairing(audio_class_features, audio_motion_features, video_class_features, video_motion_features, 1.5)
    for sample in Files:
        results[sample] = -1
        length, _ = indices.shape
        for m in range(length):
            results['audio_' + str('%04d' % indices[m][0]) + '.pkl'] = indices[m][1]

    return results

for i in range(10):
    results =test_task3('D:\郝千越文件\课程\大三秋\HQY大三秋\视听信息系统导论\大作业\第三次大作业\dataset\\task3\\test\\'+str(i))
    print(results)