import json
from tqdm import tqdm
from utils import *
from alexnet import AlexNet


def classify(net, folder_name, resize=(224, 224)):
    transform = []
    if resize:
        transform.append(torchvision.transforms.Resize(resize))
    transform.append(torchvision.transforms.ToTensor())
    transform.append(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # 归一化
    transform = torchvision.transforms.Compose(transform)
    results = []
    img_dir = folder_name + '/rgb/'
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(img_dir)))
    for img_name in img_names:
        image = Image.open(img_dir + img_name)
        image = transform(image)
        results.append(net.predict(torch.unsqueeze(image, dim=0)))
    results = torch.cat(results, dim=0)
    return torch.mean(results, dim=0).cpu().numpy()


def dump_test(file_root, save_name):
    json_data = {}
    # root = './dataset/task2/test/'
    # save_name = './dataset/task2.json'
    root = file_root
    for i in tqdm(range(10)):
        sub_root = root + str(i) + '/'
        folders = list(filter(lambda x: not x.endswith(".pkl"), os.listdir(sub_root)))
        for folder in folders:
            folder_path = sub_root + folder
            images, is_moved = video_loader(folder_path)
            json_data[folder_path] = collide_detection(images, is_moved)
    with open(save_name, "w") as f:
        json.dump(json_data, f)


def dump_train(file_root, save_name, blocks=True):
    json_data = {}
    # root = './dataset/train/'
    # save_name = './dataset/train.json'
    root = file_root
    for sub_root in os.listdir(root):
        print('\n collecting %s' % sub_root)
        sub_root = root + sub_root + '/'
        folders = list(filter(lambda x: not x.endswith(".pkl"), os.listdir(sub_root)))
        for folder in tqdm(folders):
            folder_path = sub_root + folder
            images, is_moved = video_loader(folder_path)
            if blocks:
                json_data[folder_path] = collide_detection_blocks(images, is_moved)
            else:
                json_data[folder_path] = collide_detection(images, is_moved)
    with open(save_name, "w") as f:
        json.dump(json_data, f)


def dump_file():  # 用来标记各视频撞击位置的函数，分类用不到
    dump_train('./dataset/train/', './dataset/train_blocks_0.2.json')
    dump_test('./dataset/task2/test/', "./dataset/task2_blocks_0.2.json")
    dump_test('./dataset/task3/test/', "./dataset/task3_blocks_0.2.json")


def get_video_feature(net, folder_name, resize=(224, 224)):
    """
    :param folder_name: 从当前路径访问到‘video_0000’文件夹的路径
    :param resize:默认为(224,224)
    :return: 14维特征向量，前10维为分类标签
     ['061_foam_brick', 'green_basketball', 'salt_cylinder', 'shiny_toy_gun', 'stanley_screwdriver',
     'strawberry', 'toothpaste_box', 'toy_elephant', 'whiteboard_spray', 'yellow_block']
     后4维维撞击位置[上，下，左。右]
    """
    class_feature = classify(net, folder_name, resize)
    images, is_moved = video_loader(folder_name)
    move_feature = collide_detection_blocks(images, is_moved)
    #feature = np.concatenate([class_feature, move_feature])
    return class_feature, move_feature


#if __name__ == '__main__':
    #net = AlexNet()
    #net.load_state_dict(torch.load('./alexnet.pt'))
    # idx_to_class = ['061_foam_brick', 'green_basketball', 'salt_cylinder', 'shiny_toy_gun', 'stanley_screwdriver',
    #                 'strawberry', 'toothpaste_box', 'toy_elephant', 'whiteboard_spray', 'yellow_block']
    # classes = classify(net, './dataset/task2/test/0/video_0006')



    #import json
    #import os

    #label = dict()
    #path='./dataset/train'
    #for folder in os.listdir(path):
        #for sample in os.listdir(os.path.join(path, folder)):
            #images, is_moved = video_loader(os.path.join(path, folder, sample))
            #move_feature = collide_detection_blocks(images, is_moved)
            #label[folder + '/' + sample] = move_feature

    #with open('./dataset/train.json', 'w') as f:
        #json.dump(label,f)