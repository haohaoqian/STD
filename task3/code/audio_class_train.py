import torch
import ResNet
import torchvision.transforms as transforms
import time
from audio_class_Dataset import audio_class_Dataset

batch_size = 32
lr = 0.001
num_epochs = 30
data_dir = './audio_processed'
data_transforms = transforms.Compose([transforms.ToTensor()])

full_dataset = audio_class_Dataset(data_dir, transform=data_transforms)

train_length = int(full_dataset.__len__()*0.9)
val_length = full_dataset.__len__() - train_length
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_length, val_length])

train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet.resnet18(num_classes=10).to(device)
model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#class_dict = {'061_foam_brick': 0, 'green_basketball': 1, 'salt_cylinder': 2, 'shiny_toy_gun': 3, 'stanley_screwdriver': 4, 'strawberry': 5, 'toothpaste_box': 6, 'toy_elephant': 7, 'whiteboard_spray': 8, 'yellow_block': 9}

loss = torch.nn.CrossEntropyLoss()
best_acc=0
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, train_num, val_acc_sum, val_num, start = 0.0, 0, 0, 0, 0, time.time()

    model.train()
    for X, y in train_data:
        X = X.to(device)
        y = y.to(device)
        y_hat = torch.softmax(model(X), dim=-1)
        l = loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        train_num += y.shape[0]

    model.eval()
    for X, y in val_data:
        X = X.to(device)
        y = y.to(device)
        y_hat = torch.softmax(model(X),dim=-1)
        val_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        val_num += y.shape[0]

    print('epoch %d, loss %.4f, train_acc %.3f, val_acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / train_num, train_acc_sum / train_num, val_acc_sum / val_num, time.time() - start))

    if val_acc_sum / val_num > best_acc:
        best_acc = val_acc_sum / val_num
        torch.save(model.state_dict(),'./model/class_resnet18_{}.pth'.format(best_acc))