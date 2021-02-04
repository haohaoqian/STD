import numpy as np
from rebuild import rebuild
from render import render
from depth import depth
from evaluate import evaluate
from PIL import Image

def rendering(dir):
    #z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    #imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应

    train_lvectors = np.zeros([7,3])# the direction of light
    for line in open(dir+'/train.txt'):
        i,ang1,ang2 = line.strip().split(",")
        i = int(i)
        ang1 = int(ang1)
        ang2 = int(ang2)
        train_lvectors[i-1] = (np.sin(np.pi*ang1/180)*np.cos(np.pi*ang2/180),np.sin(np.pi*ang2/180),np.cos(np.pi*ang1/180)*np.cos(np.pi*ang2/180))
    train_lvectors = -train_lvectors

    test_lvectors = np.zeros([10,3])# the direction of light
    for line in open(dir+'/test.txt'):
        i,ang1,ang2 = line.strip().split(",")
        i = int(i)
        ang1 = int(ang1)
        ang2 = int(ang2)
        test_lvectors[i-1] = (np.sin(np.pi*ang1/180)*np.cos(np.pi*ang2/180),np.sin(np.pi*ang2/180),np.cos(np.pi*ang1/180)*np.cos(np.pi*ang2/180))
    test_lvectors = -test_lvectors

    train_images = np.zeros([7, 168, 168])
    for num in range(7):
        image = Image.open(dir+'/train/'+str(num+1)+'.bmp')
        train_images[num] = np.asarray(image)

    n_s=3
    alpha,beta,s,X,Y,Z,vector = rebuild(train_images,train_lvectors,n_s)
    evaluate(alpha,beta,s,X,Y,Z,n_s,train_lvectors,train_images)

    imgs = render(alpha,beta,s,X,Y,Z,n_s,test_lvectors)
    z = depth(vector)

    return z, imgs