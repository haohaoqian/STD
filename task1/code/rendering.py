import numpy as np
from rebuild import rebuild
from render import render
from depth import depth
from visualize import visualize
from evaluate import evaluate

def rendering(dir, eng):
    # z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    # imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应
    vector, b= rebuild(dir)
    #evaluate(b, dir)  # 测试集上评估
    z = depth(vector, eng)
    imgs = render(b, dir)
    #visualize(z, b, imgs)  # 显示重建图像与深度
    return z, imgs