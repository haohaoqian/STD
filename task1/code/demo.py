import numpy as np
import os
from rendering import rendering
from PIL import Image
import matlab.engine

dirname = 'D:\\郝千越文件\\课程\\大三秋\\HQY大三秋\\视听信息系统导论\\大作业\\视听信息系统导论第一次大作业2020\\result\\dataset_online'
dirs = os.listdir(dirname)
eng = matlab.engine.start_matlab()
for dir in dirs:
    if dir[0] == 'P':
        z, imgs = rendering(os.path.join(dirname, dir), eng)
        np.save(os.path.join(dirname, dir, 'z.npy'), z)
        for i in range(len(imgs)):
            I = Image.fromarray(np.uint8(imgs[i]), mode='L')
            I.save(os.path.join(dirname, dir, str(i) + '.bmp'))