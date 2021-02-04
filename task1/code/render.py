import numpy as np

def render(b, dir):
    lvector = np.zeros([10, 3])  # the direction of light
    for line in open(dir + '/test.txt'):
        i, ang1, ang2 = line.strip().split(",")
        i = int(i)
        ang1 = int(ang1)
        ang2 = int(ang2)
        lvector[i - 1] = (np.sin(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180), np.sin(np.pi * ang2 / 180),
                          np.cos(np.pi * ang1 / 180) * np.cos(np.pi * ang2 / 180))
    lvector = -lvector
    img = np.clip(np.einsum('ij,jkl->ikl', lvector, b), 0, 255)

    return img