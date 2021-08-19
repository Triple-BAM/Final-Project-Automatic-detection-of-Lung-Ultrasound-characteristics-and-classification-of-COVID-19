import numpy as np
from skvideo.io import vread, vwrite
from skvideo.utils import rgb2gray
import os

if __name__ == '__main__':
    orig_path = r'C:\Users\dekel\Desktop\MP4_Vids\DCM10'
    orig_Dir = os.listdir(orig_path)
    for folder in np.arange(len(orig_Dir)):
        in_path = orig_path + '/' + orig_Dir[folder]
        Dir = os.listdir(in_path)
        out_path = r'C:\Users\dekel\Desktop\MP4_Vids\DCM10_cut/' + orig_Dir[folder] + '_cut'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for file in np.arange(len(Dir)):
            path = in_path + '/' + Dir[file]
            vid = vread(path)
            vid = rgb2gray(vid)
            Video = vid[:, 54:354, 240:540]
            Video[:, :45, :45] = 0
            vwrite(out_path + '/' + Dir[file], Video)

    print('Done!')
