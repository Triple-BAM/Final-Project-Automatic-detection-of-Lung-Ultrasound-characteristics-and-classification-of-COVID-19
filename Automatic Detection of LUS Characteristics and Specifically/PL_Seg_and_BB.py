import numpy as np
import matplotlib.pyplot as plt
import cv2
from skvideo.io import vread, vwrite
from Funcs import PL_Detect, SubPleural_Detect
import os
from tqdm import tqdm

if __name__ == '__main__':

    # Declaring the path to the video we want to analyze.
    init_path = r'C:\Users\dekel\Desktop\MP4_Vids\DCM12_cut'
    init_Dir = os.listdir(init_path)
    for folder in tqdm(np.arange(len(init_Dir))):
        in_path = init_path + '/' + init_Dir[folder]
        # Declaring the desired resolution. Recommended 200, and if more is needed, 300 is OK as well.
        W = 200
        Dir = os.listdir(in_path)
        for file in np.arange(len(Dir)):
            path = init_path + '/' + init_Dir[folder] + '/' +Dir[file]
            # Using the PL_Detect function, we acquire a segmented mask of the pleural line, and a boundary box of it as well.
            # As each of them might not be as accurate as we would like, we eventually plot their intersection as well.
            Seg_PL_Vid_Mask, Boundary_Box_PL_Vid_Mask = PL_Detect(path=path, W=W)

            # Make the masks ndarrays.
            Seg_PL_Vid_Mask = np.asarray(Seg_PL_Vid_Mask)
            Boundary_Box_PL_Vid_Mask = np.asarray(Boundary_Box_PL_Vid_Mask)

            # Using the SubPleural_Detect function, we acquire a mask of the sub-pleural area (under the pleural line).
            # We attempt to create a mask that does not contain any rib artifacts or other unnecessary information.
            SubPleural_Vid_Mask = SubPleural_Detect(Boundary_Box_PL_Vid_Mask, path)

            # Make the mask an ndarray.
            SubPleural_Vid_Mask = np.asarray(SubPleural_Vid_Mask)

            # Plotting the animations of:
            # 1. The boundary box on the original video.
            # 2. The segmentation on the original video.
            # 3. The intersection of the boundary box & segmentation on the original video.
            # 4. The sub-pleural area on the original video.

            Vid = vread(path)
            # fig, ax = plt.subplots(nrows=1, ncols=4)
            # fig.suptitle(Dir[file])
            Video = []


            # Looping over each frame.
            for i in np.arange(len(Vid)):
                image = Vid[i, :, :, :]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                im = cv2.resize(image, (W, W), interpolation=cv2.INTER_CUBIC)

                # ax[0].cla()
                # ax[1].cla()
                # ax[2].cla()
                # ax[3].cla()
                #
                # # Boundary box of pleural line.
                # ax[0].imshow(im, cmap=plt.cm.jet)
                # ax[0].imshow(Boundary_Box_PL_Vid_Mask[i], cmap=plt.cm.jet, alpha=0.5)
                # ax[0].set_title('Pleural Line Boundary Box')
                #
                # # Segmentation of pleural line.
                # ax[1].imshow(im, cmap=plt.cm.jet)
                # ax[1].imshow(Seg_PL_Vid_Mask[i], cmap=plt.cm.jet, alpha=0.5)
                # ax[1].set_title('Pleural Line Segmentation')
                #
                # # Intersection of boundary box and segmentation.
                # ax[2].imshow(im, cmap=plt.cm.jet)
                # ax[2].imshow(Boundary_Box_PL_Vid_Mask[i] * Seg_PL_Vid_Mask[i], cmap=plt.cm.jet, alpha=0.5)
                # ax[2].set_title('Intersection of Segmentation & Boundary Box')
                #
                # # Sub-pleural area.
                # ax[3].imshow(im, cmap=plt.cm.jet)
                # ax[3].imshow(SubPleural_Vid_Mask[i], cmap=plt.cm.jet, alpha=0.5)
                # ax[3].set_title('Sub-pleural area Mask')
                #
                # # The approximate original frame rate.
                # plt.pause(1/32)

                Video.append(im)

            Video = np.asarray(Video)
            Final_Mask_Vid = np.zeros((len(Video), W, W, 3))
            Final_Mask_Vid[..., 0] = Boundary_Box_PL_Vid_Mask * Video
            Final_Mask_Vid[..., 1] = SubPleural_Vid_Mask * Video
            out_path = r'C:\Users\dekel\Desktop\MP4_Vids\DCM12_masks/' + init_Dir[folder].rstrip('cut') + 'masks'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            vwrite(out_path + '/' + Dir[file], Final_Mask_Vid)
