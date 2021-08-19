import numpy as np
import matplotlib.pyplot as plt
import cv2
from skvideo.io import vread, vwrite
from Funcs import PL_Detect, SubPleural_Detect
import os

if __name__ == '__main__':

    W = 200
    # Declaring the path to the video we want to analyze.
    path = r'C:\Users\dekel\OneDrive\Final Project\Deep\Vid\Test_Vids'
    out_path = r'C:\Users\dekel\OneDrive\Final Project\Deep\Vid\Visual_Vids'
    Dir = os.listdir(path)
    for file in np.arange(len(Dir)):
        new_path = path + '/' + Dir[file]
        # Using the PL_Detect function, we acquire a segmented mask of the pleural line, and a boundary box of it as well.
        # As each of them might not be as accurate as we would like, we eventually plot their intersection as well.
        Seg_PL_Vid_Mask, Boundary_Box_PL_Vid_Mask = PL_Detect(path=new_path, W=W)

        # Make the masks ndarrays.
        Seg_PL_Vid_Mask = np.asarray(Seg_PL_Vid_Mask)
        Boundary_Box_PL_Vid_Mask = np.asarray(Boundary_Box_PL_Vid_Mask)

        # Using the SubPleural_Detect function, we acquire a mask of the sub-pleural area (under the pleural line).
        # We attempt to create a mask that does not contain any rib artifacts or other unnecessary information.
        SubPleural_Vid_Mask = SubPleural_Detect(Boundary_Box_PL_Vid_Mask, new_path)

        # Make the mask an ndarray.
        SubPleural_Vid_Mask = np.asarray(SubPleural_Vid_Mask)

        # Plotting the animations of:
        # 1. The boundary box on the original video.
        # 2. The segmentation on the original video.
        # 3. The intersection of the boundary box & segmentation on the original video.
        # 4. The sub-pleural area on the original video.

        Vid = vread(new_path)
        # fig, ax = plt.subplots(nrows=1, ncols=4)
        # fig.suptitle(Dir[file])
        Video = []
        Final_Boundary_Box_PL_Vid_Mask = []
        Final_SubPleural_Vid_Mask = []
        Final_Ribs_Vid = []
        Hist_List = []
        # Looping over each frame.
        for i in np.arange(len(Vid)):
            # image = Vid[i, :, :, :]
            im = Vid[i, :, :, :]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, (W, W), interpolation=cv2.INTER_CUBIC)
            histogram, bin_edges = np.histogram(im, bins=256, range=(0, 255))
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
            BB_cm = plt.get_cmap('gist_heat')
            SP_cm = plt.get_cmap('ocean')
            vid_cm = plt.get_cmap('gray')
            ribs_cm = plt.get_cmap('Wistia')
            BB_im = (BB_cm(Boundary_Box_PL_Vid_Mask[i] * im)[:, :, :3] * 255).astype(np.uint8)
            SP_im = (SP_cm(SubPleural_Vid_Mask[i] * im)[:, :, :3] * 255).astype(np.uint8)
            inds = np.argmax(Boundary_Box_PL_Vid_Mask[i], axis=0)
            ribs = im.copy()
            for k in np.arange(3):
                temp = SP_im[:, :, k]
                temp[SubPleural_Vid_Mask[i] == 0] = 0
                SP_im[:, :, k] = temp
            for k in np.arange(3):
                temp = BB_im[:, :, k]
                temp[Boundary_Box_PL_Vid_Mask[i] == 0] = 0
                BB_im[:, :, k] = temp
            im = (vid_cm(im)[:, :, :3] * 255).astype(np.uint8)
            for k in np.arange(3):
                temp = im[:, :, k]
                temp[Boundary_Box_PL_Vid_Mask[i] != 0] = 0
                temp[SubPleural_Vid_Mask[i] != 0] = 0
                im[:, :, k] = temp
            ribs = im.copy()
            for k in np.arange(3):
                if k == 2:
                    ribs[:, :, k] = np.zeros_like(temp)
                else:
                    ribs[:, :, k] = np.ones_like(temp) * 255
                    for l in np.arange(len(inds)):
                        ribs[:inds[l], l, k] = 0
                    temp = ribs[:, :, k]
                    temp[Boundary_Box_PL_Vid_Mask[i] == 1] = 0
                    temp[SubPleural_Vid_Mask[i] == 1] = 0
                    ribs[:, :, k] = temp
            for k in np.arange(3):
                temp = im[:, :, k]
                temp[ribs[:, :, 1] == 255] = 0
                im[:, :, k] = temp

            Video.append(im)
            Final_Boundary_Box_PL_Vid_Mask.append(BB_im)
            Final_SubPleural_Vid_Mask.append(SP_im)
            Final_Ribs_Vid.append(ribs)
            Hist_List.append(histogram)
        Video = np.asarray(Video)
        Final_Boundary_Box_PL_Vid_Mask = np.asarray(Final_Boundary_Box_PL_Vid_Mask)
        Final_SubPleural_Vid_Mask = np.asarray(Final_SubPleural_Vid_Mask)
        Final_Ribs_Vid = np.asarray(Final_Ribs_Vid)
        Hist_List = np.asarray(Hist_List)
        Final_Hist = np.mean(Hist_List, axis=0)

        # plt.plot(bin_edges[0:-1], Final_Hist, lw=2)
        # plt.title('Mean Histogram of' + Dir[file])
        # plt.show()

        # Final_Mask_Vid = np.zeros((len(Video), W, W, 4))
        # Final_Mask_Vid[..., 0] = (Boundary_Box_PL_Vid_Mask * Video + Video) / 2
        # Final_Mask_Vid[..., 1] = (SubPleural_Vid_Mask * Video + Video) / 2
        # Final_Mask_Vid[..., 2] = Video
        Final_Mask_Vid = Final_Boundary_Box_PL_Vid_Mask[..., :3] + Final_SubPleural_Vid_Mask[..., :3] + Video[..., :3] + Final_Ribs_Vid

        # inds1 = Boundary_Box_PL_Vid_Mask = 1
        # inds2 = SubPleural_Vid_Mask = 1
        # Final_Mask_Vid[inds1, 2] = 0
        # Final_Mask_Vid[inds2, 2] = 0
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        vwrite(out_path + '/' + Dir[file], Final_Mask_Vid)
