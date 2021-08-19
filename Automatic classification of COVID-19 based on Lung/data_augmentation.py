import numpy as np
from skvideo.io import vwrite, vread
import pandas as pd
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='data augmentation parser')
parser.add_argument('--path_videos', type=str,  help='path to videos dir')
parser.add_argument('--path_dataset', type=str,  help='path to GT Excel file')


'''
Augmentation Functions
'''


def fill(img, h, w):
    """
    Fills the image.

    :param img: Input frame
    :param h: Image height
    :param w: Image width
    :return: Final image
    """
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def horizontal_shift(img, ratio=0.0):
    """
    Shifts the frame horizontally.

    :param img: Input frame
    :param ratio: Shift ratio
    :return: Final image
    """
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = fill(img, h, w)
    return img


def vertical_shift(img, ratio=0.0):
    """
    Shifts the frame vertically.

    :param img: Input frame
    :param ratio: Shift ratio
    :return: Final image
    """
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img


def rotation(img, angle):
    """
    Rotates image.

    :param img: Input image
    :param angle: Angle at which to rotate image
    :return: Final image
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


'''
Main Functions
'''


def data_aug_0(frame):
    """
    Flip.

    :param frame: Frame
    :return: Output frame
    """
    return cv2.flip(frame, flipCode=1)


def data_aug_1(frame):
    """
    Gaussian blur.

    :param frame: Frame
    :return: Output Frame
    """
    return cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)


def data_aug_2(frame):
    """
    Rotation (+3 degrees).

    :param frame: Frame
    :return: Output frame
    """
    return rotation(frame, angle=3)


def data_aug_3(frame):
    """
    Rotation (-5 degrees).

    :param frame: Frame
    :return: Output frame
    """
    return rotation(frame, angle=-5)
def data_aug_4(frame):
    """
    Horizontal shift (0.2 ratio).

    :param frame: Frame
    :return: Output frame
    """
    return horizontal_shift(frame, ratio=0.2)


def data_aug_5(frame):
    """
    Horizontal shift (0.3 ratio).

    :param frame: Frame
    :return: Output frame
    """
    return horizontal_shift(frame, ratio=0.3)


def data_aug_6(frame):
    """
    Vertical shift (0.2 ratio).

    :param frame: Frame
    :return: Output frame
    """
    return vertical_shift(frame, ratio=0.2)


def data_aug_7(frame):
    """
    Median blur.

    :param frame: Frame
    :return: Output frame
    """
    return cv2.medianBlur(frame, ksize=3)


# Augmentation dictionary
data_aug_dict = {
    'data_aug_0': data_aug_0,
    'data_aug_1': data_aug_1,
    'data_aug_2': data_aug_2,
    'data_aug_3': data_aug_3,
    'data_aug_4': data_aug_4,
    'data_aug_5': data_aug_5,
    'data_aug_6': data_aug_6,
    'data_aug_7': data_aug_7
}


def data_aug(video_path, num_of_aug, aug_dict=data_aug_dict):
    """
    Augment all frames in the video and creates num_of_aug new videos.

    :param video_path: Path to video
    :param num_of_aug: Number of different augmentations
    :param aug_dict: Augmentation dictionary
    :return: New videos
    """
    new_videos = []
    video = vread(video_path)
    for aug_i in range(num_of_aug):  # run over augmentations
        str_data_aug = 'data_aug_' + str(aug_i)
        new_video = []
        # run over frames
        for frame_idx in range(video.shape[0]):  #  frames
            frame = video[frame_idx, :, :, :]
            aug_frame = aug_dict[str(str_data_aug)](frame)
            # # for debug:
            # plt.figure()
            # plt.imshow(aug_frame)
            # plt.show()
            # #
            new_video.append(aug_frame)

        new_video = np.stack(new_video)
        new_videos.append(new_video)
    return new_videos



if __name__ == '__main__':
    # BEFORE RUNNING: make sure to copy the all_masks_filtered excel and folder, and call them with the ext "augmented"
    args = parser.parse_args()
    path_videos = args.path_videos
    path_dataset = args.path_dataset
    # path_videos = r'C:\Users\dekel\OneDrive\Final Project\Mor_Deep\All_Masks_Filtered_Augmented'
    # path_dataset = r'C:\Users\dekel\OneDrive\Final Project\Mor_Deep\Manual_Tags_Filtered_Augmented.xls'
    df = pd.read_excel(path_dataset, dtype=str)
    count = int(df.at[df.shape[0] - 1, 'video_name']) + 1
    for idx in tqdm(np.arange(df.shape[0] - 1)):
        # 8 different augmentations for severity 0 or pleural line regular
        # 4 different augmentations if there is consolidation
        if df.at[idx, 'covid_severity_grade'] == str(0) or \
                df.at[idx, 'pleural_line_regular'] == str(1) or df.at[idx, 'consolidation'] == str(1):
            if df.at[idx, 'consolidation'] == str(1):
                num_of_aug = 4
            else:
                num_of_aug = 8
            name = df.at[idx, 'video_name']
            path_vid = path_videos + '/' + name + '.mp4'
            new_videos = data_aug(path_vid, num_of_aug)
            for i in range(num_of_aug):
                new_path_vid = path_videos + '/' + str(count).zfill(4) + '.mp4'
                # Save augmented video
                vwrite(new_path_vid, new_videos[i])
                # Excel update
                new_row = df.iloc[idx]
                new_row['video_name'] = str(count).zfill(4)
                df = df.append(new_row, ignore_index=True)
                count += 1
    # Save final updated Excel file
    df.to_excel('Manual_Tags_Filtered_Augmented.xls')

    print('done')
