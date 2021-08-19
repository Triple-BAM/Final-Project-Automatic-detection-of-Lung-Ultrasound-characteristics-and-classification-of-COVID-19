import pandas as pd
import cv2
import os
from pathlib import Path
import glob
import math
from tqdm import tqdm

# Run to split videos using a lattice splitting method.
# Outputs a new folder with split videos and a matching Excel file with appropriate features.

count_vid = 0
# Wanted number of frames per video.
frame_number = 20
path = Path(os.path.realpath(__file__)).parent
os.mkdir(Path(path, 'Masks_Final'))
to_folder = Path(path, 'Masks_Final')
from_folder = Path(path, 'All_Masks_Filtered_Augmented')
tags = pd.read_excel(Path(path, 'Manual_Tags_Filtered_Augmented.xls'))

new_tags = tags.head(0)
original_vids = glob.glob(str(Path(from_folder, '*.mp4')))
for orig_idx, original in tqdm(enumerate(original_vids)):
	cap = cv2.VideoCapture(original)
	success, image = cap.read()
	count = 0
	list_im = []
	list_im.append(image)
	while success:
		success, image = cap.read()
		list_im.append(image)
		count += 1

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	num_vid = math.floor(len(list_im)/frame_number)
	all_vid = [None] * num_vid
	for vid in range(num_vid):
		new_tags = new_tags.append(tags.iloc[orig_idx], ignore_index=True)
		new_tags.at[len(new_tags)-1, 'video_name'] = str(count_vid).zfill(4)
		all_vid[vid] = list_im[vid::num_vid][:frame_number]
		video = cv2.VideoWriter(str(Path(to_folder, str(count_vid).zfill(4) + '.mp4')), fourcc, 25, (200, 200))
		for im in all_vid[vid]:
			video.write(im)
		cv2.destroyAllWindows()
		video.release()
		del video
		count_vid += 1
new_tags.to_excel(str(Path(path, 'Manual_Tags_Final.xls')))
print('Done')
