# Lung Ultrasound Segmentation

![MIIC flow](./help_png/MIIC%20flow.png)

##### Description:
The algorithm takes an mp4 file of a LUS video scan [NxWxHx1] and segments two elements from it, returning them in a new
video with two respective channels [Nx200x200x2]:
- **Pleural-Line bounding box**: Creates a bounding box around the Pleural-Line and returns the mask that bounding box creates as channel #1.
- **Sub-Pleural Area segmentation**: Segments the Sub-Pleural Area from the image, and returns the mask that the segmentation creates as channel #2.


**Input** : MP4 LUS videos (grayscale)

**Output** : MP4 segmented LUS videos, channel #1 = Pleural-Line bounding box, channel #2 = Sub-Pleural Area segmentation [NxWxHx1 --> Nx200x200x2].

-----------------------------

##### Preprocessing Steps:
Please follow the next steps:
0. **Environment setup**: Set up the environment [IPA.yml](./IPA.yml)
1. **DICOM to MP4**: Export DICOM videos to MP4 files.
2. **Sort by patient**: Sort MP4 files into a folder with sub-folders of the patients. For example, a path to a video would be: .../All_MP4_Videos/A264/0005.mp4

##### Steps:
3. **Cut config**: Enter the path to the folder that contains the sub-folders in the init_path variable in the [Cut_LUS.py](./Cut_LUS.py) file, and declare an output folder.
4. **Cut run**: Run the [Cut_LUS.py](./Cut_LUS.py) file (after setting the configuration to Cut_LUS), in order to crop unecessary information from the former DICOM files.
5. **Main config**: Enter the path to the Cut_LUS output folder in the init_path variable in the [PL_Seg_and_BB.py](./PL_Seg_and_BB.py) file, and declare an output folder.
6. **Main run**: Run the [PL_Seg_and_BB.py](./PL_Seg_and_BB.py) file with the configuration set on PL_Seg_and_BB.
7. Wait until the process finishes, and enjoy a folder of segmented videos.

------------------------------

##### Notes:
1. Depending on the export method and DICOM reader used, the exported videos could already be sorted, so that would take care about step 2.
2. The code includes resizing the videos to 200x200, but 300x300 works just as well, and possibly sizes even smaller than 200x200 could work.
3. If the video file has large black bezels on both sides (left/right or up/down), the algorithm will probably not work, so it's best to get rid of those.
