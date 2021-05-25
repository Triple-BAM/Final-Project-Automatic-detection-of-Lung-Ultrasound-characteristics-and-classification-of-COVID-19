import numpy as np
import cv2
from skvideo.io import vread
from skimage.filters import *
from skimage.exposure import rescale_intensity
from numpy.polynomial.chebyshev import chebfit
from PIL import Image, ImageDraw


def PL_Detect(path, W):

    """
    Recieves a LUS video path and returns a boundary box + segmentation of the pleural line.
    :param path: Path to the desired video.
    :param W: Desired final size of video (will be square, that is, WxW).
    :return Seg_PL_Vid_Mask: Mask of pleural line segmentation.
    :return Boundary_Box_Vid_Mask: Mask of pleural line boundary box.
    """

    # Note: if video is rectangular with black bezels on the sides, the algorithm is prone to mistakes.

    # Loading Video.
    Vid = vread(path)

    # Declaration of both output masks.
    Seg_PL_Vid_Mask = []
    Boundary_Box_PL_Vid_Mask = []

    # Structuring element for morphological operation (Open = Erode -> Dilate) on every frame.
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))

    # Looping over each frame in the video.
    for i in np.arange(len(Vid)):

        # Frame declaration.
        image = Vid[i, :, :, :]

        # Changing frame to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resizing to desired shape (WxW).
        image = cv2.resize(image, dsize=(W, W), interpolation=cv2.INTER_CUBIC)

        # Rescaling to the range of (0,1).
        image = rescale_intensity(image, out_range=(0, 1))

        # Creating a copy of this image for further use.
        im = image.copy()

        # Addresses a problem when LUS video is very bright (mean above 0.4). We address this by thresholding out
        # certain values.
        if np.mean(im) > 0.4:
            ind = image < 0.8 * np.amax(image)
            image[ind] = 0

        # Using the crop_LUS function we crop out most of the pixels that are not part of the pleural line.
        image = crop_LUS(image=image)

        # Gaussian filtering for smoothing the image.
        image = gaussian(image=image, sigma=3)

        # Rescaling back to the range of (0,1).
        image = rescale_intensity(image, out_range=(0, 1))

        # The mask is the intersection of the cropped image and the original unfiltered image.
        Seg_PL_Mask = image * im

        # Thresholding the previous intersection.
        Seg_PL_Mask = Seg_PL_Mask > threshold_yen(image)

        # Opening with a horizontal rectangular structuring element in order to remove vertical/small noises.
        # This is the final result of the segmentation.
        Seg_PL_Mask = cv2.morphologyEx(np.float32(Seg_PL_Mask), cv2.MORPH_OPEN, struct)

        # Using the Boundary_Box_PL function, we find the boundary box of the pleural line.
        # We input the image as a boolean array of all pixels above 0.9 in the intersection of the segmentation mask
        # and the original image.
        # We declare a section width of W/20 (in the case of W=200, the boundary box will span the width of the image
        # and will be 10 pixels thick).
        Boundary_Box_PL_Mask = Boundary_Box_PL(cropped_image=(Seg_PL_Mask * im) > 0.9, line_thickness=int(W / 20))

        # Appending the masks of the current frame to the 3D video mask arrays.
        Seg_PL_Vid_Mask.append(Seg_PL_Mask)
        Boundary_Box_PL_Vid_Mask.append(Boundary_Box_PL_Mask)

    return Seg_PL_Vid_Mask, Boundary_Box_PL_Vid_Mask


def crop_LUS(image):

    """
    Recieves a LUS frame and returns a cropped mask of it that contains the pleural line.
    :param image: LUS frame after initial pre-processing.
    :return cropped_image: Cropped image.
    """

    # # # Step 1: Horizontal cutoff. All pixels above initial horizontal cutoff are zeroed.

    # Image width/height.
    W = np.shape(image)[0]

    # Thresholding image.
    Binary_im = image > threshold_yen(image)

    # Initial horizontal cutoff is done by first dividing the binary image into vertical sections, then finding the row
    # index where the sum of the section under said index and above are equal (or the difference is minimal). Later
    # a line will be fit to these indices, creating the original cutoff.

    # Declaring number of sections.
    num_sections = int(W / 15)

    # Declaring the section width
    section_width = int(W / num_sections)

    # Creating a zero vector of the wanted row indices.
    min_ind = np.zeros((W,))

    # Looping over all sections.
    for m in np.arange(num_sections):
        mini = W
        before = np.count_nonzero(Binary_im[:int(W / 2), m * section_width: (m + 1) * section_width])
        after = np.count_nonzero(Binary_im[int(W / 2):, m * section_width: (m + 1) * section_width])
        for n in np.arange(int(W / 2)):
            diff = np.count_nonzero(Binary_im[int(W / 2 - n):int(W / 2 - n + 1), m * section_width: (m + 1) * section_width])
            before = before - diff
            after = after + diff
            temp_min = np.abs(before - after)
            if temp_min < mini:
                mini = temp_min
                min_ind[m * section_width: (m + 1) * section_width] = int(W / 2) - n
            if mini == 0:
                break

    # Fitting a line to the indices, and then zeroing out all pixels above said line.
    coeff = chebfit(np.arange(W), min_ind, deg=1)
    for p in np.arange(W):
        image[:round(coeff[0] + coeff[1] * p), p] = 0

    # # # Step 2: Removing bottom third of image (99% of the time this area does not include the pleural line).
    image[int(2 * W / 3):, :] = 0

    # # # Step 3: More Horizontal cutoffs. All pixels on found horizontal lines are removed (on non-binary image).

    # We divide the image by half (vertically). Then, for each half, we find the top third of horizontal lines
    # with the highest sum of pixels, and we retain these. All other horizontal lines are zeroed.

    # Initializing a zero array that will eventually mask the current mask (further removing certain pixels).
    temp_arr = np.zeros(np.shape(image))

    # Looping over both halves.
    for i in np.arange(2):
        nz_sums = np.count_nonzero(image[:, i * int(W / 2):(i + 1) * int(W / 2)], axis=0)
        top_inds = np.argsort(nz_sums)[:int(W / 3)]
        for index in np.arange(len(top_inds)):
            temp_arr[top_inds[index], i * int(W / 2):(i + 1) * int(W / 2)] = 1

    # # # Step 4: Vertical Cutoffs. All pixels found on certain vertical lines are removed (on binary image).

    # Similar to step 2, only this time there are no sections, the lines are vertical, and the top two thirds are taken.

    # Creating the binary image.
    Binary_im = image > threshold_yen(image)

    # Finding the top two thirds vertical lines
    nz_sums = np.count_nonzero(Binary_im, axis=1)
    top_two_third_inds = np.argsort(nz_sums)[:int(2 * W / 3)]
    for index in np.arange(len(top_two_third_inds)):
        temp_arr[:, top_two_third_inds[index]] = 1

    # Multiplying the image by the array of removed pixels in order to remove them from the image.
    cropped_image = image * temp_arr

    # Rescaling to the range of (0,1).
    cropped_image = rescale_intensity(cropped_image, out_range=(0, 1))



    return cropped_image


def Boundary_Box_PL(cropped_image, line_thickness):

    """
    Recieves a cropped LUS frame and returns a boundary box that contains the pleural line.
    :param cropped_image: Binary cropped LUS frame after initial pre-processing.
    :param line_thickness: Desired thickness of boundary box (in pixels).
    :return BB_Mask: Boundary box of pleural line.
    """

    # Initializing all needed indices and values.
    max_ind = 0
    max_val = 0
    max_deg = 0
    start_ind = 0
    W = np.shape(cropped_image)[0]

    # Creating boundary boxes with different slopes (between -5 and 5 degrees with a resolution of 1 degree), and
    # checking which one of them encompasses the most of the binary cropped input image).

    # Declaring a starting upper index to reduce computational time. Final bottom index is two thirds of W.
    for i in np.arange(W):
        if np.count_nonzero(cropped_image[i, :]) != 0:
            start_ind = i
            break
    for deg in np.arange(start=-5 / 45, stop=5 / 45, step=1 / 45):
        for j in np.arange(start=start_ind, stop=int(2 * W / 3)):
            img = Image.new('L', (W, W), 0)
            ImageDraw.Draw(img).line(((0, j), (W, int(j + deg * W))), fill=1, width=line_thickness)
            mask = np.array(img)
            temp = np.count_nonzero(cropped_image * mask)
            if temp > max_val:
                max_val = temp
                max_ind = j
                max_deg = deg

    # If the cropped image was all zeros, we guess the boundary box as a horizontal (deg=0) line with a 10 pixel
    # thickness at the W/4 index.
    if max_val == 0:
        max_ind = int(W / 4)
        max_deg = 0

    # Creating the boundary box using the degree and index that led to a maximal encompassing of the cropped image.
    img = Image.new('L', (W, W), 0)
    ImageDraw.Draw(img).line(((0, max_ind), (W, int(max_ind + max_deg * W))), fill=1, width=line_thickness)
    mask = np.array(img)
    BB_Mask = mask

    return BB_Mask


def SubPleural_Detect(Boundary_Box_PL_Vid_Mask, path):

    """
    Receives a boundary box of the pleural line and returns a mask of the sub-pleural area.
    :param Boundary_Box_PL_Vid_Mask: Boundary box of the pleural line.
    :param path: Path to the desired video.
    :return SubPleural_Vid_Mask: Mask of the sub-pleural area for the video.
    """

    # Loading Video.
    Vid = vread(path)

    # Declaration of output mask.
    SubPleural_Vid_Mask = []

    # Looping over each frame in the video.
    for i in np.arange(len(Boundary_Box_PL_Vid_Mask)):

        # Declaring the size of the mask.
        W = np.shape(Boundary_Box_PL_Vid_Mask)[1]

        # Frame declaration.
        im = Vid[i, :, :, :]

        # Changing frame to grayscale
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # Resizing to desired shape (WxW).
        im = cv2.resize(im, dsize=(W, W), interpolation=cv2.INTER_CUBIC)

        # Rescaling to the range of (0,1).
        im = rescale_intensity(im, out_range=(0, 1))

        # Declaring a threshold for later use.
        thresh = threshold_triangle(im)

        # Gaussian filtering for smoothing the image.
        im = gaussian(im, sigma=3)

        # Declaring the boundary box for this particular video frame.
        BB_Mask = Boundary_Box_PL_Vid_Mask[i]

        # Declaring an array of zeros as the initial area of interest.
        Area_Of_Interest = np.zeros(np.shape(BB_Mask))

        # All pixels under the boundary box are counted initially as part of the area of interest.
        for j in np.arange(np.shape(BB_Mask)[0]):
            temp_col = BB_Mask[:, j]
            max_ind = np.amax(np.nonzero(temp_col)[0]) + 1
            Area_Of_Interest[max_ind:, j] = 1

        # Area of interest is multiplied by the image, resulting in the image under the boundary box.
        image = im * Area_Of_Interest

        # The previous image is thresholded to remove rib artifacts and other unnecessary pixels, creating the mask.
        SubPleural_Mask = image > thresh

        # Appending the mask of the current frame to the 3D video mask array.
        SubPleural_Vid_Mask.append(SubPleural_Mask)

    return SubPleural_Vid_Mask
