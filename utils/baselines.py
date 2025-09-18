import numpy as np
from PIL import Image
from utils.image_operations import crop_and_resize

"""
Return num_frames + 2 frames that gradually cross dissolve between two input frames using
a linear weightage parameter, alpha.

Args
    image0 (PIL.Image) : start image as a PIL Image object
    imageN (PIL.Image) : end image as a PIL Image object
    num_frames (int) : the number of inbetween frames
    width (int) : width of the frames
    height (int) : height of the frames

Returns
    x_dissolve_frames (List[PIL.Image]) : 

"""
def cross_dissolve(image0, imageN, num_frames, width, height):
    # convert images to numpy arrays
    image0_np = np.asarray(image0.resize((width, height)))
    imageN_np = np.asarray(imageN.resize((width, height)))

    # initialize cross dissolved frames
    x_dissolved_frames = []

    # cross dissolve with alpha
    interval = 1.0 / (num_frames + 1)
    for i in range(num_frames + 1):
        # alpha, the weightage parameter
        alpha = 1.0 - i * interval

        # create the current image by blending
        transition_frame = image0_np * alpha + imageN_np * (1 - alpha)

        # crop and resize to match the input dimensions
        transition_frame = Image.fromarray(transition_frame.astype(np.uint8))

        # append the frame to the list of frames
        x_dissolved_frames.append(transition_frame)

    return x_dissolved_frames