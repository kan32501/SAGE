import os
import cv2
from PIL import Image, ImageOps
import numpy as np
import seaborn as sns
import subprocess

def crop_and_resize(image, size=(1024, 576)):
    """
    Crops and resizes an image to target size

    Args
        image (PIL.Image.Image): input image
        size (tuple, optional): target size as (width, height)

    Returns
        PIL.Image.Image: The cropped and resized image.
    """
    target_width, target_height = size
    original_width, original_height = image.size

    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_width = int(original_height * target_ratio)
        left = (original_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = original_height
    else:
        new_height = int(original_width / target_ratio)
        top = (original_height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = original_width

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(size)

    return resized_image

def get_input_frames_by_index(video0_frames_dir, videoN_frames_dir, frame0_mask_path, frameN_mask_path, width, height, inputs_dir, save=True):
    """
    Return the filepaths and the corresponding images for the indices given as PIL.Image objects.
    Preprocesses the images to fit the argument width & height

    Function assumes that the directory contains PNGs and PNGs only.

    Args
        video0_frames_dir (string) : the directory that has all the start video frames expanded
        videoN_frames_dir (string) : the directory that has all the end video frames expanded
        width (int) : width of processed image
        height (int) : height of processed image
        inputs_dir (string) : directory to save the input frames
        save (bool) : whether to save the requested frames or not

    Returns
        inputs: dictionary containing {"image_x" : [filepath (string), image (PIL.Image)] } records
    """
    # get all the filepaths
    video0_frames_list = os.listdir(video0_frames_dir)
    videoN_frames_list = os.listdir(videoN_frames_dir)
    frame0_index = len(video0_frames_list) - 1
    frameN_index = 1
    
    # get full filepaths
    frame0_prev_path = video0_frames_dir + "/" + video0_frames_list[frame0_index - 3]
    frame0_path = video0_frames_dir + "/" + video0_frames_list[frame0_index]
    frameN_path = videoN_frames_dir + "/" + videoN_frames_list[frameN_index]
    frameN_next_path = videoN_frames_dir + "/" + videoN_frames_list[frameN_index + 3]

    # get images
    frame0_prev = Image.open(frame0_prev_path).convert('RGB')
    frame0 = Image.open(frame0_path).convert('RGB')
    frameN = Image.open(frameN_path).convert('RGB')
    frameN_next = Image.open(frameN_next_path).convert('RGB')

    frame0_mask = Image.open(frame0_mask_path).convert('RGB')
    frameN_mask = Image.open(frameN_mask_path).convert('RGB')

    # preprocess images
    frame0_prev = crop_and_resize(frame0_prev, (width, height))
    frame0 = crop_and_resize(frame0, (width, height))
    frameN = crop_and_resize(frameN, (width, height))
    frameN_next = crop_and_resize(frameN_next, (width, height))

    frame0_mask = crop_and_resize(frame0_mask, (width, height))
    frame0_mask_bin = load_mask(frame0_mask_path, width, height, as_numpy=False)
    frame0_mask_bin_inv = load_mask(frame0_mask_path, width, height, as_numpy=False, invert=True)
    frame0_masked = apply_mask(frame0, frame0_mask_bin)
    frame0_masked = crop_and_resize(frame0_masked, (width, height))
    frame0_masked_inv = apply_mask(frame0, frame0_mask_bin_inv)
    frame0_masked_inv = crop_and_resize(frame0_masked_inv, (width, height))

    frameN_mask = crop_and_resize(frameN_mask, (width, height))
    frameN_mask_bin = load_mask(frameN_mask_path, width, height, as_numpy=False)
    frameN_mask_bin_inv = load_mask(frameN_mask_path, width, height, as_numpy=False, invert=True)
    frameN_masked = apply_mask(frameN, frameN_mask_bin)
    frameN_masked = crop_and_resize(frameN_masked, (width, height))
    frameN_masked_inv = apply_mask(frameN, frameN_mask_bin_inv)
    frameN_masked_inv = crop_and_resize(frameN_masked_inv, (width, height))

    # save the PNGs
    if save:
        frame0_prev.save(inputs_dir + "/0_start_prev.png")
        frame0.save(inputs_dir + "/1_start.png")
        frameN.save(inputs_dir + "/2_end.png")
        frameN_next.save(inputs_dir + "/3_end_next.png")

        frame0_mask.save(inputs_dir + "/4_start_mask.png")
        frame0_masked.save(inputs_dir + "/5_start_masked.png")
        frame0_masked_inv.save(inputs_dir + "/6_start_masked_inv.png")

        frameN_mask.save(inputs_dir + "/7_end_mask.png")
        frameN_masked.save(inputs_dir + "/8_end_masked.png")
        frameN_masked_inv.save(inputs_dir + "/9_end_masked_inv.png")

    # return a dictionary of the filepaths & images
    inputs = dict()
    inputs["frame0_prev"] = [frame0_prev_path, frame0_prev]
    inputs["frame0"] = [frame0_path, frame0]
    inputs["frameN"] = [frameN_path, frameN]
    inputs["frameN_next"] = [frameN_next_path, frameN_next]

    return inputs

def save_out_frames(inbetween_images, 
                    out_frames_name, out_frames_dir, gif_dir,
                    bef_and_aft=True, video0_frames_dir=None, videoN_frames_dir=None,
                    width=-1, height=-1):
    """
    Saves the list of frame0s (optional) + inbetween frames + frameNs (optional) PIL images as

    - individual PNG files
    - an animated GIF.

    Option to add the before & after frames from the input clips into the output frames.

    Args
        inbetween_images (List[PIL.Image.Image]): List of images to save
        out_frames_name (string) : name of out frames, eg. "name_00.gif"
        out_frames_dir (string) : directory to save the final output PNG frames
        gif_dir (string) : directory to save the gif
        video0_frames_dir (string) : directory of the PNG frames from start video
        videoN_frames_dir (string) : directory of the PNG frames from end video
        bef_and_aft (bool): whether to save the before & after frames or not

    Returns
        gif_path (string) : filepath of gif
        out_frames (List) : list of PIL.Image() objects containing all the frames 
    """
    # create out frames directory if it doesnt exist
    if not os.path.exists(out_frames_dir): os.mkdir(out_frames_dir)
    
    # out frame index
    out_frame_index = 0

    # out frames container
    out_frames = []

    if bef_and_aft: 
        # convert all preceding frames from start video up until frame0_prev_index into PIL Images
        video0_frames_list = os.listdir(video0_frames_dir) # all filepaths in start video
        video0_frames_PIL = [Image.open(os.path.join(video0_frames_dir, video0_frame_path_i)).convert('RGB')
                        for video0_frame_path_i in video0_frames_list]
        # save in out_frames_dir
        for video0_frame in video0_frames_PIL:
            # frame filename is xx.png
            frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
            video0_frame.save(frame_path)

            # append to output frames
            out_frames.append(crop_and_resize(video0_frame, size=(width, height)))

            # increment frame count
            out_frame_index += 1

    # save each image in the inbetween list
    for inbetween_image in inbetween_images:
        # frame filename is xx.png
        frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
        inbetween_image.save(frame_path)

        # append to output frames
        out_frames.append(inbetween_image)

        # increment frame count
        out_frame_index += 1

    if bef_and_aft:
        # convert all frames from end video after frameN_next_index into PIL Images
        videoN_frames_list = os.listdir(videoN_frames_dir) # all filepaths in start video
        videoN_frames_PIL = [Image.open(os.path.join(videoN_frames_dir, videoN_frame_path_i)).convert('RGB') 
                    for videoN_frame_path_i in videoN_frames_list]
        # save in out_frames_dir
        for videoN_frame in videoN_frames_PIL:
            # frame filename is xx.png
            frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
            videoN_frame.save(frame_path)

            # append to output frames
            out_frames.append(crop_and_resize(videoN_frame, size=(width, height))) 

            # increment frame count
            out_frame_index += 1

    # make images into gif and save
    gif_path = os.path.join(gif_dir, out_frames_name + '.gif')
    duration = 60 # if not bef_and_aft else 150
    out_frames[0].save(gif_path, save_all=True, append_images=out_frames[1:], loop=0, duration=duration)

    return gif_path

def load_mask(mask_path, width, height, as_numpy=True, invert=False):
    """
    Load the mask image into a binary numpy array (or PIL Image)

    Args
        mask_path (string) : path of mask image
        width (int) : width of image
        height (int) : height of image
        as_numpy (bool) : set output type as numpy
        invert (False) : invert the mask

    Returns
        mask_img () : mask image, size (width, height, 1) as numpy or PIL
    """
    # load image as the arguement size
    mask_PIL = crop_and_resize(Image.open(mask_path), size=(width, height))
    mask_PIL = mask_PIL.convert("1")

    # invert the mask
    if invert:
        mask_PIL_L = mask_PIL.convert("L")
        mask_PIL = ImageOps.invert(mask_PIL_L).convert("1")

    # return the image
    if not as_numpy: 
        return mask_PIL
    else:
        # reduce to 1 channel
        mask_img = np.round(np.asarray(mask_PIL)).astype(np.uint8)
    
    return mask_img

def apply_mask(image, mask, save_path=None):
    """
    Use PIL.Image.composite function to mask the image

    Args
        image (PIL.Image) : source image
        mask (PIL.Image) : mask image
        save_path (string) : location to save image

    Returns
        image_masked (PIL.Image) : masked image
    """
    # create black image for the 0 pixels
    width, height = image.size
    black = Image.new(mode="RGB", size=(width, height))

    # mask image
    image_masked = Image.composite(image, black, mask)

    # save image
    if save_path:
        image_masked.save(save_path)

    return image_masked

def get_mask_bounding_box(mask):
    """
    Get the bounding box of a mask

    Args
        mask (np.array) : size (width, height, 1) binary mask

    Returns
        min_x (int) : x value with leftmost pixel of the mask
        min_y (int) : y value with the highest pixel of the mask
        max_x (int) : x value with the rightmost pixel of the mask
        max_y (int) : y value with the lowest pixel of the mask
    """
    # convert from True / False to 1/0
    mask_01 = mask.astype(np.uint8)

    # get dimensions
    height, width, _ = mask_01.shape

    # find the leftmost and rightmost occurence
    min_x = np.inf
    max_x = -np.inf
    for y in range(height):
        # get in-mask pixels in row
        row = mask_01[y, :].squeeze(axis=1) # (width,)
        in_mask_cols = np.argwhere(row == 1)

        # get first & last col with 1
        first_col = in_mask_cols[0,0] if len(in_mask_cols) != 0 else np.inf
        last_col = in_mask_cols[-1,0] if len(in_mask_cols) != 0 else -np.inf
        
        # update the leftmost occurrence
        min_x = first_col if first_col < min_x else min_x
        # update the rightmost occurence
        max_x = last_col if last_col > max_x else max_x

    # find the highest and lowest occurence
    min_y = np.inf
    max_y = -np.inf
    for x in range(width):
        # get in-mask pixels col
        col = mask_01[:, x].squeeze(axis=1) # (height,)
        in_mask_rows = np.argwhere(col == 1)

        # get first & last col with 1
        first_row = in_mask_rows[0,0] if len(in_mask_rows) != 0 else np.inf
        last_row = in_mask_rows[-1,0] if len(in_mask_rows) != 0 else -np.inf
        
        # update the leftmost occurrence
        min_y = first_row if first_row < min_y else min_y
        # update the rightmost occurence
        max_y = last_row if last_row > max_y else max_y

    return min_x, min_y, max_x, max_y

def get_mask_dimensions(mask):
    """
    Return the dimensions of a binary mask

    Args
        mask (PIL.Image) : binary mask image type "1"

    Returns
        mask_width (int) : width of mask
        mask_height (int) : height of mask
    """
    # mask image to array
    mask_np = np.expand_dims(np.asarray(mask), axis=2)

    # get bounding box
    min_x, min_y, max_x, max_y = get_mask_bounding_box(mask_np)

    # get ranges and origin
    mask_width, mask_height = max_x - min_x, max_y - min_y

    return mask_width, mask_height

def get_lines_bounding_box(lines):
    """
    return the center, the height, and width of the bounding box around the lines

    Args
        lines (np.array) : shape (n_lines, 2, 2). lines defined by two endpoints
    
    Returns
        center (np.array) : shape (2,). (x, y) coordinates of the center of the bounding box
        box_w (int) : width of bounding box
        box_h (int) : height of bounding box
    """
    if lines.size == 0:
        raise ValueError("Input lines array is empty")

    # Flatten all endpoints
    points = lines.reshape(-1, 2)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    box_w = int(max_x - min_x)
    box_h = int(max_y - min_y)

    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

    return center, box_w, box_h

def in_bounding_box(point, box_center, box_w, box_h):
    """
    Check if a point is within a bounding box

    Args
        point (np.array) : shape (2,). (x, y) coordinates of the point
        box_center (np.array) : shape (2,). (x, y) coordinates of the center of the bounding box
        box_w (int) : width of bounding box
        box_h (int) : height of bounding box

    Returns
        in_box (bool) : whether the point is in the bounding box or not
    """
    # get the box corners
    min_x = box_center[0] - box_w / 2.0
    max_x = box_center[0] + box_w / 2.0
    min_y = box_center[1] - box_h / 2.0
    max_y = box_center[1] + box_h / 2.0

    # check if point is in the bounding box
    in_box = (point[0] >= min_x) and (point[0] <= max_x) and (point[1] >= min_y) and (point[1] <= max_y)

    return in_box


def plot_lines(image, fg_lines, lw, black=True):
    """
    Plot foreground and background lines onto an image, priority on foreground lines; do not plot background lines
    if they overlap with foreground lines.

    Args:
        image (PIL.Image): The input image to paint
        fg_lines (np.array): size (n_lines, 2, 2). foreground lines that are inside the mask.
        lw (int) : width of line
    Returns:
        line_image (PIL.Image): The image with lines plotted
    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # choose color palette for lines
    n = fg_lines.shape[0]
    line_colors = sns.color_palette('husl', n_colors=n)

    # plot each line segment in foreground
    for i in range(n):
        # get endpoints
        pt1, pt2 = tuple(map(int, fg_lines[i][0])), tuple(map(int, fg_lines[i][1]))

        # plot line
        color = (line_colors[i][2] * 255, line_colors[i][1] * 255, line_colors[i][0] * 255)
        cv2.line(img_bgr, pt1, pt2, color, lw)

    # Convert back to PIL.Image
    line_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return line_image

def plot_condition_imgs(image, interped_lines_fg, lw=2, save=True, out_dir=None, black=True):
    """
    Plot the interpolated lines onto black images. Plot with priority on foreground lines; do not plot background lines
    if they overlap with foreground lines.

    Args
        image (PIL.Image) : base image 
        interped_lines_fg (List[np.array]) : list of interpolated foreground lines that are inside the masks
        lw (int) : line width on image. default is 2
        save (bool) : whether to save the images or not
        out_dir (string) : directory to save the images
        black (bool) : whether to use black or white images as background

    Returns
        conditions_images (List[PIL.Image]) : list of framewise conditions as images
    """
    # base image
    image = np.zeros_like(np.asarray(image)) if black else np.ones_like(np.asarray(image)) * 255

    # get number of frames
    num_frames = len(interped_lines_fg)

    # plot each frame condition
    condition_images = []
    for i in range(num_frames):
        # get lines
        fg_lines = interped_lines_fg[i]

        # get frame conditon
        condition_image = plot_lines(image, fg_lines, lw, black)
        condition_images.append(condition_image)

        # save imag
        if save:
            condition_image.save(os.path.join(out_dir, 'condition{:02d}.png'.format(i)))

    return condition_images

def pngs_to_mp4(input_dir, output_file, frame_file_format, framerate=24):
    """
    Convert PNG sequence to MP4 video using ffmpeg.
s
    Args
        input_dir (str) : Directory with PNG images (named sequentially like img001.png).
        output_file (str) : Path to output MP4 file.
        framerate (int) Frames per second of output video.
    """
    # ffmpeg expects sequentially numbered files
    # Example: img001.png, img002.png, ...
    pattern = os.path.join(input_dir, frame_file_format)  # adjust padding as needed

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-framerate", str(framerate),
        "-i", pattern,
        "-c:v", "mpeg4",
        "-pix_fmt", "yuv420p",
        output_file,
    ]

    subprocess.run(cmd, check=True)