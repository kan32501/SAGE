import numpy as np
from utils.util import *
from utils.image_operations import get_mask_bounding_box
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment

def cartesian_to_hough(cart_lines):
    """
    Convert two endpoints into an infinite polar line form with the Hough Transform

    Args
        cart_lines (np.array) : lines defined by two endpoints. shape (n_lines, 2, 2)

    Returns
        hough (np.array) : shape (n_lines, 1, 2). lines defined by
                                - (r) the shortest distance to the line from the origin
                                - (theta) and the angle from the x axis
                            
    """
    # get the number of lines
    n_lines = cart_lines.shape[0]

    # initialize list of polar form lines
    hough = np.zeros(shape=(n_lines, 1, 2))

    # get the polar form for each line
    for line in range(n_lines):
        # get line
        x_a, x_b = cart_lines[line, 0, 0], cart_lines[line, 1, 0]
        y_a, y_b = cart_lines[line, 0, 1], cart_lines[line, 1, 1]

        # get m
        m = (y_b - y_a) / (x_b - x_a) 
        # if the line is vertical
        if x_b - x_a == 0.0:
            # positive or negative infinity
            m = np.inf if y_b - y_a > 0 else -1 * np.inf

        # get b (if the line is not vertical)
        b = y_a - m * x_a 

        # convert to r, theta
        theta = np.arctan(-1.0 / m)
        rho = b * np.sin(theta)

        # handle the inf gradient cases
        if m == np.inf:
            theta = np.pi / 2 # positive inf gradient
            rho = x_a
        if m == -1 * np.inf:
            theta = -1 * np.pi / 2 # negative inf gradient
            rho = x_a

        # normalize representation so reversed endpoints give same (rho, theta)
        if rho < 0:
            rho = -rho
            theta += np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
    
        # book keep the polar line parameters
        hough[line, 0, 0] = rho
        hough[line, 0, 1] = theta

    return hough

def get_midpoints(cart_lines):
    """
    Return the midpoint of a line defined by two endpoints

    Args
        cart_lines (np.array) : lines defined by two endpoints. shape: (n_lines, 2, 2)

    Returns
        midpoints (np.array)
    """
    # get the number of lines
    n_lines = cart_lines.shape[0]

    # initialize the list of n midpoints
    midpoints = np.zeros(shape=(n_lines, 1, 2))

    # get the average x and y coords
    midpoints[:, 0, 0] = np.average(cart_lines[:, :, 0], axis=1) # x
    midpoints[:, 0, 1] = np.average(cart_lines[:, :, 1], axis=1) # y

    return midpoints

def normalize_lines_by_mask(mask, lines):
    """
    Normalize line endpoints with respect to the bounding box of a mask.

    The normalized coordinates has the origin at the center of the bounding box.

    Args
        mask (PIL.Image) : PIL Image binary mask, type "1"
        lines (np.array) : size (n_lines, 2, 2)

    Returns
        lines_normalized (np.array) : size (n_lines, 2, 2) normalized [-1,1] w.r.t bounding box
    """
    # mask image to array
    mask_np = np.expand_dims(np.asarray(mask), axis=2)

    # get bounding box
    min_x, min_y, max_x, max_y = get_mask_bounding_box(mask_np)

    # get ranges and origin
    mask_width, mask_height = max_x - min_x, max_y - min_y
    origin_x, origin_y = (max_x + min_x) / 2, (max_y + min_y) / 2

    # normalization function
    def normalize(point):
        # assume point is a (2,) numpy array
        norm_x = (point[0] - origin_x) / (mask_width / 2)
        norm_y = (origin_y - point[1]) / (mask_height / 2) # the y starts at the top and goes down. so reverse.

        # START HERE: create an array with the new coordinates
        normalized_point = np.asarray([norm_x, norm_y]) # (2,) array

        return normalized_point

    # intialize output
    lines_normalized = np.zeros_like(lines)

    # normalize each coordinate
    n_lines = lines.shape[0]
    for line in range(n_lines):
        # normalize both endpoints
        norm_start_pt = normalize(lines[line, 0, :])
        norm_end_pt = normalize(lines[line, 1, :])

        # bookkeep
        lines_normalized[line, 0, :] = norm_start_pt
        lines_normalized[line, 1, :] = norm_end_pt

    return lines_normalized

def match_lines_optim(norm_midpoints0, norm_midpointsN):
    """
    Match the lines based on lowest L2 DISTANCE of midpoints (x, y), normalized with respect to the mask bounding box

    Minimize the summed L2 distance and produce an optimal 1:1 mapping

    Args
        norm_midpoints0 (np.array) : size (n_points, 1, 2) midpoints for frame0, normalized by mask bounding box
        norm_midpointsN (np.array) : size (n_points, 1, 2) midpoints for frameN, normalized by mask bounding box

    Returns
        matched_indices0 (np.array) : size (min_lines,) indices of the matched lines
        matched_indicesN (np.array) : size (min_lines,) indices of the matched lines
    """
    # get number of lines from both
    n_lines0 = norm_midpoints0.shape[0]
    n_linesN = norm_midpointsN.shape[0]

    # the set of lines with less lines chooses
    min_lines = min(n_lines0, n_linesN)

    # initialize matched lines list & all other parameters
    matched_indicesA = np.zeros(shape=(min_lines,), dtype=np.uint8)
    matched_indicesB = np.zeros(shape=(min_lines,), dtype=np.uint8)

    # set the features we are matching on
    midpointsA = norm_midpoints0 if n_lines0 < n_linesN else norm_midpointsN
    midpointsB = norm_midpointsN if n_lines0 < n_linesN else norm_midpoints0

    def l2_dist_matrix(midpointsA, midpointsB):
        """
        Compute L2 distance matrix between two sets of midpoints.
        
        Args
            midpoints0 (np.ndarray): size (n_lines, 1, 2), first set of points.
            midpointsN (np.ndarray): size (n_lines, 1, 2), second set of points.

        Returns
            np.ndarray: Shape (n_lines, n_lines), L2 distance matrix.
        """
        # remove singleton dimension
        A = midpointsA
        B = np.resize(midpointsB, (1, min_lines, 2))

        # compute the difference in each axis and find the magnitude
        diff = B - A # broadcasting creates a square matrix (min_lines, min_lines, 2)
        l2_dist_matrix = np.square(np.linalg.norm(diff, axis=2))

        return l2_dist_matrix

    # get the cost matrix
    l2_dist_matrix = l2_dist_matrix(midpointsA, midpointsB)

    # get the minimum cost assignment
    matched_indicesA, matched_indicesB = linear_sum_assignment(l2_dist_matrix)
        
    # return the matching
    if n_lines0 < n_linesN:
        return matched_indicesA, matched_indicesB
    else:
        return matched_indicesB, matched_indicesA

def interp_lines_linear(matched_lines0, matched_linesN, num_frames=13):
    """
    Generate the frame-wise pose/edge conditions that will be passed to ControlNext

    Args
        matched_lines0 (np.array) : size (n_lines, 2, 2), lines in frame0, matched by index
        matched_linesN (np.array) : size (n_lines, 2, 2), lines in frameN, matched by index
        num_frames (int) : # of interpolated frames to generate

    Returns
        conditions_images (list) : list of framewise pose/edge conditions as numpy images
    """
    # linearly interpolate between the two matched line segments, in line space
    interped_lines = []
    # from 0 to the number of inbetween frames
    for i in range(num_frames):
        # lambda weight 
        frac = i / (num_frames - 1)
        # interped is a numpy array, size (n_lines, 2, 2) for each endpoint
        interped = interpolate_matches_linear(matched_lines0, matched_linesN, frac) # return the interpolated lines
        interped_lines.append(interped)

    return interped_lines

def interp_lines_spline(linesA, linesB, maskA, maskB, flowA, flowB, num_frames, viz=False):
    """
    Use a spline curve to interpolate the positions of two sets of lines

    Args
        linesA (np.array) : size (n_lines, 2, 2), lines in start frame, matched by index
        linesB (np.array) : size (n_lines, 2, 2), lines in end frame, matched by index
        maskA (PIL.Image) : PIL Image binary mask for start frame, type "1"
        maskB (PIL.Image) : PIL Image binary mask for end frame, type "1"
        flowA (np.array) : size (H, W, 2), optical flow for start frame
        flowB (np.array) : size (H, W, 2), optical flow for end frame
        num_frames (int) : # of interpolated frames to generate

    Returns
        interped_lines (list) : list of framewise pose/edge conditions as numpy arrays, size (n_lines, 2, 2)
    """
    # convert masks to numpy
    maskA = np.round(np.asarray(maskA)).astype(np.uint8)
    maskB = np.round(np.asarray(maskB)).astype(np.uint8)

    centers1 = linesA.mean(axis=1)
    centers2 = linesB.mean(axis=1)
    length1 = np.linalg.norm(linesA[:, 0] - linesA[:, 1], axis=1)
    length2 = np.linalg.norm(linesB[:, 0] - linesB[:, 1], axis=1)
    inter_length = length1 + (length2 - length1) * np.linspace(0, 1, num_frames)[:, None]

    # Get bbox through masks
    points1 = np.array(np.nonzero(maskA)).T[:, ::-1]  # (N, 2) (x, y)
    points2 = np.array(np.nonzero(maskB)).T[:, ::-1]
    minx1, miny1 = points1.min(axis=0)
    maxx1, maxy1 = points1.max(axis=0)
    minx2, miny2 = points2.min(axis=0)
    maxx2, maxy2 = points2.max(axis=0)
    bbox_center1 = np.array([(minx1 + maxx1) / 2, (miny1 + maxy1) / 2])
    bbox_center2 = np.array([(minx2 + maxx2) / 2, (miny2 + maxy2) / 2])

    # Gather pixel flow according to the mask
    # flow_norm = flows / (np.linalg.norm(flows, axis=-1, keepdims=True) + 1e-6)
    flows1 = flowA[np.where(maskA)[0], np.where(maskA)[1]]  # (N, 2)
    flows2 = flowB[np.where(maskB)[0], np.where(maskB)[1]]

    # normalize
    flows1 = flows1 / (np.linalg.norm(flows1, axis=-1, keepdims=True) + 1e-6) 
    flows2 = flows2 / (np.linalg.norm(flows2, axis=-1, keepdims=True) + 1e-6)

    avg_dir1 = flows1.reshape(-1, 2).mean(axis=0)
    avg_dir2 = flows2.reshape(-1, 2).mean(axis=0)

    # Visualize the bbox and direction
    if viz:
        arrow_scale = 10
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        ax[0].set_xlim(0, 576)
        ax[1].set_xlim(0, 576)
        ax[2].set_xlim(0, 576)
        ax[0].set_ylim(0, 1024)
        ax[1].set_ylim(0, 1024)
        ax[2].set_ylim(0, 1024)
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[2].invert_yaxis()
        lc1 = mc.LineCollection(linesA, linewidths=2)
        ax[0].add_collection(lc1)
        lc2 = mc.LineCollection(linesB, linewidths=2)
        ax[1].add_collection(lc2)
        lc1 = mc.LineCollection(linesA, linewidths=2, colors='blue', alpha=0.3)
        lc2 = mc.LineCollection(linesB, linewidths=2, colors='red', alpha=0.3)
        ax[2].add_collection(lc1)
        ax[2].add_collection(lc2)

        ax[0].add_patch(plt.Rectangle((minx1, miny1), maxx1 - minx1, maxy1 - miny1, fill=False, edgecolor='green', linewidth=2))
        ax[1].add_patch(plt.Rectangle((minx2, miny2), maxx2 - minx2, maxy2 - miny2, fill=False, edgecolor='green', linewidth=2))
        ax[2].add_patch(plt.Rectangle((minx1, miny1), maxx1 - minx1, maxy1 - miny1, fill=False, edgecolor='green', linewidth=2))
        ax[2].add_patch(plt.Rectangle((minx2, miny2), maxx2 - minx2, maxy2 - miny2, fill=False, edgecolor='green', linewidth=2))
        ax[0].arrow(bbox_center1[0], bbox_center1[1], avg_dir1[0] * 50, avg_dir1[1] * 50, color='red', width=2)
        ax[1].arrow(bbox_center2[0], bbox_center2[1], avg_dir2[0] * 50, avg_dir2[1] * 50, color='red', width=2)
        ax[2].arrow(bbox_center1[0], bbox_center1[1], avg_dir1[0] * 50, avg_dir1[1] * 50, color='red', width=2)
        ax[2].arrow(bbox_center2[0], bbox_center2[1], avg_dir2[0] * 50, avg_dir2[1] * 50, color='red', width=2)
        # plt.imshow(maskA)

    point_seq = []
    n_supporting = 12
    for i in range(n_supporting):
        point_seq.append(bbox_center1 + avg_dir1 * i * 10)
    for i in range(n_supporting - 1, -1, -1):
        point_seq.append(bbox_center2 - avg_dir2 * i * 10)
    point_seq = np.array(point_seq)

    # Fit
    tck, u = splprep(point_seq.T, s=0)
    u_fine = np.linspace(0, 1, num_frames)
    x_fine, y_fine = splev(u_fine, tck)
    inter_mid_points = np.array([x_fine, y_fine]).T

    if viz:
        ax[2].plot(x_fine, y_fine, color='red')

    # Interpolated bbox length
    bbox_length1 = np.linalg.norm(np.array([maxx1 - minx1, maxy1 - miny1]))
    bbox_length2 = np.linalg.norm(np.array([maxx2 - minx2, maxy2 - miny2]))
    interpolated_bbox_length = bbox_length1 + (bbox_length2 - bbox_length1) * np.linspace(0, 1, num_frames)

    # Now start to interpolate each line inside the bbox
    # First get the angle
    dir1 = linesA[:, 1] - linesA[:, 0]
    dir1 = dir1 / np.linalg.norm(dir1, axis=1, keepdims=True)
    dir1[dir1[..., 0] < 0] = -dir1[dir1[..., 0] < 0]
    dir2 = linesB[:, 1] - linesB[:, 0]
    dir2 = dir2 / np.linalg.norm(dir2, axis=1, keepdims=True)
    dir2[dir2[..., 0] < 0] = -dir2[dir2[..., 0] < 0]

    # make the lines rotate minimally - angle invariance for endpoint ordering
    dot_product = np.sum(dir1 * dir2, axis=1)
    dir2[dot_product < 0] = -dir2[dot_product < 0]

    # compute the standardized angle
    angle1 = np.arctan2(dir1[:, 1], dir1[:, 0])
    angle2 = np.arctan2(dir2[:, 1], dir2[:, 0])
    interpolated_angle = angle1 + (angle2 - angle1) * np.linspace(0, 1, num_frames)[:, None]

    # Then interpolate the center points with respect to center of bbox and bbox length
    centers1_shifted = centers1 - bbox_center1
    centers2_shifted = centers2 - bbox_center2
    centers1_scaled = centers1_shifted / bbox_length1
    centers2_scaled = centers2_shifted / bbox_length2
    centers_inter_scaled = centers1_scaled[None] + (centers2_scaled - centers1_scaled) * np.linspace(0, 1, num_frames)[:, None, None]
    centers_inter = centers_inter_scaled * interpolated_bbox_length[:, None, None] + inter_mid_points[:, None, :]

    # Rebuild all endpoints
    out_directions = np.stack((np.cos(interpolated_angle), np.sin(interpolated_angle)), axis=2)
    line_endpoints = np.stack((centers_inter - out_directions * inter_length[..., None] / 2,
                              centers_inter + out_directions * inter_length[..., None] / 2), axis=2)

    if viz:
        id_line = 14
        lc = mc.LineCollection(linesA[id_line:id_line+1], linewidths=3, colors='red', alpha=1)
        ax[0].add_collection(lc)
        lc = mc.LineCollection(linesB[id_line:id_line+1], linewidths=3, colors='red', alpha=1)
        ax[1].add_collection(lc)
        lc = mc.LineCollection(line_endpoints[:,id_line], linewidths=2, colors='red', alpha=1)
        ax[2].add_collection(lc)
        plt.show()

    # convert to desired output format: List[np.array]
    line_endpoints = [line_endpoints[i, :, :, :] for i in range(num_frames)]

    return line_endpoints

def filter_fg_lines(mask, lines, tolerance=15):
    """
    Filter lines based on whether they are foreground or background with respect to a binary mask.
    Give some tolerance to foreground lines so that the lines on the boundary are included

    Args:
        mask (PIL.Image, mode "1") : binary mask on source image
        lines (np.array) : size (n, 2, 2), lines in source image defined by endpoints

    Returns
        mask_lines (np.array) : size (x, 2, 2) lines in source image that are inside the mask
    """
    # convert mask to array of 0/1
    mask_arr = np.array(mask).astype(np.uint8)
    h, w = mask_arr.shape

    # check each line
    fg_lines = []

    for line in lines:
        # extract coordinates
        (x1, y1), (x2, y2) = line.astype(int)

        # skip if endpoints are outside image bounds
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        # return whether this pixel is within tolerance distance of a mask pixel
        def pixel_in_tolerance(x, y):
            # get square around the point with radius r=tolerance
            min_x, max_x = max(0, x - tolerance), min(w - 1, x + tolerance)
            min_y, max_y = max(0, y - tolerance), min(h - 1, y + tolerance)

            # check if any pixel in this square is in the mask
            return np.any(mask_arr[min_y:(max_y+1), min_x:(max_x+1)] == 1)

        # determine if it is a foreground or background line
        if pixel_in_tolerance(x1, y1) and pixel_in_tolerance(x2, y2):
            fg_lines.append(line)

    # return filtered lines or empty array if none valid
    fg_lines = np.array(fg_lines) if fg_lines is not None else np.empty((0, 2, 2), dtype=lines.dtype)
    
    return fg_lines