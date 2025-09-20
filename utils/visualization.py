import cv2
import numpy as np
import os
from PIL import Image
import seaborn as sns
from utils.image_operations import crop_and_resize, apply_mask

def plot_flow(image, points, flow, magnitude=100.0):
    """
    Plot optical flow vectors onto an image at the specified points.

    Args:
        image (PIL.Image): The input image to paint
        points (np.array): size (n, 1, 2). [x, y] origin points of the flow vectors 
        flow (np.array): size (n, 2). [u, v] flow vectors corresponding to the points.
    Returns:
        flow_image (PIL.Image): The image with flow vectors plotted.

    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # plot each flow vector as an arrow
    n = points.shape[0]
    if flow.shape[0] != n: raise ValueError("Number of points and flow vectors don't match.")
    for i in range(n):
        # get entries
        x, y = points[i, 0, 0], points[i, 0, 1]
        u, v = flow[i, 0, 0], flow[i, 0, 1]

        # endpoints
        start = (int(x), int(y))
        end = (int(x + magnitude * u), int(y + magnitude * v))

        # plot arrow
        color = (0, 230, 0)
        cv2.arrowedLine(img_bgr, start, end, color, 1, tipLength=0.3)

    # Convert back to PIL.Image
    flow_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return flow_image

def plot_lines(image, lines, color=True, number=True):
    """
    Plot lines onto an image

    Args:
        image (PIL.Image): The input image to paint
        lines (np.array): size (n, 2, 2). [[x_a, y_a] , [x_b, y_b]] line segments to plot
        color (bool) : color each line differently
        number (bool) : number each line by index
    Returns:
        line_image (PIL.Image): The image with lines plotted
    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # choose color palette for lines
    n = lines.shape[0]
    line_colors = sns.color_palette('husl', n_colors=n)

    # plot each line segment
    for i in range(n):
        # get endpoints
        pt1, pt2 = tuple(map(int, lines[i][0])), tuple(map(int, lines[i][1]))
        midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

        # plot line
        color = (line_colors[i][2] * 255, line_colors[i][1] * 255, line_colors[i][0] * 255) if color else (255, 0, 0)
        cv2.line(img_bgr, pt1, pt2, color, 2)

        # plot number
        if number:
            org = (midpoint[0] + 2, midpoint[1] + 2)
            cv2.putText(img_bgr, str(i), tuple(map(int, org)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Convert back to PIL.Image
    line_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return line_image

def plot_points(image, points, coordinates=False, color=False):
    """
    Plot points onto an image

    Args:
        image (PIL.Image): The input image to paint
        points (np.array): size (n, 1, 2). [x, y] coords to plot
        coordinates (bool): If True, display the coordinates of each point next to it.
    Returns:
        point_image (PIL.Image): The image with points plotted
    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # choose color palette for points
    n = points.shape[0]
    line_colors = sns.color_palette(n_colors=n)

    # plot each point
    for i in range(n):
        x, y = points[i, 0, 0], points[i, 0, 1]
        point = (int(x), int(y))
        cv2.circle(img_bgr, point, radius=3, color=(0, 0, 255), thickness=-1)  # red circle

        # plot coordinates slightly diagonally below the point
        if coordinates:
            # choose color for text
            color = (line_colors[i][2] * 255, line_colors[i][1] * 255, line_colors[i][0] * 255) if color else (0, 0, 255)
            
            # put text slightly shifted
            x_shift = 5
            y_shift = -5

            # plot coordinates
            cv2.putText(img_bgr, f"({x:.1f}, {y:.1f})", (int(x) + x_shift, int(y) + y_shift), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Convert back to PIL.Image
    point_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return point_image

def visualize_lines(height, width,
                    matched_lines0, matched_linesN, 
                    midpoints0, midpointsN,
                    output_dir,
                    frame0_mask=None, frameN_mask=None,
                    bg_white=True):
    """
    Visualize detected lines and midpoints on start and end image

    Args
        height (int) : height of the image
        width (int) : width of the image
        matched_lines0 (np.array) : size (n_lines, 2, 2) line segments in frame0
        matched_linesN (np.array) : size (n_lines, 2, 2) line segments in frameN
        midpoints0 (np.array) : size (n_lines, 1, 2) midpoints of matched_lines0
        midpointsN (np.array) : size (n_lines, 1, 2) midpoints of matched_linesN
        output_dir (string) : directory to save the visualizations
        frame0_mask (PIL.Image) : mask on frame0, optional
        frameN_mask (PIL.Image) : mask on frameN, optional
        bg_white (bool) : if True, use white background; else black background
    
    Returns
        None
    """
    # initialize base white images
    img0 = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255) if bg_white else Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    imgN = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255) if bg_white else Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    
    # apply masks if provided
    if frame0_mask is not None:
        img0 = apply_mask(img0, frame0_mask)
    if frameN_mask is not None:
        imgN = apply_mask(imgN, frameN_mask)

    # plot line segments, colored
    img0_lines_colored = plot_lines(img0, matched_lines0, color=True, number=False)
    imgN_lines_colored = plot_lines(imgN, matched_linesN, color=True, number=False)
    img0_lines_colored.save(os.path.join(output_dir, "frame0_lines_colored.png"))
    imgN_lines_colored.save(os.path.join(output_dir, "frameN_lines_colored.png"))

    # plot line segments, colored, with midpoints
    # if flow0_midpts is None:
    img0_lines_colored_midpts = plot_points(img0_lines_colored, midpoints0, coordinates=False, color=False)
    imgN_lines_colored_midpts = plot_points(imgN_lines_colored, midpointsN, coordinates=False, color=False)
    img0_lines_colored_midpts.save(os.path.join(output_dir, "frame0_lines_colored_midpts.png"))
    imgN_lines_colored_midpts.save(os.path.join(output_dir, "frameN_lines_colored_midpts.png"))

    # plot line segments, colored, numbered
    img0_lines_colored_numbered = plot_lines(img0, matched_lines0, color=True, number=True)
    imgN_lines_colored_numbered = plot_lines(imgN, matched_linesN, color=True, number=True)
    img0_lines_colored_numbered.save(os.path.join(output_dir, "frame0_lines_colored_numbered.png"))
    imgN_lines_colored_numbered.save(os.path.join(output_dir, "frameN_lines_colored_numbered.png"))

def visualize_flow_field(flow, output_path, scale=1.0):
    """
    Visualize optical flow by drawing then as arrows.

    Args
        flow (np.array) : size (H, W, 2) optical flow of an image
        output_path (string) : directory to save the visualized flow image

    Returns
        img_PIL (PIL.Image) : the visualized flow image
    """
    # dimensions of image
    H, W, _ = flow.shape

    # initialize image
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    # visual parameters
    color = (0, 0, 255) # color of OF arrows
    spacing = 20 # between each arrow

    # plot arrows
    for y in range(0, H, spacing):
        for x in range(0, W, spacing):
            dx, dy = flow[y, x]
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)

            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

    # bookkeep
    img_PIL = Image.fromarray(img)
    img_PIL.save(output_path)

    return img_PIL