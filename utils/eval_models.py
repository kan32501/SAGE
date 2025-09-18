import numpy as np
import torch

from models.gluestick import batch_to_np, numpy_image_to_torch
from models.SEARAFT.custom import calc_flow

"""
Run SEA-RAFT. Copied code from demo_custom in SEA-RAFT/custom.py

Args:
    path : path to save the jpg results in
    model (nn.Module) : the SEA-RAFT module
    frame0_prev (PIL.Image) : frame preceding frame0, PIL Image
    frame0 (PIL.Image) : frame0, PIL Image
    frameN (PIL.Image) : frameN, PIL Image
    frameN_next(PIL.Image) : frame following frameN, PIL Image
    device : GPU device. default is torch.device('cuda')

Returns:
    flow0 (np.array) : size (H, W, 2) OF from frame0_prev to frame0
    flowN (np.array) : size (H, W, 2) OF from frameN to frameN_next
"""
def infer_SEARAFT(model, args, frame0_prev, frame0, frameN, frameN_next, device=torch.device('cuda')):
    # convert PIL images to numpy
    frame0_prev_np = np.asarray(frame0_prev)
    frame0_np = np.asarray(frame0)
    frameN_np = np.asarray(frameN)
    frameN_next_np = np.asarray(frameN_next)

    # convert to torch tensor
    frame0_prev_tensor = torch.tensor(frame0_prev_np, dtype=torch.float32).permute(2, 0, 1)
    frame0_tensor = torch.tensor(frame0_np, dtype=torch.float32).permute(2, 0, 1)
    frameN_tensor = torch.tensor(frameN_np, dtype=torch.float32).permute(2, 0, 1)
    frameN_next_tensor = torch.tensor(frameN_next_np, dtype=torch.float32).permute(2, 0, 1)

    # move to CUDA
    frame0_prev_tensor = frame0_prev_tensor[None].to(device)
    frame0_tensor = frame0_tensor[None].to(device)
    frameN_tensor = frameN_tensor[None].to(device)
    frameN_next_tensor = frameN_next_tensor[None].to(device)

    # infer optical flow from calc_flow
    # calc_flow() extracts the final iteration of the RAFT model
    flow0_tensor, info1 = calc_flow(args, model, frame0_prev_tensor, frame0_tensor)
    flowN_tensor, info2 = calc_flow(args, model, frameN_tensor, frameN_next_tensor)

    # convert to numpy
    flow0 = flow0_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    flowN = flowN_tensor[0].permute(1, 2, 0).detach().cpu().numpy()

    return flow0, flowN

"""
Runs the Gluestick Two-View pipeline to extract matching lines from frame0 & frameN.
Then linearly interpolates between the frame0 and frameN in line space. 
Interpolated poses are visualized with colored lines on a black image.

Args:
    model (torch.nn.Module): The GlueStick model
    frame0 (PIL.Image.Image): The start image.
    frameN (PIL.Image.Image): The end image.

Returns:
    lines0: detected lines in frame0
    linesN: detected lines in frameN
"""
def infer_gluestick(model, frame0, frameN, device=torch.device('cuda')):
    # convert PIL image into grayscale np arrays
    gray0 = np.array(frame0.convert('L'))
    grayN = np.array(frameN.convert('L'))

    # preprocess images for the model 
    torch_gray0, torch_grayN = numpy_image_to_torch(gray0), numpy_image_to_torch(grayN) # convert np arrays to PyTorch tensors
    torch_gray0, torch_grayN = torch_gray0.to(device)[None], torch_grayN.to(device)[None] # move tensors to GPU
    x = {'image0': torch_gray0, 'image1': torch_grayN} # prepare input format

    # run inference with the model, obtain prediction
    pred = model(x)

    # convert (nested) model output to np arrays
    pred = batch_to_np(pred)

    # extract the outputs from prediction
    kp0, kpN = pred["keypoints0"], pred["keypoints1"] # extract keypoints in im0 & im1
    m0 = pred["matches0"] # extract keypoints across im0 & im1
    # ** line_seg0 and line_seg1 holds the results of the line detection **
    line_seg0, line_seg1 = pred["lines0"], pred["lines1"] # extract lines in im0 & im1
    line_matches = pred["line_matches0"] #

    # get valid matching keypoints
    valid_matches = m0 != -1 # check that there are valid matches
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kpsN = kpN[match_indices]

    # get valid matching line segments
    valid_matches = line_matches != -1 # check that there are valid matches
    match_indices = line_matches[valid_matches]

    # **NOTE: WE ARE IGNORING THE GLUESTICK MATCHING RESULTS**
    # matched_lines0 = line_seg0[valid_matches]
    # matched_linesN = line_seg1[match_indices]

    # extract the lines
    # np.array, shape (n_lines, 2, 2) for
    # [[x1, y1]
    #  [x2, y2]]
    lines0, linesN = pred["lines0"], pred["lines1"]

    return lines0, linesN