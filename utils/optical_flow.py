import numpy as np
from PIL import Image
from utils.line_operations import get_midpoints

def query_flow_at_points(flow, points):
    """
    Query the optical flow at certain points in an image

    Args
        flow (np.array) : size (H, W, 2) optical flow of an image
        points (np.array) : size (n_points, 1, 2) points in [x, y] form

    Returns
        flow_queries (np.array) : size (n_points, 1, 2) [u, v] flow direction for each point
    """
    # get number of points
    n_pts = points.shape[0]

    # flip [x, y] to [y, x] for H,W compatibility
    points_hw = np.zeros_like(points)
    points_hw[:, :, 0] = points[:, :, 1]
    points_hw[:, :, 1] = points[:, :, 0]

    # initialize the flow record
    flow_queries = np.zeros(shape=(n_pts, 1, 2))

    # extract flow at each point
    for i in range(n_pts):
        # get coordinate [x, y] (floats)
        pt = points_hw[i, 0, :]
        # optical flow at [y, x] in image
        flow_yx = flow[int(pt[0]), int(pt[1]), :] # [u, v]

        # bookkeeping at ith point
        flow_queries[i, :, :] = flow_yx

    return flow_queries

def normalize_flow(flow):
    """
    Normalize the optical flow to be in the range [-1, 1] in both u,v directions with 
    respect to the max u, max v values.

    Since we want to preserve the direction of the flow, we normalize based on the
    optical flow vectors' magnitudes. The vector with the maximum magnitude will
    be normalized to have a magnitude of 1.

    Args
        flow (np.array) : size (H, W, 2) optical flow
    
    Returns
        flow_normalized (np.array) : size (H, W, 2) normalized flow
    """
    # get maximum magnitude vector
    mags = np.linalg.norm(flow, axis=2)
    max_mag = np.max(mags)
    
    # normalize the flow
    flow_normalized = np.zeros_like(flow)
    flow_normalized[:, :, 0] = flow[:, :, 0] / max_mag if max_mag != 0 else flow[:, :, 0]
    flow_normalized[:, :, 1] = flow[:, :, 1] / max_mag if max_mag != 0 else flow[:, :, 1]

    return flow_normalized

def interpolate_flow(flow0, flow1, num_frames):
    """
    Apply a linear interpolation between two vector fields (optical flow fields). 

    Args
        flow0 (np.array) : starting optical flow
        flow1 (np.array) : ending optical flow
        num_frames (int) : number of intermediate steps

    Returns
        intermediate_flows (List[np.array]) : num_frames numpy arrays of intermediate optical flows
    """
    # initialize output list
    intermediate_flows = []
    intermediate_flows.append(flow0) # add the first flow

    # generate each intermediate step
    interval = 1.0 / (num_frames + 1)
    for i in range(num_frames):
        # weightage parameter
        alpha = interval * (i + 1)

        # LERP
        intermediate_flow = flow0 * (1 - alpha) + flow1 * alpha

        # bookkeep
        intermediate_flows.append(intermediate_flow)

    # add the last flow
    intermediate_flows.append(flow1) 

    return intermediate_flows



