import os
import cv2

def init_results_dirs(output_dir, visualize):
    """
    Create the incremental trial directory and its child directories for the following components:
    - input frames
    - feature visualizations
    - conditions path
    - output frames

    Args
        output_dir (string) : path to the directory for all results frames

    Returns
        trial_dir (string) : the filepath for this run/trial
        inputs_dir (string) : the filepath containing the input frames 
        features_dir (string) : the filepath containing the feature visualization 
        conditions_dir (string) : the filepath containing the condition frames 
        out_frames_dir (string) : the filepath containing the output inbetweening frames
    """
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # increment the child folder number in results directory by 1
    # i.e. if the last folder is /003, then set new directory as /004
    output_dir_num_files = len(os.listdir(output_dir))
    trial_dir = os.path.join(output_dir, '{:04d}'.format(output_dir_num_files))
    # create the new folder for this trial
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

    # create the paths
    input_dir = trial_dir + "/inputs"
    conditions_dir = trial_dir + "/conditions"
    out_frames_dir = trial_dir + "/out_frames"

    # create directories if they doesn't exist
    if not os.path.exists(input_dir): os.mkdir(input_dir)
    if not os.path.exists(conditions_dir): os.mkdir(conditions_dir)
    if not os.path.exists(out_frames_dir): os.mkdir(out_frames_dir)

    # for visualizations only
    if visualize:
        visuals_dir = trial_dir + "/visuals"
        fg_dir = visuals_dir + "/lines_fg"
        bg_dir = visuals_dir + "/lines_bg"
        all_dir = visuals_dir + "/lines_all"
        flow_dir = visuals_dir + "/flow_field"
        if not os.path.exists(visuals_dir): os.mkdir(visuals_dir)
        if not os.path.exists(fg_dir): os.mkdir(fg_dir)
        if not os.path.exists(bg_dir): os.mkdir(bg_dir)
        if not os.path.exists(all_dir): os.mkdir(all_dir)
        if not os.path.exists(flow_dir): os.mkdir(flow_dir)

    if visualize:
        return trial_dir, input_dir, conditions_dir, out_frames_dir, fg_dir, bg_dir, all_dir, flow_dir
    else:
        return trial_dir, input_dir, conditions_dir, out_frames_dir

def extract_frames_from_mp4(mp4_filepath):
    """
    Take an mp4 file and extract the frames as pngs into a folder

    Args
        mp4_filepath (string) : filepath of the mp4

    Returns
        output_dir (string) : directory of the output png frames
    """
    # get the name of the mp4 file
    mp4_filename = mp4_filepath.split("/")[-1] # get the string after the last folder branch split
    mp4_filename = mp4_filename.split(".mp4")[0] # get the string before the .mp4 extn

    # specify output directory:
    output_dir_len = len(mp4_filepath) - len(".mp4")
    output_dir = mp4_filepath[:output_dir_len]
    
    # create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # capture the video 
    vidcap = cv2.VideoCapture(mp4_filepath)

    # read the first frame
    read_success, png = vidcap.read()

    # error message for if it doesn't work
    if not read_success:
        raise Exception("mp4 read() unsuccessful")
    
    # current frame count
    frame_count = 0
    # iteratively read over the frames until none are read
    while read_success:
        # output path
        output_path = output_dir + "/" + mp4_filename + "-" + str(frame_count) + ".png"

        # save the current frame as a png
        cv2.imwrite(output_dir + "/" + mp4_filename + "-" + str(frame_count) + ".png", png)

        # go to the next frame       
        read_success, png = vidcap.read()

        # increment the frame count
        frame_count += 1

    return output_dir
