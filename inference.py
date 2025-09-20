from utils.file_io import *
from utils.eval_models import *
from utils.image_operations import *
from utils.line_operations import *
from utils.optical_flow import *
from utils.visualization import *
from utils.baselines import *

def generate_transition(args, GlueStick_model, FCVG_model, SEARAFT_model, progress=True, visualize=True):
    # progress
    if progress: print("\n–– GENERATING TRANSITION ––")

    # create a new numbered directory for this iterations in /results, with subfolders
    if visualize:
        curr_trial_dir, inputs_dir, conditions_dir, out_frames_dir, lines_dir, flow_dir = init_results_dirs(args.output_dir, visualize)
    else:
        curr_trial_dir, inputs_dir, conditions_dir, out_frames_dir = init_results_dirs(args.output_dir, visualize)

    # get input frames
    inputs = get_input_frames_by_index(args.video0_frames_dir, args.videoN_frames_dir, 
                                       args.frame0_mask_path, args.frameN_mask_path,
                                       args.width, args.height,
                                       inputs_dir, save=True)
    

    # get preprocessed input images as PIL Images
    frame0_prev, frame0 = inputs["frame0_prev"][1], inputs["frame0"][1]
    frameN, frameN_next = inputs["frameN"][1], inputs["frameN_next"][1]

    # get line matches with gluestick, convert to polar form, get midpoints
    lines0, linesN = infer_gluestick(GlueStick_model, frame0, frameN)
    if progress: 
        print("-> Lines detected")

    # infer optical flow with SEA-RAFT
    flow0, flowN = infer_SEARAFT(SEARAFT_model, args, frame0_prev, frame0, frameN, frameN_next)

    # load foreground/background masks
    frame0_mask = load_mask(args.frame0_mask_path, args.width, args.height, as_numpy=False)
    frameN_mask = load_mask(args.frameN_mask_path, args.width, args.height, as_numpy=False)
    frame0_mask_np = load_mask(args.frame0_mask_path, args.width, args.height, as_numpy=True)
    frameN_mask_np = load_mask(args.frameN_mask_path, args.width, args.height, as_numpy=True)

    # filter lines out into foreground/background
    lines0_fg, linesN_fg = filter_fg_lines(frame0_mask, lines0), filter_fg_lines(frameN_mask, linesN)
    
    # normalize the foreground lines based on the mask bounding box (we don't normalize the background)
    norm_lines0_fg = normalize_lines_by_mask(frame0_mask, lines0_fg)
    norm_linesN_fg = normalize_lines_by_mask(frameN_mask, linesN_fg)

    # extract midpoints of normalized foreground lines
    norm_midpoints0_fg, norm_midpointsN_fg = get_midpoints(norm_lines0_fg), get_midpoints(norm_linesN_fg)

    # run a cost minimization algorithm to optimally match lines based on position, for foreground + background
    matched_indices0_fg, matched_indicesN_fg = match_lines_optim(norm_midpoints0_fg, norm_midpointsN_fg)
 
    # reorder the lines and other parameters based on the matched indices
    matched_lines0_fg, matched_linesN_fg = lines0_fg[matched_indices0_fg], linesN_fg[matched_indicesN_fg]
 
    # get midpoints (line positions) of the matched lines
    matched_midpoints0_fg, matched_midpointsN_fg = get_midpoints(matched_lines0_fg), get_midpoints(matched_linesN_fg)
 
    # visualize the lines with midpoints, and optical flow field
    if visualize:
        # foreground
        visualize_lines(args.height, args.width,
                        matched_lines0_fg, matched_linesN_fg, 
                        matched_midpoints0_fg, matched_midpointsN_fg, 
                        lines_dir, frame0_mask=None, frameN_mask=None) # can apply masks if desired

        # visualize optical flow field
        visualize_flow_field(flow0, os.path.join(flow_dir, "frame0_flow.png"), scale=1.0)
        visualize_flow_field(flowN, os.path.join(flow_dir, "frameN_flow.png"), scale=1.0)

    # interpolate the detected lines using spline trajectories
    interped_lines_fg = interp_lines_spline(matched_lines0_fg, matched_linesN_fg, frame0_mask_np, frameN_mask_np, flow0, flowN, num_frames=args.frame_count)

    # generate c1-c2, the frame-wise conditions
    framewise_cond_imgs = plot_condition_imgs(frame0, interped_lines_fg, lw=2, save=True, out_dir=conditions_dir, black=True)
    if visualize:
        conditions = [framewise_cond_imgs[0]] + [framewise_cond_imgs[0]] + framewise_cond_imgs + [framewise_cond_imgs[-1]] + [framewise_cond_imgs[-1]]
        # make images into gif and save
        gif_path = os.path.join(curr_trial_dir, 'conditions.gif')
        duration = 100 # if not bef_and_aft else 150
        conditions[0].save(gif_path, save_all=True, append_images=conditions[1:], loop=0, duration=duration)

    # progress update
    if progress: print("-> Generated frame-wise conditions")

    # run inference on diffusion model 
    if progress: print("-> Running video diffusion pipeline")
    video_frames = FCVG_model(
        frame0, # start image
        frameN, # end image
        framewise_cond_imgs, # control images
        decode_chunk_size=2,
        num_frames=args.frame_count,
        motion_bucket_id=127.0, 
        fps=7,
        control_weight=args.control_weight, 
        width=args.width, 
        height=args.height, 
        min_guidance_scale=3.0, 
        max_guidance_scale=3.0, 
        frames_per_batch=args.batch_frames, 
        num_inference_steps=args.num_inference_steps, 
        overlap=args.overlap).frames
    
    # flatten the output into one list of result images
    inbetween_frames = [img for sublist in video_frames for img in sublist]

    # save generated inbetween frames as PNGs and as a gif in results/xxx/out_frames
    out_gif_path = save_out_frames(inbetween_frames, 
                                               "transition", out_frames_dir, curr_trial_dir,
                                               video0_frames_dir=args.video0_frames_dir, videoN_frames_dir=args.videoN_frames_dir,
                                               width=args.width, height=args.height)
    print("–– OUTPUT TRANSITION GIF SAVED IN " + out_gif_path + " ––")

    # export and save generated inbetween frames as MP4
    out_mp4_path = os.path.join(curr_trial_dir, 'transition.mp4')
    pngs_to_mp4(out_frames_dir, out_mp4_path, framerate=30)
    print("–– OUTPUT TRANSITION MP4 SAVED IN " + out_mp4_path + " ––")