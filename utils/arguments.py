import argparse
from models.SEARAFT.config.parser import json_to_args

def parse_args():
    """
    this function loads all of the parameters needed for the video transition generation
    
    allows script customization through the command line. eg. python filename.py --param_name "VALUE"

    Args

    Returns
        args (argparse.Namespace) : Parsed arguments
    """
    # new instance of parser
    parser = argparse.ArgumentParser()

    # height and width of the transition frames
    parser.add_argument("--height",
                            type=int,
                            default=576
                        )
    parser.add_argument("--width",
                            type=int,
                            default=1024
                        )
    
    # number of inbetween frames to generate
    parser.add_argument("--frame_count",
                            type=int,
                            default=13
                        )
    
    # input video frames' directories
    parser.add_argument("--video0_frames_dir", 
                        type=str, 
                        default="./example/videos/cab"
                        )
    parser.add_argument("--videoN_frames_dir", 
                        type=str, 
                        default="./example/videos/train"
                        )

    # start frame mask from videoA
    parser.add_argument("--frame0_mask_path", 
                        type=str,
                        default="./example/masks/cab_mask.png"
                        )
    
    # end frame mask from videoB
    parser.add_argument("--frameN_mask_path", 
                        type=str, 
                        default="./example/masks/train_mask.png"
                        )

    # assign the arguments to objects in the parser in one go
    args = parser.parse_args()
    args_dict = args.__dict__ # 

    # parse FCVG args and load them into the args dictionary
    FCVG_args = parse_FCVG_args()
    for index, (key, value) in enumerate(vars(FCVG_args).items()):
        args_dict[key] = value

    # parse SEARAFT args
    SEARAFT_args = parse_SEARAFT_args()
    for index, (key, value) in enumerate(vars(SEARAFT_args).items()):
        args_dict[key] = value

    return args

def parse_SEARAFT_args():
    """
    set arguments for the SEARAFT optical flow model
    """
    # new instance of parser
    parser = argparse.ArgumentParser()

    # SEARAFT optical flow model parameters
    json_filepath = './models/SEARAFT/config/eval/spring-M.json'

    # the config JSON filepath
    parser.add_argument('--cfg', 
                            help='experiment configure file name', 
                            required=False, 
                            type=str,
                            default=json_filepath
                        )
    # the model path
    parser.add_argument('--SEARAFT_path', 
                            help='checkpoint path', 
                            type=str, 
                            default='./models/SEARAFT/models/Tartan-C-T-TSKH-kitti432x960-M.pth'
                        ) 
    # OR the model url 
    parser.add_argument('--url', 
                            help='checkpoint url', 
                            type=str, 
                            default=None)
    # inference device
    parser.add_argument('--device', 
                            help='inference device', 
                            type=str, 
                            default='cpu'
                        )

    args = parser.parse_args()
    args_dict = args.__dict__

    # load attributes from SEARAFT config file into args
    json_args = json_to_args(json_filepath) # get the SEARAFT args in config file
    for index, (key, value) in enumerate(vars(json_args).items()):
        args_dict[key] = value

    return args


def parse_FCVG_args():
    """
    set arguments for the Framewise Conditioned Video Generator model
    """
    # new instance of parser
    parser = argparse.ArgumentParser()

    # path to base I2V model
    parser.add_argument("--pretrained_model_name_or_path", 
                            type=str, 
                            default='stabilityai/stable-video-diffusion-img2vid-xt-1-1'
                        )

    # the controlnext and unet, stored in .safetensors
    parser.add_argument("--controlnext_path",
                            type=str,
                            default='checkpoints/controlnext.safetensors',
                        )
    parser.add_argument("--unet_path",
                            type=str,
                            default='checkpoints/unet.safetensors',
                        )

    # max no. of frames in between
    parser.add_argument("--max_frame_num",
                            type=int,
                            default=25
                        ) 

    # results folder
    parser.add_argument("--output_dir",
                            type=str,
                            default='./results'
                        ) 

    # parameters for the FCVG
    # frames per denoising batch
    parser.add_argument("--batch_frames",
                            type=int,
                            default=25
                        ) 
    # the influence of the control conditions on generated vid
    parser.add_argument("--control_weight",
                            type=float,
                            default=1.0
                        ) 
    # the no. of frames overlapping from the start image & inbetween frames/end image & inbetween frame
    parser.add_argument("--overlap",
                            type=int,
                            default=6
                        )
    
    parser.add_argument("--num_inference_steps",
                            type=int,
                            default=25
                        )

    # GlueStick line matching model parameters
    # max no. of keypoints for line detection
    parser.add_argument('--max_pts', 
                            type=int, 
                            default=1000
                        )
    # max no. of lines detected 
    parser.add_argument('--max_lines', 
                            type=int, 
                            default=300
                        )

    args = parser.parse_args()

    return args