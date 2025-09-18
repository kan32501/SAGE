import torch
import os
import sys
from safetensors.torch import load_file

# base models' imports
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from models.controlnext_vid_svd import ControlNeXtSDVModel
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from pipeline.pipeline_FCVG import StableVideoDiffusionPipelineControlNeXtReverse

# Gluestick imports
from models.gluestick import GLUESTICK_ROOT
from models.gluestick.models.two_view_pipeline import TwoViewPipeline

# SEARAFT imports
SEARAFT_dir = os.path.dirname(os.path.dirname(__file__)) + "/models/SEARAFT"
sys.path.append(SEARAFT_dir) # python should search in this directory
sys.path.append(SEARAFT_dir + "/core") # python should search in this directory
from models.SEARAFT.core.raft import RAFT
from models.SEARAFT.core.utils.utils import load_ckpt

def load_models(args, progress=True):
    """
    Load the required models for the transition generation
    """
    # progress updates
    if progress: print("\n–– LOADING MODELS ––")

    # set CUDA as GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the required models
    FCVG_model = load_FCVG(args, progress)
    GlueStick_model = load_gluestick(args, device, progress)
    SEARAFT_model = load_SEARAFT(args, device, progress)

    return FCVG_model, GlueStick_model, SEARAFT_model

def load_FCVG(args, progress=True):
    """
    Load the FCVG model with ControlNext, Diffusion UNet, VAE, and CLIP image encoder.
    """
    # initialize the diffusion model that takes noisy frames, conditions, & timestep and denoises
    if progress: print("-> Loading ControlNext Model")
    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        # location for cache for huggingface models
        cache_dir = "/cs/student/projects3/cgvi/2024/akamokan/HUGGING_FACE_CACHE",
        # the pretrained model, stable-video-diffusion-img2vid-xt-1-1
        # does not have a diffusion_pytorch_model.bin file, but has a .safetensor
        # file to use. So use the .safetensor
        use_safetensors=True,
        # reduces size of floating point from 32 to 16 bits. saves space on GPU
        variant="fp16",
    )

    # load ControlNext weights
    if progress: print("-> Loading ControlNext Weights")
    controlnext = ControlNeXtSDVModel()
    controlnext.load_state_dict(load_tensor(args.controlnext_path))

    # Load weights for FCVG model. The unet contains the denoising information
    unet.load_state_dict(load_tensor(args.unet_path), strict=False)

    # load image encoder from SVD
    if progress: print("-> Loading CLIP Encoder Model")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder")
    
    # load VAE from SVD
    if progress: print("-> Loading Autoencoder Model")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, 
        # location for cache for huggingface models
        cache_dir = "/cs/student/projects3/cgvi/2024/akamokan/HUGGING_FACE_CACHE",
        # use_safetensors=True, # use .safetensors format instead of .bin
        subfolder="vae") # pretrained_model_name_or_path/vae
        # variant="fp16") # use the fp16 instead of fp32 to safe space on GPU
        # the pretrained model, stable-video-diffusion-img2vid-xt-1-1
        # does not have a diffusion_pytorch_model.bin file, but has a .safetensor
        # file to use. So use the .safetensor
        # use_safetensors=True)

    # initialize FCVG inference model with ControlNext, the denoising unet, the vae, image encoder
    if progress: print("-> Loading FCVG Model")
    FCVG_model = StableVideoDiffusionPipelineControlNeXtReverse.from_pretrained(
        args.pretrained_model_name_or_path,
        # location for cache for huggingface models
        cache_dir = "/cs/student/projects3/cgvi/2024/akamokan/HUGGING_FACE_CACHE",
        controlnext=controlnext, 
        unet=unet,
        vae=vae,
        image_encoder=image_encoder)
    
    # pipeline.to(dtype=torch.float16)
    # a function that saves space on the GPU
    FCVG_model.enable_model_cpu_offload()

    return FCVG_model

def load_gluestick(args, device, progress=True):
    """
    Load the Gluestick line detection model
    """
    # Define Gluestick model configuration
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True, # this isn't even used ...
        # specify extractor model (detects keypoints/interest points)
        'extractor': {
            'name': 'wireframe',
            # superpoint parameters
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            # wireframe parameters
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': args.max_lines,
        },
        # specify line matcher model
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    # initialize GlueStick on the GPU
    if progress: print("-> Loading GlueStick Model")
    GlueStick_model = TwoViewPipeline(conf).to(device).eval()

    return GlueStick_model

def load_SEARAFT(args, device, progress=True):
    """
    Load the SEARAFT model for optical flow estimation.
    """
    if progress: print("-> Loading SEARAFT Model")
    # Define SEARAFT config
    if args.SEARAFT_path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.SEARAFT_path is not None:
        SEARAFT_model = RAFT(args)
        load_ckpt(SEARAFT_model, args.SEARAFT_path)
    else:
        SEARAFT_model = RAFT.from_pretrained(args.url, args=args)

    # move SEARAFT to CUDA
    SEARAFT_model = SEARAFT_model.to(device)
    # set SEARAFT to inference mode
    SEARAFT_model.eval()

    return SEARAFT_model

def load_tensor(tensor_path):
    """
    load the .safetensors from a filepath

    Args:
        tensor_path (str): Path to the tensor file.

    Returns:
        Any: Loaded tensor object, or a PyTorch state dict.
    """
    # check validity of filepath entension for .bin/.safetensors
    if os.path.splitext(tensor_path)[1] == '.bin':
        return torch.load(tensor_path)
    elif os.path.splitext(tensor_path)[1] == ".safetensors":
        return load_file(tensor_path)
    else:
        print("without supported tensors")
        os._exit()