# PAPER TITLE

DESCRIPTION

## Results

Animated Transition Results

<p float="left">
  <img src="readme_results/tshirt_to_street_ours.gif" width="200" />
  <img src="readme_results/bicycle_wakeboard.gif" width="200" />
  <img src="readme_results/piste_skatepark.gif" width="200" />
</p>

Baseline Comparisons

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Ours</td>
        <td>Cross-Dissolve</td>
        <td><a href="https://arxiv.org/abs/2312.07409">DiffMorpher</td>
        <td><a href="https://arxiv.org/abs/2111.14818">Blended Diffusion</a></td>
    </tr>
  <tr>
  <td>
    <img src=readme_results/turtle_tiger_ours.gif width="175">
  </td>
  <td>
    <img src=readme_results/turtle_tiger_xdis.gif width="175">
  </td>
  <td>
    <img src=readme_results/turtle_tiger_DM.gif width="175">
  </td>
  <td>
    <img src=readme_results/turtle_tiger_BD.gif width="175">
  </td>
  </tr>
  <tr>
  <td>
    <img src=readme_results/f1_surf_ours.gif width="175">
  </td>
  <td>
    <img src=readme_results/f1_surf_xdis.gif width="175">
  </td>
  <td>
    <img src=readme_results/f1_surf_DM.gif width="175">
  </td>
  <td>
    <img src=readme_results/f1_surf_BD.gif width="175">
  </td>
  </tr>
  <tr>
  <td>
    <img src=readme_results/skyscraper_heliski_ours.gif width="175">
  </td>
  <td>
    <img src=readme_results/skyscraper_heliski_xdis.gif width="175">
  </td>
  <td>
    <img src=readme_results/skyscraper_heliski_DM.gif width="175">
  </td>
  <td>
    <img src=readme_results/skyscraper_heliski_BD.gif width="175">
  </td>
  </tr> 
</table>



## Startup Guide
#### 1. Setup conda environment

```bash
$ git clone https://github.com/kan32501/<REPO-NAME>.git
$ cd <REPO-NAME>
```

```bash=
$ git clone https://github.com/kan32501/<REPO-NAME>
$ cd <REPO-NAME>
```

```bash
$ conda create -n <REPO-NAME> python=3.10.14
$ conda activate <REPO-NAME>
$ pip install -r requirements.txt
```

#### 2. Download required base models

Please refer to [Framewise Conditions-driven Video Generation](https://github.com/Tian-one/FCVG) (Steps 1-3) & [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT?tab=readme-ov-file) (Step 4) if there are any issues.

1. Download the [Gluestick](https://github.com/cvg/GlueStick) weights and put them in `./models/resources/weights`.

```bash
$ wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar -P models/resources/weights
```

2. Download the  [DWPose](https://github.com/IDEA-Research/DWPose) pretrained weights `dw-ll_ucoco_384.onnx` and `yolox_l.onnx` [here](https://drive.google.com/drive/folders/1Ftv-jR4R8VtnOyy38EVLRa0yLz0-BnUY?usp=sharing), then put them in `./checkpoints/dwpose`. 

3. Download the FCVG model [here](https://drive.google.com/drive/folders/1qIvr9WO8qk3NUdztxweTmexfkHt8oRDB?usp=sharing), put them in `./checkpoints`

4. Download a model from SEARAFT model files [here](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW). Then move it into `./models/searaft/models` . Then set `--SEARAFT_path` in `arguments.py` as `./models/SEARAFT/models/<MODEL-NAME>.pth`

#### 3. Run the inference script

Run.

```bash
$ python main.py
```

See `arguments.py` to customize input parameters. Decrease `-n` to < 10 if receiving `CUDA error: out of memory

>   -frame_count : number of intermediate frames, default=12\
>   -video0_frames_dir : path to directory of frames in videoA\
>   -videoN_frames_dir : path to directory of frames in videoB\
>   -frame0_mask_path: path to mask for last frame in videoA\
>   -frameN_mask_path: path to mask for first frame in videoB\
>   --height : output frames height, default is 576\
>   --width: output frames width, default is 1024\


## Acknowledgements

Thank you to the work of [Frame-wise Conditions-driven Video Generation](https://github.com/Tian-one/FCVG), which our codebase is modeled from!
