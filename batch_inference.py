import subprocess

commands = [
    # 'python main.py --video0_frames_dir "./example/videos/helicopter" --videoN_frames_dir "./example/videos/boat" --frame0_mask_path "./example/masks/helicopter_mask.png" --frameN_mask_path "./example/masks/boat_mask.png"',
    # 'python main.py --video0_frames_dir "./example/videos/fish" --videoN_frames_dir "./example/videos/giraffe" --frame0_mask_path "./example/masks/fish_mask.jpg" --frameN_mask_path "./example/masks/giraffe_mask.jpg"',
    # 'python main.py --height 1024 --width 576 --video0_frames_dir "./example/videos/skatepark" --videoN_frames_dir "./example/videos/biker" --frame0_mask_path "./example/masks/skatepark_mask.jpg" --frameN_mask_path "./example/masks/biker_mask.jpg"',

    # 'python main.py --height 1024 --width 576 --video0_frames_dir "./example/videos/dress" --videoN_frames_dir "./example/videos/highway" --frame0_mask_path "./example/masks/dress_mask.png" --frameN_mask_path "./example/masks/highway_mask.png"',
    # 'python main.py --height 1024 --width 576 --video0_frames_dir "./example/tshirt/" --videoN_frames_dir "./example/videos/street" --frame0_mask_path "./example/masks/tshirt_mask.jpg" --frameN_mask_path "./example/masks/street_mask.jpg"',
    # 'python main.py --height 1024 --width 576 --video0_frames_dir "./example/videos/handkerchief" --videoN_frames_dir "./example/videos/cruise" --frame0_mask_path "./example/masks/handkerchief_mask.jpg" --frameN_mask_path "./example/masks/cruise_mask.jpg"',
    # 'python main.py --height 1024 --width 576 --video0_frames_dir "./example/videos/makeup" --videoN_frames_dir "./example/videos/beach" --frame0_mask_path "./example/masks/makeup_mask.png" --frameN_mask_path "./example/masks/beach_mask.png"',

    # 'python main.py --video0_frames_dir "./example/videos/dress" --videoN_frames_dir "./example/videos/highway" --frame0_mask_path "./example/masks/dress_highway-42_mask.png" --frameN_mask_path "./example/masks/dress_highway-100_mask.png"',

    'python main.py --video0_frames_dir "./example/videos/racoon_A" --videoN_frames_dir "./example/videos/racoon_B" --frame0_mask_path "./example/masks/racoon_A_mask.png" --frameN_mask_path "./example/masks/racoon_B_mask.png"',
    'python main.py --video0_frames_dir "./example/videos/flowers_A" --videoN_frames_dir "./example/videos/flowers_B" --frame0_mask_path "./example/masks/flowers_A_mask.png" --frameN_mask_path "./example/masks/flowers_B_mask.png"'
]

for cmd in commands:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)