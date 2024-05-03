import deeplabcut
import os
import sys
import glob

"""to execute the script, do: python extract.py path_to_folder_with_videos"""
config_file_path = '/gpfs/projects/acad/behavior/DLC_model/primary_DeepLabCut_model/LassanceLab-main-model-2023-09-14/config.yaml'

my_path = sys.argv[1]

video_file_path = glob.glob(os.path.join(my_path, "*.mp4"))

VideoType = 'mp4'

deeplabcut.add_new_videos(
    config_file_path,
    video_file_path,
    copy_videos=False,
    coords=None,
    extract_frames=False,
)
