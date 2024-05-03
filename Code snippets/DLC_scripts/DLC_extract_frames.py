import deeplabcut
import os
import sys
import glob

"""to execute the script, do: python extract.py path_to_folder_with_videos"""
config_file_path = '/gpfs/projects/acad/behavior/DLC_model/primary_DeepLabCut_model/LassanceLab-main-model-2023-09-14/config.yaml'

my_path = sys.argv[1]

# video_file_path = ["/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort6/2023-11-06_ID064_POm.mp4"]
video_file_path = glob.glob(os.path.join(my_path, "*.mp4"))

VideoType = 'mp4'

deeplabcut.extract_frames(
    config_file_path,
    mode='automatic',
    algo='kmeans',
    userfeedback=False,
    crop=False,
    cluster_step=50,
    videos_list=video_file_path)