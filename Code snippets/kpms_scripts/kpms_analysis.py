import keypoint_moseq as kpms

project_dir = '2024-02-04_conspecific_odor'

FPS = 200


def config(): return kpms.load_config(project_dir)

dlc_config = '/gpfs/projects/acad/behavior/DLC_model/primary_DeepLabCut_model/LassanceLab-main-model-2023-09-14/config.yaml'
kpms.setup_project(project_dir, deeplabcut_config=dlc_config)

kpms.update_config(
    project_dir,
    video_dir='/gpfs/projects/acad/behavior/DLC_model/primary_DeepLabCut_model/LassanceLab-main-model-2023-09-14/videos/',
    anterior_bodyparts=['nose'],
    posterior_bodyparts=['tail_base'],
    use_bodyparts=[
        'nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye',
        'right_eye', 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2',
        'mid_backend3', 'tail_base', 'left_shoulder', 'left_midside', 'left_hip',
        'right_shoulder', 'right_midside', 'right_hip', 'head_midpoint'])

# load data (e.g. from DeepLabCut)
# can be a file, a directory, or a list of files
keypoint_data_path = [
    '/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort1',
    '/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort2',
    '/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort3',
    '/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort4',
    '/gpfs/projects/acad/behavior/videos/2023-11-13_conspecific_odor_videos/cohort5'
    ]
coordinates, confidences, bodyparts = kpms.load_keypoints(
    keypoint_data_path, 'deeplabcut')

# Adjust for better parallelizing 

from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(8)

from jax_moseq.utils import set_mixed_map_gpus
set_mixed_map_gpus(5)
#

# format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

# pca = kpms.fit_pca(**data, **config())
# kpms.save_pca(pca, project_dir)

# kpms.print_dims_to_explain_variance(pca, 0.9)
# kpms.plot_scree(pca, project_dir=project_dir)
# kpms.plot_pcs(pca, project_dir=project_dir, **config())

# use the following to load an already fit model
pca = kpms.load_pca(project_dir)

kpms.update_config(project_dir, latent_dim=7)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())

# optionally modify kappa
# model = kpms.update_hypparams(model, kappa=NUMBER)

num_ar_iters = 50

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir,
    ar_only=True, num_iters=num_ar_iters)


# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters)

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e4)

# run fitting for an additional 500 iters
model = kpms.fit_model(
    model, data, metadata, project_dir, model_name, ar_only=False,
    start_iter=current_iter, num_iters=current_iter+200)[0]

# modify a saved checkpoint so syllables are ordered by frequency
kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

# load the most recent model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name)

# extract results
results = kpms.extract_results(model, metadata, project_dir, model_name)

results = kpms.load_results(project_dir, model_name)
kpms.generate_trajectory_plots(
    coordinates, results, project_dir, model_name, **config())

kpms.generate_grid_movies(results, project_dir,
                          model_name, coordinates=coordinates, **config())

kpms.plot_similarity_dendrogram(
    coordinates, results, project_dir, model_name, **config())
