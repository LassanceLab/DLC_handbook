import keypoint_moseq as kpms

project_dir = 'demo_project'


def config(): return kpms.load_config(project_dir)


dlc_config = '/gpfs/projects/acad/behavior/softs/moseq_scripts/moseq_demodata/dlc_project/config.yaml'
kpms.setup_project(project_dir, deeplabcut_config=dlc_config)

kpms.update_config(
    project_dir,
    video_dir='/gpfs/projects/acad/behavior/softs/moseq_scripts/moseq_demodata/dlc_project/videos/',
    anterior_bodyparts=['nose'],
    posterior_bodyparts=['spine4'],
    use_bodyparts=[
        'spine4', 'spine3', 'spine2', 'spine1',
        'head', 'nose', 'right ear', 'left ear'])

# load data (e.g. from DeepLabCut)
# can be a file, a directory, or a list of files
keypoint_data_path = '/gpfs/projects/acad/behavior/softs/moseq_scripts/moseq_demodata/dlc_project/videos/'
coordinates, confidences, bodyparts = kpms.load_keypoints(
    keypoint_data_path, 'deeplabcut')

# format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

# use the following to load an already fit model
# pca = kpms.load_pca(project_dir)

kpms.update_config(project_dir, latent_dim=4)

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
