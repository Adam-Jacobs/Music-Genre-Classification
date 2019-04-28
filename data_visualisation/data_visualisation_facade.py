from data_visualisation import feature_visualisation


def create_data_visualisation(perplexity_upper, step_increment, num_tracks, features_file_path, image_name, normalise, save_path):
    for perplexity in range(step_increment, perplexity_upper + 1, step_increment):
        feature_visualisation.create_tsne_plot(perplexity, num_tracks, features_file_path, image_name, normalise=normalise, save_path=save_path)
