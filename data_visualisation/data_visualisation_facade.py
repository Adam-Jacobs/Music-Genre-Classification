from data_visualisation import feature_visualisation


def create_data_visualisation(perplexity, num_tracks, feature_type, image_name, normalise, save_path):
    feature_visualisation.create_tsne_plot(perplexity, num_tracks, feature_type, image_name, normalise=normalise, save_path=save_path)
