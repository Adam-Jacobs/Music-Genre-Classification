import feature_visualisation as vis

# TODO make a way to load in the data and keep it here instead of loading it in every time before a plot

if __name__ == "__main__":
    perplex_upper = int(input('Upper limit to perplexity: '))
    step_num = int(input('Step amount: '))
    num_tracks = int(input('Number of tracks to create the plot from: '))
    feature_type = input('Spectrogram Features (SF) or Numerical Features (NF): ')
    feature_type = feature_type.upper()

    if feature_type != 'SF' and feature_type != 'NF':
        print('Feature type not recognized')
    else:
        num_plots = int(perplex_upper / step_num)
        print('This will generate ' + str(num_plots) + ' plots')

        for perplexity in range(step_num, perplex_upper + 1, step_num):
            vis.create_tsne_plot(perplexity, num_tracks, feature_type, str(num_tracks) + ' tracks, perplexity=' + str(perplexity))
