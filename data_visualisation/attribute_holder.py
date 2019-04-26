from tkinter import IntVar
from tkinter import StringVar

class DVAttributes:
    def __init__(self):
        self.perplexity_limit = StringVar()
        self.step_increment = StringVar()
        self.num_tracks = StringVar()
        self.features_file_path = ''
        self.save_dir_path = ''
        self.normalise = IntVar()

    '''def __init__(self, perplexity_limit, step_increment, num_tracks, features_file_path, save_dir_path, normalise):
        self.perplexity_limit = perplexity_limit
        self.step_increment = step_increment
        self.num_tracks = num_tracks
        self.features_file_path = features_file_path
        self.save_dir_path = save_dir_path
        self.normalise = normalise'''
