from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import os
from data_visualisation import data_visualisation_facade
from data_visualisation.attribute_holder import DVAttributes
from classification import classification_facade
from common import file_utils


class UI:
    def __init__(self, window):
        window.title('Music Genre Classification System')
        window_width = 1000
        window_height = 500
        window.geometry(str(window_width+20) + 'x' + str(window_height+40))
        window.resizable(0, 0)

        self.window = window
        # self.status_bar = Label(self.window, text='hello there', relief=SUNKEN, anchor=W)
        # self.status_bar.pack(side=BOTTOM, fill=X)

        # Feature Extraction Variables
        self.__FEfeatures_combobox = None
        self.__FEpaths_to_extract = None
        self.__FEnum_files_to_extract_label = None
        self.__FEcurrent_save_dir_path = None
        self.__FEcurrent_save_dir_label = None

        # Classification variables
        self.__CLclassifier_combobox = None
        self.__CLcurrent_features_file_path = None
        self.__CLcurrent_features_file_label = None
        self.__CLcurrent_save_dir_path = None
        self.__CLcurrent_save_dir_label = None

        # Data Visualisation Variables
        self.__DVAttributes = DVAttributes()
        self.__DVcurrent_features_file_label = None
        self.__DVcurrent_save_dir_path_label = None

        page_feature_extraction, page_data_visualisation, page_classification = self.populate_notebook(window_width, window_height)

        self.populate_feature_extraction_page(page_feature_extraction)
        self.populate_data_visualisation_page(page_data_visualisation)
        self.populate_classification_page(page_classification)

    '''Adds elements wo window toolbar'''
    def populate_toolbar(self):
        toolbar = Menu(self.window)
        self.window.config(menu=toolbar)

        menu1 = Menu(toolbar, tearoff=0)
        toolbar.add_cascade(label='Pages', menu=menu1)
        menu1.add_command(label='Feature Extraction', command=self.populate_feature_extraction_page)
        menu1.add_command(label='Data Visualisation', command=self.populate_data_visualisation_page)
        menu1.add_command(label='Classification', command=self.populate_classification_page)

    '''Creates the tabbed pages that will contain the different pages of functionality in the window'''
    def populate_notebook(self, width, height):
        notebook = Notebook(self.window)

        page_feature_extraction = Frame(notebook, width=width, height=height)
        page_data_visualisation = Frame(notebook, width=width, height=height)
        page_classification = Frame(notebook, width=width, height=height)

        notebook.add(page_feature_extraction, text='Feature Extraction')
        notebook.add(page_data_visualisation, text='Data Visualisation')
        notebook.add(page_classification, text='Classification')
        notebook.grid(column=0)

        return page_feature_extraction, page_data_visualisation, page_classification

    '''Adds the elements that make up the Feature Extraction page to the window'''
    def populate_feature_extraction_page(self, page):
        # Feature Choice
        label = Label(page, text='Feature Type: ')
        label.grid(row=0, column=0)

        combobox = Combobox(page, state="readonly", values=["Numerical", "Spectrogram"])
        combobox.grid(row=0, column=1)
        combobox.current(0)
        self.__FEfeatures_combobox = combobox

        # Select Music Files to Extract
        label = Label(page, text='Files to Extract: ')
        label.grid(row=2, column=0)

        label = Label(page, text='no file(s) selected...')
        label.grid(row=2, column=1)
        self.__FEnum_files_to_extract_label = label

        button = Button(page, text='Select Music File', command=self.select_music_file)
        button.grid(row=3, column=0)

        button = Button(page, text='Select Directory', command=self.select_music_directory)
        button.grid(row=3, column=1)

        # Select Save Directory
        label = Label(page, text='no directory selected...')
        label.grid(row=4, column=1)
        self.__FEcurrent_save_dir_label = label

        button = Button(page, text='Select Save Directory', command=lambda: self.select_save_directory('fe'))
        button.grid(row=4, column=0)

        button = Button(page, text='Extract Features', command=self.extract_features)
        button.grid(row=5, column=0)

    '''Adds the elements that make up the Data Visualisation page to the window'''
    def populate_data_visualisation_page(self, page):
        # Upper Perplexity Limit
        label = Label(page, text='Perplexity Limit: ')
        label.grid(row=0, column=0)

        entry = Entry(page, textvariable=self.__DVAttributes.perplexity_limit)
        entry.grid(row=0, column=1)

        # Step Increment
        label = Label(page, text='Step Increment: ')
        label.grid(row=1, column=0)

        entry = Entry(page, textvariable=self.__DVAttributes.step_increment)
        entry.grid(row=1, column=1)

        # Num Tracks
        label = Label(page, text='Number of Tracks: ')
        label.grid(row=2, column=0)

        entry = Entry(page, textvariable=self.__DVAttributes.num_tracks)
        entry.grid(row=2, column=1)

        # Select Features File
        label = Label(page, text='no file selected...')
        label.grid(row=3, column=1)
        self.__DVcurrent_features_file_label = label

        button = Button(page, text='Select Features File', command=lambda: self.select_features_file('dv'))
        button.grid(row=3, column=0)

        # Select Save Directory
        label = Label(page, text='no directory selected...')
        label.grid(row=4, column=1)
        self.__DVcurrent_save_dir_path_label = label

        button = Button(page, text='Select Save Directory', command=lambda: self.select_save_directory('dv'))
        button.grid(row=4, column=0)

        # Normalise Checkbox
        checkbutton = Checkbutton(page, text="Normalise Features", variable=self.__DVAttributes.normalise)
        checkbutton.grid(row=5, column=0)

        # Create Plots Button
        button = Button(page, text='Create Plots', command=self.create_plots)
        button.grid(row=6, column=0)

    '''Adds the elements that make up the Classification page to the window'''
    def populate_classification_page(self, page):
        # Classifier Choice
        label = Label(page, text='Classifier: ')
        label.grid(row=0, column=0)

        combobox = Combobox(page, state="readonly", values=["Random Forest", "Convolutional Neural Network"])
        combobox.grid(row=0, column=1)
        combobox.current(0)
        self.__CLclassifier_combobox = combobox

        # Section - Training
        label = Label(page, text='Training: ')
        label.grid(row=1, column=0)

        # Select Features File
        label = Label(page, text='no file selected...')
        label.grid(row=3, column=1)
        self.__CLcurrent_features_file_label = label

        button = Button(page, text='Select Features File', command=lambda: self.select_features_file('cl'))
        button.grid(row=3, column=0)

        # Select Save Directory
        label = Label(page, text='no directory selected...')
        label.grid(row=4, column=1)
        self.__CLcurrent_save_dir_label = label

        button = Button(page, text='Select Save Directory', command=lambda: self.select_save_directory('cl'))
        button.grid(row=4, column=0)

        # Train Classifier Button
        button = Button(page, text='Train Classifier', command=self.train_classifier)
        button.grid(row=5, column=0)

    '''Displays status in bottom of window'''
    def add_status(self, message):
        self.status_bar.config(text=message)
        self.status_bar.pack(side=BOTTOM, fill=X)

    '''Removes status display from window'''
    def clear_status(self):
        self.status_bar.pack_forget()

    def select_features_file(self, page_origin):
        file_path = filedialog.askopenfilename(title='Select Features File',
                                               filetypes=[('csv files', '*.csv'), ('pickle files', '*.pickle')])

        file_name = os.path.split(file_path)[1]

        if page_origin == 'dv':
            self.__DVAttributes.features_file_path = file_path
            self.__DVcurrent_features_file_label.config(text=file_name)
        elif page_origin == 'cl':
            self.__CLcurrent_features_file_path = file_path
            self.__CLcurrent_features_file_label.config(text=file_name)

    def select_music_file(self):
        file_path = filedialog.askopenfilename(title='Select Music File', filetypes=[('mp3 files', '*.mp3')])

        file_name = os.path.split(file_path)[1]
        self.__FEpaths_to_extract = [file_path]
        self.__FEnum_files_to_extract_label.config(text=file_name)

    def select_music_directory(self):
        dir_path = filedialog.askdirectory(title='Select Directory Containing Music Files')

        self.__FEpaths_to_extract = file_utils.get_files(dir_path, extension='.mp3')
        self.__FEnum_files_to_extract_label.config(text=str(len(self.__FEpaths_to_extract)))

    def select_save_directory(self, page_origin):
        dir_path = filedialog.askdirectory(title='Select Save Directory')

        dir_name = os.path.split(dir_path)[1]

        if page_origin == 'fe':
            self.__FEcurrent_save_dir_path = dir_path
            self.__FEcurrent_save_dir_label.config(text=dir_name)
        elif page_origin == 'dv':
            self.__DVAttributes.save_dir_path = dir_path
            self.__DVcurrent_save_dir_path_label.config(text=dir_name)
        elif page_origin == 'cl':
            self.__CLcurrent_save_dir_path = dir_path
            self.__CLcurrent_save_dir_label.config(text=dir_name)

    '''Logic for extracting features from selected music files in Feature Extraction page'''
    def extract_features(self):
        pass

    '''Logic for creating t-SNE graphs in for Data Visualisation page'''
    def create_plots(self):
        data_visualisation_facade.create_data_visualisation(int(self.__DVAttributes.perplexity_limit.get()),
                                                            int(self.__DVAttributes.step_increment.get()),
                                                            int(self.__DVAttributes.num_tracks.get()),
                                                            self.__DVAttributes.features_file_path, "test",
                                                            self.__DVAttributes.normalise.get(),
                                                            self.__DVAttributes.save_dir_path)

    '''Logic for training a classifier for Classification page'''
    def train_classifier(self):
        classification_facade.train(self.__CLclassifier_combobox.get(), self.__CLcurrent_features_file_path, self.__CLcurrent_save_dir_path)


window = Tk()
UI(window)
window.mainloop()
