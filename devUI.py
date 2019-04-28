from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import os
from data_visualisation import data_visualisation_facade
from data_visualisation.attribute_holder import DVAttributes


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

        # Classification variables
        self.__current_model_label = None
        self.__current_model_name = None

        # Data Visualisation Variables
        self.__DVAttributes = DVAttributes()
        self.__current_features_file_label = None
        self.__current_save_dir_path_label = None

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
        pass

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
        button = Button(page, text='Select Features File', command=self.select_features_file)
        button.grid(row=3, column=0)

        label = Label(page, text='no file selected...')
        label.grid(row=3, column=1)
        self.__current_features_file_label = label

        # Select Save Directory
        button = Button(page, text='Select Save Directory', command=self.select_save_directory)
        button.grid(row=4, column=0)

        label = Label(page, text='no directory selected...')
        label.grid(row=4, column=1)
        self.__current_save_dir_path_label = label

        # Normalise
        checkbutton = Checkbutton(page, text="Normalise Features", variable=self.__DVAttributes.normalise)
        checkbutton.grid(row=5, column=0)

        # Create Plots Button
        button = Button(page, text='Create Plots', command=self.create_plots)
        button.grid(row=6, column=0)

    '''Adds the elements that make up the Classification page to the window'''
    def populate_classification_page(self, page):

        # Current Model Label
        label = Label(page, text='No model loaded')
        if self.__current_model_name is not None:
            label.config(text=self.__current_model_name)
        label.grid(row=0, column=0)
        self.__current_model_label = label

        # Load Model Button
        button = Button(page, text='Load Model')
        button.grid(row=0, column=1)

        # Train New Model Button
        button = Button(page, text='Train New Model')
        button.grid(row=0, column=2)

    '''Displays status in bottom of window'''
    def add_status(self, message):
        self.status_bar.config(text=message)
        self.status_bar.pack(side=BOTTOM, fill=X)

    '''Removes status display from window'''
    def clear_status(self):
        self.status_bar.pack_forget()

    def select_features_file(self):
        file_path = filedialog.askopenfilename(title='Select Features File',
                                               filetypes=[('csv files', '*.csv'), ('pickle files', '*.pickle')])

        self.__DVAttributes.features_file_path = file_path
        self.__current_features_file_label.config(text=os.path.split(file_path)[1])

    def select_save_directory(self):
        dir_path = filedialog.askdirectory(title='Select Save Directory')
        self.__DVAttributes.save_dir_path = dir_path
        self.__current_save_dir_path_label.config(text=os.path.split(dir_path)[1])

    '''Main logic for creating the data visualisations'''
    def create_plots(self):
        data_visualisation_facade.create_data_visualisation(int(self.__DVAttributes.perplexity_limit.get()),
                                                            int(self.__DVAttributes.step_increment.get()),
                                                            int(self.__DVAttributes.num_tracks.get()),
                                                            self.__DVAttributes.features_file_path, "test",
                                                            self.__DVAttributes.normalise.get(),
                                                            self.__DVAttributes.save_dir_path)


window = Tk()
UI(window)
window.mainloop()
