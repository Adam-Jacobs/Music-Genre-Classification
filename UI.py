from tkinter import *
from tkinter.ttk import *


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

    '''Creates the tabbed pages that will contain the differen pages of functionality in the window'''
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
        pass

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

    # Temp
    def method(self):
        pass


window = Tk()
UI(window)
window.mainloop()
