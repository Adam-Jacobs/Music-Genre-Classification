from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from classification import classification_facade


class UI:
    def __init__(self, window):
        window.title('Music Genre Classification System')
        window_width = 1000
        window_height = 500
        window.geometry(str(window_width+20) + 'x' + str(window_height+40))
        window.resizable(0, 0)

        self.window = window
        self.classification_result_label = None
        self.populate_window()

    '''Adds the elements that make up the Classification page to the window'''
    def populate_window(self):
        # Classify mp3 file Button
        button = Button(self.window, text='Classify mp3 genre', command=self.classify_mp3)
        button.grid(row=0, column=0)

        # Genre Classification Label
        label = Label(self.window, text='No mp3 currently classified')
        label.grid(row=0, column=1)
        self.classification_result_label = label

        # Organise Library Button
        #button = Button(self.window, text='Organise Library', command=self.organise_library)
        #button.grid(row=0, column=0)

        # Genre Classification Label
        #label = Label(self.window, text='No mp3 currently classified')
        #label.grid(row=0, column=1)
        #self.classification_result_label = label

    def classify_mp3(self):
        file_path = filedialog.askopenfilename(title='Select mp3 file to classify', filetypes=[('mp3 files', '*.mp3')])
        result = classification_facade.classify(file_path)
        self.update_prediction(result)

    def organise_library(self):
        library_path = filedialog.askdirectory(title='Select your music Library')
        #result = classification_facade.classify(file_path)
        # Find all paths of mp3 files only within library
        # Classify each music file within directory
        # Create new directory
        # Organise songs into their genre


    def update_prediction(self, prediction):
        self.classification_result_label.config(text=prediction)

    '''Displays a pie chart illustrating the percentages of songs in their music library belonging to each genre'''
    def display_library_statistics(self):
        pass



window = Tk()
UI(window)
window.mainloop()