import pickle


class LabelManipulator:
    def __init__(self):
        self.genre_top_levels = []
        self.top_level_genres = []
        self.load_genres()

    def load_genres(self):
        pickle_in = open("dataset labels\\pickles\\genre_top_levels.pickle", "rb")

        self.genre_top_levels = pickle.load(pickle_in)

        for _, pair in enumerate(self.genre_top_levels):
            if pair[1] not in self.top_level_genres:
                self.top_level_genres.append(pair[1])


    def get_genre_top_level(self, id):
        for _, pair in enumerate(self.genre_top_levels):
            if str(id) == str(pair[0]):
                return pair[1]

        return -1


    '''Forces genre labels into values 0 - 15'''
    def categorise_genre(self, genre_id):
        return self.top_level_genres.index(genre_id)


    '''Gets the original genre id from categorised index (which should be 0 - 15)'''
    def uncategorise_genre(self, categorised_genre_id):
        return self.top_level_genres[categorised_genre_id]
