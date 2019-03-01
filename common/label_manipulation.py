import pickle


genre_top_levels = []


def load_genres():
    pickle_in = open("..\\dataset labels\pickles\\genre_top_levels.pickle", "rb")
    genre_top_levels.extend(pickle.load(pickle_in))


def get_genre_top_level(id):
    if len(genre_top_levels) == 0:
        load_genres()

    for _, pair in enumerate(genre_top_levels):
        if str(id) == str(pair[0]):
            return pair[1]
    return -1
