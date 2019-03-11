import pickle


genre_top_levels = []
top_level_genres = []


def load_genres():
    try:
        pickle_in = open("..\\dataset labels\pickles\\genre_top_levels.pickle", "rb")
    except Exception:
        pickle_in = open("..\\..\\dataset labels\pickles\\genre_top_levels.pickle", "rb")

    genre_top_levels.extend(pickle.load(pickle_in))

    for _, pair in enumerate(genre_top_levels):
        if pair[1] not in top_level_genres:
            top_level_genres.append(pair[1])


def get_genre_top_level(id):
    if len(genre_top_levels) == 0:
        load_genres()

    for _, pair in enumerate(genre_top_levels):
        if str(id) == str(pair[0]):
            return pair[1]

    return -1


# Forces genre labels into values 0 - 15
def categorise_genre(genre):
    if len(genre_top_levels) == 0:
        load_genres()

    return top_level_genres.index(genre)
