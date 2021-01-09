import pickle
from config import PICKLE_FILENAME


# loading the model from the saved file
with open(PICKLE_FILENAME, 'rb') as f_in:
    model = pickle.load(f_in)

