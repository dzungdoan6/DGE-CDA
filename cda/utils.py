import pickle

def save_variable(file_name, data):
    with open(file_name, "wb") as file:
        pickle.dump(data, file);

def load_variable(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data;