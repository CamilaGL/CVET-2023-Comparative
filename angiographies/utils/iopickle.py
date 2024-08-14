import pickle

def readPickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def writePickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)