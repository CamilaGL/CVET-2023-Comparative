import os
import re

def hasDigits(inputString):
    return bool(re.search(r'\d', inputString))

join = os.path.join

def makedir(outputpath):
    os.makedirs(outputpath, exist_ok = True) #if the folder doesn't exist, we make it

def isdir(dirpath):
    return os.path.isdir(dirpath)

def isfile(filepath):
    return os.path.isfile(filepath)

def getsubdirs(ipath, join = True, contains = None, prefix = None, suffix = None, number = False, sort = True):
    '''join: return dir merged with path
    contains: check that this is part of the path
    prefix: check if this is at the beginning of the path
    suffix: check if this is at the end of the path
    number: check if dir has digit
    sort: return dirs sorted'''
    
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(ipath, i) for i in os.listdir(ipath) if os.path.isdir(os.path.join(ipath, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))
            and (contains is None or contains in i)
            and (not number or hasDigits(i))]
    if sort:
        res.sort()
    return res

def getsubfiles(ipath, join=True, contains = None, prefix=None, suffix=None, sort=True):
    '''join: return filename merged with path
    contains: check that this is part of the path
    prefix: check if this is at the beginning of the path
    suffix: check if this is at the end of the path
    sort: return files sorted'''

    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(ipath, i) for i in os.listdir(ipath) if os.path.isfile(os.path.join(ipath, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))
            and (contains is None or contains in i)]
    if sort:
        res.sort()
    return res
