import json

def writejson(dict, outputFile):
    with open(outputFile, 'w') as f:
        json.dump(dict, f, sort_keys=True, indent=4)

def readjson(inputFile):
    with open(inputFile, 'r') as f:
        a = json.load(f)
    return a