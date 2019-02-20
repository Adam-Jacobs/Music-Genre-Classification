import os

def getFiles(dir):
    fileList = os.listdir(dir)
    files = list()
    for entry in fileList:
        path = os.path.join(dir, entry)
        if os.path.isdir(path):
            files = files + getFiles(path)
        else:
            files.append(path)

    return files