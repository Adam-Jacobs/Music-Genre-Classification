import os


def get_files(dir):
    file_list = os.listdir(dir)
    files = []
    for entry in file_list:
        path = os.path.join(dir, entry)
        if os.path.isdir(path):
            files = files + get_files(path)
        else:
            files.append(path)

    return files


def get_file_names(dir, include_extension=False):
    file_list = os.listdir(dir)
    files = []
    for entry in file_list:
        path = os.path.join(dir, entry)
        if os.path.isdir(path):
            files = files + get_files(path)
        else:
            if include_extension:
                files.append(entry)
            else:
                files.append(os.path.splitext(entry)[0])

    return files
