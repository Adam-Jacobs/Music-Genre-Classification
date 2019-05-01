import os


def get_files(dir_path, extension=None):
    """
    Finds all files within a directory

    Parameters:
    dir_path : string
        Path of directory to search for files within
    extension : string (optional)
        The extension a file must have to be included in the list of returned files

    Returns:
    list of strings
        The paths of all files contained within the search directory
    """
    file_list = os.listdir(dir_path)
    files = []
    for entry in file_list:
        # Construct whole path of file
        path = os.path.join(dir_path, entry)

        # Recursive call to this function to find files within subdirectories
        if os.path.isdir(path):
            files = files + get_files(path, extension=extension)
        else:
            # Check if extension is specified
            if extension is not None:
                # Skip this entry if it's extension does not match the specified extension
                if extension.replace('.', '') != os.path.splitext(entry)[1].replace('.', ''):
                    continue

            # Add the path of this file to the list of found files
            files.append(path)

    return files


def get_file_names(dir_path, include_extension=False):
    """
    Finds all files within a directory

    Parameters:
    dir_path : string
        Path of directory to search for files within
    extension : string (optional)
        Whether to include the file's extension with the file name

    Returns:
    list of strings
        The names of all files contained within the search directory
    """
    file_list = os.listdir(dir_path)
    files = []
    for entry in file_list:
        path = os.path.join(dir_path, entry)
        if os.path.isdir(path):
            files = files + get_files(path)
        else:
            if include_extension:
                files.append(entry)
            else:
                files.append(os.path.splitext(entry)[0])

    return files
