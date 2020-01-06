from os.path import isdir


def verify_folder_existence(path):
    if not isdir(path):
        raise Exception(f'Folder with this path: {path} does not exist')
