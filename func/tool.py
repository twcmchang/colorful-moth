import os
def get_fname (path):
    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created: %s' %(path))