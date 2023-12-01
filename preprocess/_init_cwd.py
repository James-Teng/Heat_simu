# import this file to change current working directory to project root
# all relative path start from project root
# '.' == project root

import os

project_root = 'Heat_simu'
cur_dir = os.path.dirname(__file__)
while os.path.basename(cur_dir) != project_root:
    cur_dir = os.path.dirname(cur_dir)
os.chdir(cur_dir)
print(f'current working directory change to \'{cur_dir}\'\n')

