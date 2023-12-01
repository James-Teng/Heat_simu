# import this file to change current working directory to project root
# all relative path start from project root
# '.' == project root

import os

project_root = 'Heat_simu'
cur_dir = os.path.dirname(__file__)
loop_limit = 10
while os.path.basename(cur_dir) != project_root and loop_limit > 0:
    cur_dir = os.path.dirname(cur_dir)
    loop_limit -= 1
os.chdir(cur_dir)
print(f'current working directory change to \'{cur_dir}\'\n')

