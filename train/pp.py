# get project root
# project path

import os
import logging

_project_dirname = 'Heat_simu'

_cur_dir = os.path.dirname(os.getcwd())
_loop_cnt = 0
while os.path.basename(_cur_dir) != _project_dirname:
    _cur_dir = os.path.dirname(_cur_dir)
    _loop_cnt += 1
    assert _loop_cnt <= 5, "check if you are running in Project folder"

project_root = _cur_dir


def abs_path(path_in_project):
    return os.path.normpath(os.path.join(project_root, path_in_project))
