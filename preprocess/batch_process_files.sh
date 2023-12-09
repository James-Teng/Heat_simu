#!/bin/bash

# 指定文件夹路径和要执行的 Python 脚本路径
folder_path="../data/data3_gap/2interval"
py_script_path="./preprocess/all_txt_to_tenor.py"

# 遍历文件夹中的所有文件
for file_path in "$folder_path"/*; do
    # 检查文件是否为普通文件
    if [ -f "$file_path" ]; then
        # 构建要执行的命令
        command="python $py_script_path -p $file_path"

        # 执行命令
        eval "$command"
    fi
done