import os
import sys
import time

path = r"D:\Pro\Anaconda3"
for root, dir, files in os.walk(path):
    for file in files:
        full_path = os.path.join(root, file)
        mtime = os.stat(full_path).st_mtime
        file_modify_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

        if (str.find(file_modify_time, "2020-06-29") >= 0):
            print("{0} 修改时间是: {1}".format(full_path, file_modify_time))
