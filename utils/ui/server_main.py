'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-04-28 11:37:37
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-05-10 17:54:50
FilePath: \yolov5-mask-42\utils\ui\server_main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-

# First, check the file and delete the existing directory
import os
os.chdir("/project/train/src_repo/")
# Start data processing
os.system("python server_data_gen.py")
os.system("python train_server.py >> /project/train/log/log.txt")
# Start train
