import subprocess
import os

folders = ['/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-02-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-04-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-05-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-09-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-10-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-11-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/MOT17-13-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-01-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-03-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-06-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-07-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-08-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-12-FRCNN/img1/',
           '/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-14-FRCNN/img1/',]

for folder in folders:
  file_list = os.listdir(folder)
  os.chdir(folder)
  for file in file_list:
    command = "ffmpeg -y -i " + str(file) + " -vf scale=416:416 " + str(file)
    subprocess.Popen(command, shell=True)
    print(file)