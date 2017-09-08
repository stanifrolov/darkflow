import os
import subprocess

all_motc_dirs = ['/home/frolov/U/MOTC/MOT17/train/MOT17-02-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-04-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-05-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-09-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-10-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-11-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/train/MOT17-13-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-01-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-03-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-06-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-07-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-08-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-12-FRCNN/img1/out',
                '/home/frolov/U/MOTC/MOT17/test/MOT17-14-FRCNN/img1/out']


for folder in all_motc_dirs:
  os.chdir(folder)
  command = "ffmpeg -y -framerate 25 -pattern_type glob -i '*.jpg' /home/frolov/U/" + folder.split("/")[7].split("-")[1] + ".mp4"
  subprocess.Popen(command, shell=True)


all_other_dirs = ['/home/frolov/U/boy/out',
                  '/home/frolov/U/tud/out']

for folder in all_other_dirs:
  os.chdir(folder)
  command = "ffmpeg -y -framerate 25 -pattern_type glob -i '*.jpg' /home/frolov/U/" + folder.split("/")[4] + ".mp4"
  subprocess.Popen(command, shell=True)