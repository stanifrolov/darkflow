import os

all_dirs = ['/home/frolov/U/MOTC/MOT17/train/MOT17-02-FRCNN/img1/out',
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


for folder in all_dirs:
  file_list = os.listdir(folder)
  os.chdir(folder)
  command = "ffmpeg -framerate 25 -pattern_type glob -i '*.jpg' video.mp4"
  os.subprocess.Popen(command, shell=True)