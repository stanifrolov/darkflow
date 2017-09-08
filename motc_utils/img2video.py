import os
import subprocess

all_dirs = ['/home/frolov/U/MOTC/MOT17/train/MOT17-02-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-04-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-05-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-09-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-10-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-11-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/train/MOT17-13-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-01-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-03-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-06-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-07-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-08-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-12-FRCNN/img1',
            '/home/frolov/U/MOTC/MOT17/test/MOT17-14-FRCNN/img1',
            '/home/frolov/U/boy',
            '/home/frolov/U/tud']


def get_dataset_name_from_path(path):
  if 'MOT17' in path:
    return path.split("/")[6] + "-" + path.split("/")[7].split("-")[1]
  else:
    return path.split("/")[4]


def img_to_video(path):
    print("Creating video")
    os.chdir(path + "/out")
    command = "ffmpeg -y -framerate 25 -pattern_type glob -i '*.jpg' /home/frolov/U/" + get_dataset_name_from_path(path) + ".mp4"
    subprocess.Popen(command, shell=True)


for path in all_dirs:
  img_to_video(path)