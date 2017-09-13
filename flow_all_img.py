from darkflow.cli import cliHandler
from motc_utils import img2video
import os

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

otb_path = '/home/frolov/U/otb/'
otb_dirs = [otb_path + dir + '/img' for dir in os.listdir(otb_path) if os.path.isdir(os.path.join(otb_path, dir))]

vot2015_path = '/home/frolov/U/vot2015/'
vot2015_dirs = [vot2015_path + dir for dir in os.listdir(vot2015_path) if os.path.isdir(os.path.join(vot2015_path, dir))]

all_dirs.extend(otb_dirs)
all_dirs.extend(vot2015_dirs)

for path in all_dirs:
    print("Now flowing " + path)
    try:
        command = './flow --imgdir ' + path + ' --model cfg/full_motnet.cfg --load -1 --batch 1 --seq_length 12 --threshold 0.5'
        command = command.split()
        cliHandler(command)
    except:
        print("Error or Done")
        img2video.img_to_video(path)
