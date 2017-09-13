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
            '/home/frolov/U/tud']

/home/frolov/U/otb/Dancer/img
/home/frolov/U/otb/Biker/img
/home/frolov/U/otb/Jogging/img
/home/frolov/U/otb/Girl/img
/home/frolov/U/otb/Jumping/img
/home/frolov/U/otb/Skater/img
/home/frolov/U/otb/Subway/img
/home/frolov/U/otb/Freeman1/img
/home/frolov/U/otb/Dudek/img
/home/frolov/U/otb/Human2/img
/home/frolov/U/otb/Matrix/img
/home/frolov/U/otb/Soccer/img
/home/frolov/U/otb/David/img
/home/frolov/U/otb/KiteSurf/img
/home/frolov/U/otb/Diving/img
/home/frolov/U/otb/Boy/img
/home/frolov/U/otb/Human6/img
/home/frolov/U/otb/David2/img
/home/frolov/U/otb/FleetFace/img
/home/frolov/U/otb/Human4/img
/home/frolov/U/otb/FaceOcc2/img
/home/frolov/U/otb/Human8/img
/home/frolov/U/otb/Dancer2/img
/home/frolov/U/otb/Walking2/img
/home/frolov/U/otb/Human9/img
/home/frolov/U/otb/Couple/img
/home/frolov/U/otb/Skater2/img
/home/frolov/U/otb/Surfer/img
/home/frolov/U/otb/Crowds/img
/home/frolov/U/otb/Trellis/img
/home/frolov/U/otb/Human7/img
/home/frolov/U/otb/Girl2/img
/home/frolov/U/otb/Singer1/img
/home/frolov/U/otb/Gym/img
/home/frolov/U/otb/Mhyang/img
/home/frolov/U/otb/Skating2/img
/home/frolov/U/otb/Jump/img
/home/frolov/U/otb/David3/img
/home/frolov/U/otb/Human3/img
/home/frolov/U/otb/Woman/img
/home/frolov/U/otb/BlurBody/img
/home/frolov/U/otb/Human5/img
/home/frolov/U/vot2015/bag
/home/frolov/U/vot2015/bolt2
/home/frolov/U/vot2015/graduate
/home/frolov/U/vot2015/leaves
/home/frolov/U/vot2015/marching
/home/frolov/U/vot2015/glove
/home/frolov/U/vot2015/crossing
/home/frolov/U/vot2015/sphere
/home/frolov/U/vot2015/gymnastics1
/home/frolov/U/vot2015/singer3
/home/frolov/U/vot2015/fernando
/home/frolov/U/vot2015/gymnastics2
/home/frolov/U/vot2015/book
/home/frolov/U/vot2015/iceskater2
/home/frolov/U/vot2015/gymnastics4
/home/frolov/U/vot2015/tunnel
/home/frolov/U/vot2015/handball2
/home/frolov/U/vot2015/racing
/home/frolov/U/vot2015/bolt1
/home/frolov/U/vot2015/blanket
/home/frolov/U/vot2015/fish4
/home/frolov/U/vot2015/dinosaur
/home/frolov/U/vot2015/pedestrian1
/home/frolov/U/vot2015/wiper
/home/frolov/U/vot2015/ball2
/home/frolov/U/vot2015/car1
/home/frolov/U/vot2015/handball1
/home/frolov/U/vot2015/motocross2
/home/frolov/U/vot2015/birds2
/home/frolov/U/vot2015/road
/home/frolov/U/vot2015/girl
/home/frolov/U/vot2015/hand
/home/frolov/U/vot2015/iceskater1
/home/frolov/U/vot2015/singer1
/home/frolov/U/vot2015/bmx
/home/frolov/U/vot2015/godfather
/home/frolov/U/vot2015/fish3
/home/frolov/U/vot2015/fish2
/home/frolov/U/vot2015/birds1
/home/frolov/U/vot2015/helicopter
/home/frolov/U/vot2015/gymnastics3
/home/frolov/U/vot2015/tiger
/home/frolov/U/vot2015/soldier
/home/frolov/U/vot2015/motocross1
/home/frolov/U/vot2015/butterfly
/home/frolov/U/vot2015/ball1
/home/frolov/U/vot2015/singer2
/home/frolov/U/vot2015/octopus
/home/frolov/U/vot2015/soccer2
/home/frolov/U/vot2015/pedestrian2
/home/frolov/U/vot2015/shaking
/home/frolov/U/vot2015/matrix
/home/frolov/U/vot2015/sheep
/home/frolov/U/vot2015/basketball
/home/frolov/U/vot2015/nature
/home/frolov/U/vot2015/rabbit
/home/frolov/U/vot2015/fish1
/home/frolov/U/vot2015/traffic
/home/frolov/U/vot2015/car2
/home/frolov/U/vot2015/soccer1


otb_path = '/home/frolov/U/otb/'
otb_dirs = [otb_path + dir + '/img' for dir in os.listdir(otb_path) if os.path.isdir(os.path.join(otb_path, dir))]

vot2015_path = '/home/frolov/U/vot2015/'
vot2015_dirs = [vot2015_path + dir for dir in os.listdir(vot2015_path) if os.path.isdir(os.path.join(vot2015_path, dir))]

for el in otb_dirs:
	print(el)

for el in vot2015_dirs:
	print(el)

#all_dirs.extend(otb_dirs)
#all_dirs.extend(vot2015_dirs)

#for path in all_dirs:
#    print("Now flowing " + path)
#    try:
#        command = './flow --imgdir ' + path + ' --model cfg/full_motnet.cfg --load -1 --batch 1 --seq_length 12 --threshold 0.5'
#        command = command.split()
#        cliHandler(command)
#    except:
#        print("Error or Done")
#        img2video.img_to_video(path)

#response = input("All images flowed. Do you want to create videos? (y/n): ")

#if response is "y":
#  for path in all_dirs:
#      print("Creating video for " + path)
#      img2video.img_to_video(path)
