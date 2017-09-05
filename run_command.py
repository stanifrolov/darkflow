from darkflow.cli import cliHandler

"""
Test data
"""
#command = './flow --imgdir /home/frolov/U/MOTC/MOT17/train/MOT17-05-FRCNN/img1 --model cfg/full_motnet.cfg --load -1 --batch 1 --seq_length 30'
#command = './flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-14-FRCNN/img1 --model cfg/yolo.cfg --load bin/yolo.weights'

#command = './flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-14-FRCNN/img1 --model cfg/tiny_motnet.cfg --load 1 --batch 6 --gpu 0.9'

#command = './flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/VOCdevkit/VOC2007/JPEGImages --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'

"""
Working configs
"""
#command = './flow --model cfg/yolo.cfg --load bin/yolo.weights'
#command = './flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'

"""
Training Commands
"""
#command = './flow --model cfg/full_motnet.cfg --load bin/yolo.weights --train --gpu 0.9 --seq_length 10'
#command = './flow --model cfg/tiny_motnet.cfg --load bin/tiny-yolo-voc.weights --train --gpu 0.9 --seq_length 3'

"""
Training on Machine
"""
command = './flow --model cfg/full_motnet.cfg --train --load bin/yolo.weights --gpu 0.9 --batch 2 --seq_length 12'

"""
Run the command
"""
command = command.split()
cliHandler(command)
