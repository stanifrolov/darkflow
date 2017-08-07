from darkflow.cli import cliHandler

"""
Test data
"""
#command = './flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-14-FRCNN/img1 --model cfg/full_motnet.cfg --load 5000 --batch 1 --seq_length 30 --threshold 0.55'
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
command = './flow --model cfg/full_motnet.cfg --load 17000 --train --gpu 0.9'

"""
Run the command
"""
command = command.split()
cliHandler(command)
