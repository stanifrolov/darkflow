from darkflow.cli import cliHandler

"""
Test data
"""
#command = 'flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/test/MOT17-14-FRCNN/img1 --model cfg/yolo.cfg --load bin/yolo.weights'
#command = 'flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/VOCdevkit/VOC2007/JPEGImages --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'

"""
Working configs
"""
#command = 'flow --model cfg/yolo.cfg --load bin/yolo.weights'
#command = 'flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'

"""
Training
"""
command = './flow --model cfg/full_motnet.cfg --train --gpu 0.9'

"""
Run the command
"""
command = command.split()
cliHandler(command)
