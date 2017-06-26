from darkflow.cli import cliHandler


#command = 'flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/VOCdevkit/VOC2007/JPEGImages --model cfg/yolo.cfg --load bin/yolo.weights'
#command = 'flow --imgdir /Users/sfrolov/master-thesis/code/darkflow/VOCdevkit/VOC2007/JPEGImages --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'


#command = './flow --model cfg/tiny-yolo-voc.cfg --train --gpu 1.0 --dataset ./VOCdevkit/VOC2007/JPEGImages --annotation ./VOCdevkit/VOC2007/Annotations'
command = './flow --model cfg/motnet.cfg --train --gpu 1.0 --trainer adam'

command = command.split()

cliHandler(command)
