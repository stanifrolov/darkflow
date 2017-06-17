from darkflow.cli import cliHandler


#command = 'flow --imgdir /Users/sfrolov/master-thesis/img/boy --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights'
#command = 'flow --model cfg/tiny-yolo-voc.cfg --train --dataset ./VOCdevkit/VOC2007/JPEGImages --annotation ./VOCdevkit/VOC2007/Annotations'
command = 'flow --model cfg/motnet.cfg --train'

command = command.split()

cliHandler(command)

