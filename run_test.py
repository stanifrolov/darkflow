from darkflow.cli import cliHandler
import os

def getLatestCheckpoint():
    imgdir = os.path.join('/home/frolov/U/darkflow/ckpt')  # gibt path von Checkpoints an
    file_list = []
    for dirpath, dirnames, files in os.walk(imgdir):   # os.walk gibt Pfad an, alle Ordner im Ordner und alle Dateien im Ordner.
            for f in files:
                if f.endswith('.index'):
                    filename = os.path.join(dirpath, f)
                    file_list.append(filename)
    file_list.sort()
    file = file_list[len(file_list) - 1]
    file = file.split('.index')
    file = file[0].split('-')
    return file[len(file) - 1]

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
last_ckpt = getLatestCheckpoint()
command = './flow --model cfg/full_motnet.cfg --train --gpu 0.9 --batch 1 --seq_length 2' + ' --load ' + last_ckpt 

"""
Run the command
"""
command = command.split()
cliHandler(command)
