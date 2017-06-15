from darkflow.cli import cliHandler

command = ["flow",
           "--model",
           "cfg/tiny-yolo-voc.cfg",
           "--load",
           "bin/tiny-yolo-voc.weights",
           "--demo",
           #"camera"]
           "/Users/sfrolov/master-thesis/img/DSC_0275.MOV",
           "--saveVideo"]

command = 'flow --model cfg/motnet.cfg --train'
command = command.split()

cliHandler(command)

# flow --imgdir /Users/sfrolov/master-thesis/img/boy --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights
