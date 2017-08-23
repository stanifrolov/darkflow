from darkflow.cli import cliHandler

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
        
for path in all_dirs:
    print("Now flowing " + path)
    command = './flow --imgdir ' + path + ' --model cfg/yolo.cfg --load -1 --gpu 0.9 --batch 16 --seq_length 1'
    command = command.split()
    cliHandler(command)