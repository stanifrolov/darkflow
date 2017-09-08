import os
import atexit
import subprocess

def get_dataset_name_from_path(path):
  if 'MOT17' in path:
    return path.split("/")[6] + "-" + path.split("/")[7].split("-")[1]
  else:
    return path.split("/")[4]


def img_to_video(path):
    print("Creating video")
    os.chdir(path + "/out")
    command = "ffmpeg -y -framerate 25 -pattern_type glob -i '*.jpg' /home/frolov/U/" + get_dataset_name_from_path(path) + ".mp4"
    pro  = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    atexit.register(pro.terminate)