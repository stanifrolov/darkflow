from darkflow.cli import cliHandler
from motc_utils import img2video

with open("img_paths.txt", encoding="utf-8") as file:
  my_list = file.readlines()
  all_dirs = [x.strip() for x in my_list]

  for path in all_dirs:
    print("Now flowing " + path)
    try:
      command = './flow --imgdir ' + path + ' --model cfg/full_motnet.cfg --load -1 --batch 1 --seq_length 12 --threshold 0.5'
      command = command.split()
      cliHandler(command)
    except:
      print("Error or Done")

  response = input("All images flowed. Do you want to create videos? (y/n): ")

  if response is "y":
    for path in all_dirs:
      print("Creating video for " + path)
      img2video.img_to_video(path)
  file.close()
