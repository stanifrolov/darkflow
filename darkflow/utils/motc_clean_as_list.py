"""
parse PASCAL VOC xml annotations
"""

import configparser
import glob
import os
import pickle
import sys


def _pp(l):  # pretty printing
  for i in l: print('{}: {}'.format(i, l[i]))


def motc_clean_as_list(ANN, exclusive=False):
  start_directory = os.getcwd()

  if os.path.isfile("motc_dump.p"):
    return pickle.load(open('motc_dump.p', 'rb'))

  os.chdir(ANN)
  annotations = glob.glob('*FRCNN')
  size = len(annotations)
  dumps = {}
  for i, folder in enumerate(annotations):
    # progress bar
    sys.stdout.write('\r')
    percentage = 1. * (i + 1) / size
    progress = int(percentage * 20)
    bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
    bar_arg += [folder]
    sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
    sys.stdout.flush()

    # actual parsing
    os.chdir(folder)

    config = configparser.ConfigParser()
    seq_info = "seqinfo.ini"
    config.read(seq_info)
    seq_info = config['Sequence']
    img_width = int(seq_info['imWidth'])
    img_height = int(seq_info['imHeight'])

    gt_file = open("gt/gt.txt")
    for line in gt_file:
      splitted_line = line.split(',')
      x_min, y_min, x_max, y_max = motc_gt_to_voc_gt(splitted_line)
      frame_number = int(splitted_line[0])
      if x_min > 0 and y_min > 0 and x_max > 0 and y_max > 0:
        key = str(folder + "/img1/" + '{:06d}'.format(frame_number) + ".jpg")
        if key not in dumps:
          dumps[key] = [img_width, img_height, [["object", x_min, y_min, x_max, y_max]]]
        else:
          dumps[key][2].append(["object", x_min, y_min, x_max, y_max])

    os.chdir("..")
    gt_file.close()

  dumplist = []
  for key, value in dumps.items():
    dumplist.append([key, value])

  # gather all stats
  stat = dict()
  for dump in dumplist:
    all_obj_in_image = dump[1][2]
    for current in all_obj_in_image:
      if current[0] in stat:
        stat[current[0]] += 1
      else:
        stat[current[0]] = 1

  print('\nStatistics:')
  _pp(stat)
  print('Dataset size: {}'.format(len(dumplist)))

  os.chdir(start_directory)

  pickle.dump(dumplist, open('motc_dump.p', 'wb'))
  return dumplist


def motc_gt_to_voc_gt(splitted_line):
  bb_left = int(splitted_line[2])
  bb_top = int(splitted_line[3])
  bb_width = int(splitted_line[4])
  bb_height = int(splitted_line[5])
  x_min = bb_left
  y_min = bb_top
  x_max = bb_left + bb_width
  y_max = bb_top + bb_height
  return x_min, y_min, x_max, y_max