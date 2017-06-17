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
  dumps = list()
  start_directory = os.getcwd()

  if os.path.isfile("motc_dump.p"):
    return pickle.load(open('motc_dump.p', 'rb'))

  os.chdir(ANN)
  annotations = glob.glob('*FRCNN')
  size = len(annotations)

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

    all_images = glob.glob('img1/*')
    curr_img_number = 1
    for image in all_images:
      all_obj_in_image = list()
      gt_file = open("gt/gt.txt")
      for line in gt_file:
        splitted_line = line.split(',')
        frame_number = int(splitted_line[0])
        if frame_number == curr_img_number:
          # id = splitted_line[1]
          id = "object" # motc dataset only has ids, not classes
          bb_left = int(splitted_line[2])
          bb_top = int(splitted_line[3])
          bb_width = int(splitted_line[4])
          bb_height = int(splitted_line[5])
          x_min, y_min, x_max, y_max = motc_gt_to_voc_gt(bb_left, bb_top, bb_width, bb_height)
          if x_min > 0 and y_min > 0 and x_max > 0 and y_max > 0:
            all_obj_in_image.append([id, x_min, y_min, x_max, y_max])
      gt_file.close()
      curr_img_object = [[folder + "/" + image, [img_width, img_height, all_obj_in_image]]]
      dumps += curr_img_object
      curr_img_number += 1
    os.chdir("..")

  # gather all stats
  stat = dict()
  for dump in dumps:
    all_obj_in_image = dump[1][2]
    for current in all_obj_in_image:
      if current[0] in stat:
        stat[current[0]] += 1
      else:
        stat[current[0]] = 1

  print('\nStatistics:')
  _pp(stat)
  print('Dataset size: {}'.format(len(dumps)))

  os.chdir(start_directory)

  pickle.dump(dumps, open('motc_dump.p', 'wb'))
  return dumps


def motc_gt_to_voc_gt(bb_left, bb_top, bb_width, bb_height):
  x_min = bb_left
  y_min = bb_top
  x_max = bb_left + bb_width
  y_max = bb_top + bb_height
  return x_min, y_min, x_max, y_max