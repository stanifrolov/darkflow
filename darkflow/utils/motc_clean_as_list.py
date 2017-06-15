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
    w = seq_info['imWidth']
    h = seq_info['imHeight']

    all_images = glob.glob('img1/*')
    curr_img_number = 1
    for image in all_images:
      # TODO: build all from gt.txt
      all_obj_in_image = list()
      gt_file = open("gt/gt.txt")
      for line in gt_file:
        splitted_line = line.split(',')
        frame = splitted_line[0]
        if int(frame) == curr_img_number:
          id = splitted_line[1]
          bb_left = splitted_line[2]
          bb_top = splitted_line[3]
          bb_width = splitted_line[4]
          bb_height = splitted_line[5]
          all_obj_in_image.append([1, bb_left, bb_top, bb_width, bb_height])  # id 1 for just detected object
      gt_file.close()
      curr_img_object = [[folder + "/" + image, [w, h, all_obj_in_image]]]
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
