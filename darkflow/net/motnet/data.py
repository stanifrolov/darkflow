import os
import glob
from copy import deepcopy

import numpy as np
from numpy.random import permutation as perm
import random as random

from darkflow.utils.motc_clean_as_list import motc_clean_as_list


def parse(self, exclusive=False):
  meta = self.meta
  ext = '.parsed'
  ann = self.FLAGS.annotation
  if not os.path.isdir(ann):
    msg = 'Annotation directory not found {} .'
    exit('Error: {}'.format(msg.format(ann)))
  print('\n{} parsing {}'.format(meta['model'], ann))
  dumps = motc_clean_as_list(ann, exclusive)
  return dumps


def _batch(self, chunk):
  """
  Takes a chunk of parsed annotations
  returns value for placeholders of net's
  input & loss layer correspond to this chunk
  """
  meta = self.meta
  labels = meta['labels']

  H, W, _ = meta['out_size']
  C, B = meta['classes'], meta['num']
  anchors = meta['anchors']

  # preprocess
  jpg = chunk[0]
  w, h, allobj_ = chunk[1]
  allobj = deepcopy(allobj_)
  path = os.path.join(self.FLAGS.dataset, jpg)
  img = self.preprocess(path, allobj) # set allobj to None to disable noise augm for overfit

  # Calculate regression target
  cellx = 1. * w / W
  celly = 1. * h / H
  for obj in allobj:
    centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
    centery = .5 * (obj[2] + obj[4])  # ymin, ymax
    cx = centerx / cellx
    cy = centery / celly

    # if cx >= W or cy >= H: return None, None
    if cx >= W:
      cx = W - 0.01
    if cy >= H:
      cy = H - 0.01

    obj[3] = float(obj[3] - obj[1]) / w
    obj[4] = float(obj[4] - obj[2]) / h
    obj[3] = np.sqrt(obj[3])
    obj[4] = np.sqrt(obj[4])
    obj[1] = cx - np.floor(cx)  # centerx
    obj[2] = cy - np.floor(cy)  # centery
    obj += [int(np.floor(cy) * W + np.floor(cx))]

  # show(im, allobj, S, w, h, cellx, celly) # unit test

  # Calculate placeholders' values
  probs = np.zeros([H * W, B, C])
  confs = np.zeros([H * W, B])
  coord = np.zeros([H * W, B, 4])
  proid = np.zeros([H * W, B, C])
  prear = np.zeros([H * W, 4])
  for obj in allobj:
    probs[obj[5], :, :] = [[0.] * C] * B
    probs[obj[5], :, labels.index(obj[0])] = 1.
    proid[obj[5], :, :] = [[1.] * C] * B
    coord[obj[5], :, :] = [obj[1:5]] * B
    prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * W  # xleft
    prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * H  # yup
    prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * W  # xright
    prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * H  # ybot
    confs[obj[5], :] = [1.] * B

  # Finalise the placeholders' values
  upleft = np.expand_dims(prear[:, 0:2], 1)
  botright = np.expand_dims(prear[:, 2:4], 1)
  wh = botright - upleft
  area = wh[:, :, 0] * wh[:, :, 1]
  upleft = np.concatenate([upleft] * B, 1)
  botright = np.concatenate([botright] * B, 1)
  areas = np.concatenate([area] * B, 1)

  # value for placeholder at input layer
  inp_feed_val = img
  # value for placeholder at loss layer
  loss_feed_val = {
    'probs': probs, 'confs': confs,
    'coord': coord, 'proid': proid,
    'areas': areas, 'upleft': upleft,
    'botright': botright
  }

  return inp_feed_val, loss_feed_val


def shuffle(self):
  batch_size = self.FLAGS.batch
  data = self.parse()
  size = len(data)

  print('Dataset of {} instance(s)'.format(size))
  if batch_size > size: self.FLAGS.batch = batch_size = size
  batch_per_epoch = int(size / batch_size)
  seq_length = self.FLAGS.seq_length

  for epoch in range(self.FLAGS.epoch):
    shuffle_idx = perm(np.arange(size))
    for batch in range(batch_per_epoch):
      x_batch = list()
      feed_batch = dict()

      for step in range(batch * batch_size, batch * batch_size + batch_size):
        start_img = data[shuffle_idx[step]]
        #start_img = data[0]
        start_img = fit_to_seq_length(self.FLAGS.dataset, data, start_img, seq_length)
        idx_of_start = data.index(start_img)
        #print("start_img is " + start_img[0])
        for seq in range(seq_length):
          train_instance = data[idx_of_start + seq]
          print("train_img is " + train_instance[0])

          inp, new_feed = self._batch(train_instance)
          if inp is None:
            print("WARNING: Input is None - Continue Feed Loop")
            continue
          x_batch += [np.expand_dims(inp, 0)]

          for key in new_feed:
            new = new_feed[key]
            old_feed = feed_batch.get(key,
                                      np.zeros((0,) + new.shape))
            feed_batch[key] = np.concatenate([
              old_feed, [new]
            ])

      x_batch = np.concatenate(x_batch, 0)
      yield x_batch, feed_batch

    print('Finish {} epoch(es)'.format(epoch + 1))


def fit_to_seq_length(path_to_dataset, data, start_img, seq_length):
    path_to_img = path_to_dataset + start_img[0].split("/")[0] + "/img1"
    num_of_img_in_set = get_num_of_img_in_set(path_to_img)
    num_of_start_img = int(start_img[0].split("/")[2].split(".")[0])
    if num_of_start_img <= (num_of_img_in_set - seq_length + 1):
      return start_img
    else:
      new_start_img = data[data.index(start_img) - seq_length + 1]
      print("INFO: Fit to sequence - wanted " + start_img[0] + " - but now take " + new_start_img[0])
      return new_start_img


def get_num_of_img_in_set(path_to_img):
  return len(glob.glob1(path_to_img, "*.jpg"))