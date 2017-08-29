"""
tfnet secondary (helper) methods
"""
import os
import sys
from time import time as timer

import cv2
import tensorflow as tf

from darkflow.utils.loader import create_loader

old_graph_msg = 'Resolving old graph def {} (no guarantee)'


def build_train_op(self):
  global_step = tf.Variable(0, trainable=False, name="global_step_new")
  #self.framework.loss(self.top.inp.out) # last layer before recurrent
  self.framework.loss(self.out)
  self.say('Building {} train op'.format(self.meta['model']))
  optimizer = self._TRAINER[self.FLAGS.trainer](learning_rate(global_step, self))
  #var_list = [var for var in tf.trainable_variables() if "recurrent" in var.name] # only recurrent layer trainable
  #self.train_op = optimizer.minimize(self.framework.loss, global_step=global_step, var_list=var_list) # only recurrent layer trainable
  self.train_op = optimizer.minimize(self.framework.loss, global_step=global_step)
  #gradients = optimizer.compute_gradients(self.framework.loss)
  #gradients = [(tf.clip_by_global_norm(gradients, 5.0), var) for grad, var in gradients]
  #self.train_op = optimizer.apply_gradients(gradients)


def learning_rate(global_step, self):
  self.FLAGS.learningRate = tf.train.exponential_decay(
    self.FLAGS.lr, # Base learning rate.
    global_step,
    100, # Decay step.
    0.9, # Decay rate.
    staircase=True)
  return self.FLAGS.learningRate


def load_from_ckpt(self):
  if self.FLAGS.load < 0:  # load lastest ckpt
    with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
      last = f.readlines()[-1].strip()
      load_point = last.split(' ')[1]
      load_point = load_point.split('"')[1]
      load_point = load_point.split('-')[-1]
      self.FLAGS.load = int(load_point)

  load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
  load_point = '{}-{}'.format(load_point, self.FLAGS.load)
  self.say('Loading from {}'.format(load_point))
  try:
    self.saver.restore(self.sess, load_point)
  except:
    #load_old_graph(self, load_point)
    pass

def say(self, *msgs):
  if not self.FLAGS.verbalise:
    return
  msgs = list(msgs)
  for msg in msgs:
    if msg is None: continue
    print(msg)


def load_old_graph(self, ckpt):
  ckpt_loader = create_loader(ckpt)
  self.say(old_graph_msg.format(ckpt))

  for var in tf.global_variables():
    name = var.name.split(':')[0]
    args = [name, var.get_shape()]
    val = ckpt_loader(args)
    assert val is not None, \
      'Cannot find and load {}'.format(var.name)
    shp = val.shape
    plh = tf.placeholder(tf.float32, shp)
    op = tf.assign(var, plh)
    self.sess.run(op, {plh: val})


def _get_fps(self, frame):
  elapsed = int()
  start = timer()
  preprocessed = self.framework.preprocess(frame)
  feed_dict = {self.inp: [preprocessed]}
  net_out = self.sess.run(self.out, feed_dict)[0]
  processed = self.framework.postprocess(net_out, frame, False)
  return timer() - start


def camera(self):
  file = self.FLAGS.demo
  SaveVideo = self.FLAGS.saveVideo

  if file == 'camera':
    file = 0
  else:
    assert os.path.isfile(file), \
      'file {} does not exist'.format(file)

  camera = cv2.VideoCapture(file)
  assert camera.isOpened(), \
    'Cannot capture source'

  assert camera is not None, \
    'Cannot capture video - object is None'

  cv2.namedWindow('', 0)
  _, frame = camera.read()
  while frame is None:
    _, frame = camera.read()

  assert frame is not None, \
    'Cannot read camera - frame is None'

  assert frame.shape is not None, \
    'Cannot read camera - frame is None'

  height, width, _ = frame.shape
  cv2.resizeWindow('', width, height)

  if SaveVideo:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if file == 0:
      fps = 1 / _get_fps(self, frame)
      if fps < 1:
        fps = 1
    else:
      fps = round(camera.get(cv2.CAP_PROP_FPS))
    videoWriter = cv2.VideoWriter(
      'video.mp4v', fourcc, fps, (width, height))

  # buffers for demo in batch
  buffer_inp = list()
  buffer_pre = list()

  elapsed = int()
  start = timer()
  self.say('Press [ESC] to quit demo')
  # Loop through frames
  while camera.isOpened():
    elapsed += 1
    _, frame = camera.read()
    if frame is None:
      print('\nEnd of Video')
      break
    preprocessed = self.framework.preprocess(frame)
    buffer_inp.append(frame)
    buffer_pre.append(preprocessed)

    # Only process and imshow when queue is full
    if elapsed % self.FLAGS.queue == 0:
      feed_dict = {self.inp: buffer_pre}
      net_out = self.sess.run(self.out, feed_dict)
      for img, single_out in zip(buffer_inp, net_out):
        postprocessed = self.framework.postprocess(
          single_out, img, False)
        if SaveVideo:
          videoWriter.write(postprocessed)
        cv2.imshow('', postprocessed)
      # Clear Buffers
      buffer_inp = list()
      buffer_pre = list()

    if elapsed % 5 == 0:
      sys.stdout.write('\r')
      sys.stdout.write('{0:3.3f} FPS'.format(
        elapsed / (timer() - start)))
      sys.stdout.flush()
    choice = cv2.waitKey(1)
    if choice == 27: break

  sys.stdout.write('\n')
  if SaveVideo:
    videoWriter.release()
  camera.release()
  cv2.destroyAllWindows()


def to_darknet(self):
  darknet_ckpt = self.darknet

  with self.graph.as_default() as g:
    for var in tf.global_variables():
      name = var.name.split(':')[0]
      var_name = name.split('-')
      l_idx = int(var_name[0])
      w_sig = var_name[1].split('/')[-1]
      l = darknet_ckpt.layers[l_idx]
      l.w[w_sig] = var.eval(self.sess)

  for layer in darknet_ckpt.layers:
    for ph in layer.h:
      layer.h[ph] = None

  return darknet_ckpt
