import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

#dump = pickle.load(open('voc_dump.p', 'rb'))
dump = pickle.load(open('motc_dump_resized.p', 'rb'))
fig, ax = plt.subplots(1)

for entry in dump:
  #path = "/Users/sfrolov/master-thesis/code/darkflow/VOCdevkit/VOC2007/JPEGImages/" + entry[0]
  path = "/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/" + entry[0]
  boxes = [[var for var in group if not str(var) == var] for group in entry[1][2]]
  im = np.array(Image.open(path), dtype=np.uint8)

  # Create figure and axes
  plt.cla()
  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  for box in boxes:
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

  #plt.show()
  fig.savefig("/Users/sfrolov/master-thesis/code/darkflow/MOTC/MOT17/train/gt_eval/" + entry[0])