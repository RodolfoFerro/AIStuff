# ========================================================================
# Author: Rodolfo Ferro
# Contact: https://rodolfoferroxyz/
#
# Title: Intro to Machine Learning (ML Zero to Hero, part 4)
# From: https://www.youtube.com/watch?v=u2TjZzNuly8
# ========================================================================


import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import zipfile


# We get the rock_paper_scissors dataset:
train_dataset_cmd = """
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O /tmp/rps.zip
"""

test_dataset_cmd = """
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O /tmp/rps-test-set.zip
"""

os.system(train_dataset_cmd)
os.system(test_dataset_cmd)

# We unzip the files:
local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

# We can verify the length of the unzipped images:
rock_dir = os.path.join('/tmp/rps/rock')
paper_dir = os.path.join('/tmp/rps/paper')
scissors_dir = os.path.join('/tmp/rps/scissors')

print('Total training rock images:', len(os.listdir(rock_dir)))
print('Total training paper images:', len(os.listdir(paper_dir)))
print('Total training scissors images:', len(os.listdir(scissors_dir)))

# TODO. Finish the notebook.
