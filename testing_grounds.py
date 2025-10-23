import skimage as ski
from natsort import natsorted, ns
import os
import numpy as np

goldfinches = os.listdir('data/Bird_Species_Dataset/AMERICAN_GOLDFINCH')
print(goldfinches)