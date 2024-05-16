from variograd_utils import *
import os

data = dataset()

print("\n\nGenerating average group surface.")
data.generate_avg_surf("L", 10)
data.generate_avg_surf("R", 10)
data.__init__()
