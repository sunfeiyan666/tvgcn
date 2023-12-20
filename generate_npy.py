# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import numpy as np

import pandas as pd



if __name__ == "__main__":
	data = pd.read_csv("./toy.csv")
	np.savez("./toy", data)
