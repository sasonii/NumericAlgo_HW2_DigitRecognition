import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')


# ====================== Prepare New Test Set ======================
dataset = loadmat('mnist.mat')

test = dataset['test'][0][0]
test_count = test[0][0][0]
test_height = test[1][0][0]
test_width = test[2][0][0]
test_images = test[3]
test_labels = test[4]

num_images = test_count
new_test_images = np.transpose(test_images, (2, 0, 1))
A_new_test = np.reshape(new_test_images, (num_images, 28 ** 2))
A_new_test = np.hstack((A_new_test, np.ones((num_images, 1))))
true_labels = test_labels

# ============================ Predict ==============================
UNCLASSIFIED = -1
pred = UNCLASSIFIED * np.ones((num_images, 1))

# TODO: compute your predictions

# =========================== Evaluate ==============================
acc = np.mean(pred == true_labels) * 100
print('Accuracy={:.2f}% ({:.0f} wrong examples)'.format(acc, (1-acc/100) * num_images))
