import numpy as np
from numpy.linalg import pinv
from scipy.io import loadmat
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')


# ======================= Parameters ===========================
N = 4000
digit1 = 0
digit2 = 7

# ==================== Load MNIST dataset ======================
dataset = loadmat('mnist.mat')

training = dataset['training'][0][0]
training_count = training[0][0][0]
training_height = training[1][0][0]
training_width = training[2][0][0]
training_images = training[3]
training_labels = training[4]
training_labels = np.squeeze(training_labels)

# new test set for clause 3
test = dataset['test'][0][0]
test_count = test[0][0][0]
test_height = test[1][0][0]
test_width = test[2][0][0]
test_images = test[3]
test_labels = test[4]
test_labels = np.squeeze(test_labels)
A_new_test = np.reshape(test_images, (test_count, 28**2))
A_new_test = np.hstack((A_new_test, np.ones((test_count, 1))))

# ------- Little bit of exploration to feel the data -------------
if True:
    print(training_images.shape)
    plt.figure(0)
    plt.imshow(training_images[:, :, 10], cmap='gray')
    plt.title('This image label is ' + str(training_labels[10]))
    plt.show(block=False)
# ----------------------------------------------------------------

images_per_digit_1 = training_images[:, :, training_labels == digit1]
images_per_digit_2 = training_images[:, :, training_labels == digit2]

if True:
    plt.figure(1)
    for k in range(5):
        plt.imshow(images_per_digit_1[:, :, k], cmap='gray')
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure(1)
    for k in range(5):
        plt.imshow(images_per_digit_2[:, :, k], cmap='gray')
        plt.axis('image')
        plt.axis('off')
        plt.pause(0.1)
        plt.show(block=False)

# ======================= Create A, b ============================
A_all = np.zeros((2*N, 28*28 + 1))
b_all = np.zeros((2*N, 1))
for i in range(N):
    A_all[2*i-1, :-1] = np.reshape(images_per_digit_1[:, :, i], (1,28*28))
    A_all[2*i,   :-1] = np.reshape(images_per_digit_2[:, :, i], (1,28*28))
    b_all[2*i-1]      = +1
    b_all[2*i]        = -1
A_all[:, -1] = 1

# ========================= Solve LS ==============================
A_train = A_all[:N, :]
b_train = b_all[:N]

x = pinv(A_train) @ b_train

# ===================== Prepare Test Set ==========================
A_test = A_all[N:2*N, :]
b_test = b_all[N:2*N]

# ===================== Check Performance ===========================
predC = np.sign(A_train @ x)
trueC = b_train
print('Train Error:')
acc = np.mean(predC == trueC) * 100
print('Accuracy={:.2f}% ({:.0f} wrong examples)'.format(acc, (1-acc/100)*N))

predC = np.sign(A_test @ x)
trueC = b_test
print('Train Error:')
acc = np.mean(predC == trueC) * 100
print('Accuracy={:.2f}% ({:.0f} wrong examples)'.format(acc, (1-acc/100)*N))

# ================= Show the Problematic Images =====================
error = predC != trueC
error = np.squeeze(error)
problematic = A_test[error, :]
if True:
    plt.figure(2)
    for k in range(len(problematic)):
        plt.imshow(np.reshape(problematic[k, :-1], (28, 28)), cmap='gray')
        product = problematic[k] @ x
        product = np.squeeze(product)
        plt.title('problematic digit number {} :{:+.4f}'.format(k, product))
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        # plt.waitforbuttonpress()
