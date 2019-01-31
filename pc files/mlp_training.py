import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print 'Loading collected data...'
t1 = cv2.getTickCount()

# load collected data
img_arr = np.zeros((1, 38400))
label_arr = np.zeros((1, 4), 'float')
collected_data = glob.glob('collection_data/*.npz')

# exit if not found
if not collected_data:
    print "Data not found, exiting..."
    sys.exit()

for single_npz in collected_data:
    with np.load(single_npz) as data:
        temp = data['train']
        labels_temp = data['train_labels']
    img_arr = np.vstack((img_arr, temp))
    label_arr = np.vstack((label_arr, labels_temp))

x = img_arr[1:, :]
y = label_arr[1:, :]
print 'Image array shape: ', x.shape
print 'Label array shape: ', y.shape

t2 = cv2.getTickCount()
time0 = (t2 - t1)/ cv2.getTickFrequency()
print 'Images loaded in {0:.2f} secs'.format(time0)

# splitting data into 80% training data and 20% test data
train, test, train_labels, test_labels = train_test_split(x, y, test_size=0.2)

t3 = cv2.getTickCount()

# create MLP
layers = np.int32([38400, 40, 4])
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layers)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setBackpropMomentumScale(0.0)
model.setBackpropWeightScale(0.001)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 500, 0.0001))
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

print 'Training MLP ...'
n_itr = model.train(np.float32(train), cv2.ml.ROW_SAMPLE, np.float32(train_labels))

t4 = cv2.getTickCount()
time1 = (t4 - t3)/cv2.getTickFrequency()
print 'Training completed in {0:.2f} secs'.format(time1)

# training data
ret_0, resp_0 = model.predict(train)
prediction_0 = resp_0.argmax(-1)
true_labels_0 = train_labels.argmax(-1)

train_rate = np.mean(prediction_0 == true_labels_0)
print 'Train accuracy: ', "{0:.2f}%".format(train_rate * 100)

# testing data
ret_1, resp_1 = model.predict(test)
prediction_1 = resp_1.argmax(-1)
true_labels_1 = test_labels.argmax(-1)

test_rate = np.mean(prediction_1 == true_labels_1)
print 'Test accuracy: ', "{0:.2f}%".format(test_rate * 100)

# save the multilayer perceptron model
model.save('mlp_trained/trained_network.xml')
