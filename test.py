import csv
from numpy import array, empty, float32
with open('./hsv.csv', newline='') as csvfile:
    data = array(list(csv.reader(csvfile)))
test = empty((data.shape[0], data.shape[1] - 1), float32)
for idx, temp in enumerate(data):
    test[idx] = temp[1: 1441]
# temp = data[1:11][1:2]
print(test[:test.shape[0]-1].shape)