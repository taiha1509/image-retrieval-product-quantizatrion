import csv
from image_query import train, search, encode
from numpy import array, empty, float32
import numpy as np
import time

def read_image():
    with open('./hsv_10k.csv', newline='') as csvfile:
        csv_data = array(list(csv.reader(csvfile)))
    data = empty((csv_data.shape[0], csv_data.shape[1] - 1), float32)
    print(data.shape)
    image_name = array([], 'str')
    for idx, temp in enumerate(csv_data):
        data[idx] = temp[1: csv_data.shape[1]]
        image_name = np.append(image_name, temp[0])
    image_name = np.delete(image_name, image_name.shape[0] - 1)
    return data, image_name


if __name__ == '__main__':
    # N tổng số lương ảnh
    # Nt số lượng ảnh để phân cụm
    # D số chiều của 1 vector
    data, image_name = read_image()
    N = data.shape[0] - 1
    Nt = data.shape[0] - 1
    D = data.shape[1] 
    vec = data[1:]
    vec_train = vec
    # hàng cuối cùng
    query = data[N: N+1][0]  # a 128-dim query vector
    # số lương các sub vector được chia
    M = 12
    codeword = train(vec_train, M)
    # short-code (pq-code) của toàn bộ tập dữ liệu, mỗi phần tử gồm 8 phần tử đại diện cho subvector tương ứng thuộc cụm nào
    pqcode = encode(codeword, vec)
    start = time.time() * 1000
    dist = search(codeword, pqcode, query)
    end = time.time() * 1000
    dist_with_name = empty((N, 2), dtype='object')
    for i in range(0, N, 1):
        dist_with_name[i][0] = image_name[i]
        dist_with_name[i][1] = dist[i]
    list_ids = array(dist_with_name[:, 1].argsort())
    for index, id_ in enumerate(list_ids):
        # get 10 images that most like the query image
        if(index < 10):
            print("Image name: {} -> Dist: {}".format(dist_with_name[id_][0], dist_with_name[id_][1]))
    print('searching time: ', round(end) - round(start))