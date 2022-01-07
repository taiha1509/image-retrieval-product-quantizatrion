import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.spatial.distance import cdist

# chia tập các vetor, phân cụm và trả về các codeword
def train(vec, M, Ks=256):
    '''
    :param M: số lượng sub-vectors của từng vector
    :param Ks: số cluster áp dụng trên từng tập sub-vectors
    '''
    Ds = int(vec.shape[1] / M)  # số chiều 1 sub-vector
    # tạo M codebooks
    # tạo mảng chứa toàn bộ codeword
    # mỗi codebook gồm Ks codewords
    codeword = np.empty((M, Ks, Ds), np.float32)
    for m in range(M):
        # mảng chứa tập các sub_vector cột thứ m
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # thực hiện k-means trên từng tập sub-vector thứ m
        centroids, labels = kmeans2(vec_sub, Ks)
        # centroids: (Ks x Ds)
        # labels: vec.shape[0]
        codeword[m] = centroids
    return codeword

def encode(codeword, vec):
    # M sub-vectors
    # Ks clusters ứng với từng tập sub-vector 
    # Ds: số chiều 1 sub-vector
    M, Ks, Ds = codeword.shape

    # tạo pq-code cho n samples (với n = vec.shape[0])
    # mỗi pq-code gồm M giá trị
    pqcode = np.empty((vec.shape[0], M), np.uint8)

    for m in range(M):
        # tập các sub_vectors
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # codes: 1 mảng gồm n phần tử (n = vec.shape[0]), lưu giữ cluster index gần nhất của sub-vector thứ m của từng vector
        # distances: 1 mảng gồm n phần từ (n = vec.shape[0]), lưu giữ khoảng cách giữa sub-vector thứ m của từng vector với centroid gần nhất
        codes, distances = vq(vec_sub, codeword[m])
        # codes: vec.shape[0]
        # distances: vec.shape[0]
        pqcode[:, m] = codes

    return pqcode
vec = np.random.uniform(1, 20, [10,10])
M=2
Ks = 5
cw = train(vec, M, Ks)
rs = encode(cw, vec)
print(rs)