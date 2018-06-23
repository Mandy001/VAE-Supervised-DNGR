import math
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
import tensorflow as tf
import matplotlib.pyplot as plt

class DNGR(object):
    
    def __init__(self, graph, Kstep, dim, XY):
        self.alpha = 0.98
        self.g = graph
        self.Kstep = Kstep
        self.dim = dim
        self.train()

    # def getAdjMat(self):
    #     graph = self.g.G
    #     node_size = self.g.node_size
    #     look_up = self.g.look_up_dict
    #     adj = np.zeros((node_size, node_size))
    #     for edge in self.g.G.edges():
    #         adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
    #         adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
    #     # ScaleSimMat
    #     return np.matrix(adj / np.sum(adj, axis=1))

    def getAdj(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():       # 取一条边出来
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat
        return adj
    #
    # def GetProbTranMat(self, Ak):
    #     probTranMat = np.log(Ak / np.tile(
    #         np.sum(Ak, axis=0), (self.node_size, 1))) \
    #         - np.log(1.0 / self.node_size)
    #     probTranMat[probTranMat < 0] = 0
    #     probTranMat[probTranMat == np.nan] = 0
    #     return probTranMat
    #
    # def GetRepUseSVD(self, probTranMat, alpha):
    #     U, S, VT = la.svd(probTranMat)
    #     Ud = U[:, 0:self.dim]
    #     Sd = S[0:self.dim]
    #     return np.array(Ud) * np.power(Sd, alpha).reshape((self.dim))

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def scale_sim_mat(self, mat):
        # Scale Matrix by row
        mat = mat - np.diag(np.diag(mat))       # 将对角线元素置零
        D_inv = np.diag(np.reciprocal(np.sum(mat, axis=0)))     # 将每列之和取倒数放在对角线元素位置
        D_inv[np.isnan(D_inv)] = 0.0
        D_inv[np.isinf(D_inv)] = 0.0
        D_inv[np.isneginf(D_inv)] = 0.0     # 为了防止主对角线元素为inf或者nan
        mat = np.dot(D_inv, mat)        # 相当于每一行乘以一个与该行相连的节点的数目的倒数（当前行为第i行，表示第i个节点，假设i节点有m条边，那么该行每个数都乘以1/m）
        return mat      # 这样做事使得每一行的概率相加为1

    def PPMI_matrix(self, M):
        M = self.scale_sim_mat(M)
        nm_nodes = len(M)
        col_s = np.sum(M, axis=0).reshape(1, nm_nodes)
        row_s = np.sum(M, axis=1).reshape(nm_nodes, 1)
        D = np.sum(col_s)
        rowcol_s = np.dot(row_s, col_s)
        PPMI = np.log(np.divide(D * M, rowcol_s))
        PPMI[np.isnan(PPMI)] = 0.0
        PPMI[np.isinf(PPMI)] = 0.0
        PPMI[np.isneginf(PPMI)] = 0.0
        PPMI[PPMI < 0] = 0.0

        return PPMI

    def random_surfing(self, adj_matrix, max_step, alpha):
        # Random Surfing
        nm_nodes = len(adj_matrix)
        adj_matrix = self.scale_sim_mat(adj_matrix)
        P0 = np.eye(nm_nodes, dtype='float32')
        M = np.zeros((nm_nodes, nm_nodes), dtype='float32')
        P = np.eye(nm_nodes, dtype='float32')
        for i in range(0, max_step):        # 0,1,2,3 一共四步迭代
            P = alpha * np.dot(P, adj_matrix) + (1 - alpha) * P0
            M = M + P
        return M

    def train(self):
        self.adj = self.getAdj()
        self.surfing_matrix = self.random_surfing(self.adj, self.Kstep, self.alpha)
        PPMI = self.PPMI_matrix(self.surfing_matrix)
        print(PPMI)

        input_dim = PPMI.shape[1]  # PPMI.shape -> (2405, 2405)
        input_placeholder = tf.placeholder(tf.float32, PPMI.shape)

        #1 layer
        #hidden_dims = [self.dim]  # [258, 128]
        #2 layers
        hidden_dims = [self.dim * 2, self.dim]  # [258, 128]
        #3 layers
        # hidden_dims = [self.dim * 4, self.dim * 2, self.dim]
        #4 layers
        #hidden_dims = [self.dim * 8, self.dim * 4, self.dim * 2, self.dim]
        #5 layers
        # hidden_dims = [self.dim * 16, self.dim * 8, self.dim * 4, self.dim * 2, self.dim]
        encoder_dims = hidden_dims

        gaussian_noise = tf.truncated_normal(PPMI.shape, stddev=1.0)

        current_layer = input_placeholder + gaussian_noise
        for dim in encoder_dims:
            current_layer = tf.layers.dense(current_layer, dim, tf.nn.relu)
        last_encoder_layer = current_layer

        decoder_dims = list(reversed(hidden_dims))[1:] + [input_dim]
        for i, dim in enumerate(decoder_dims):
            if i == len(decoder_dims) - 1:
                activation = None
            else:
                activation = tf.nn.relu
            current_layer = tf.layers.dense(current_layer, dim, activation)
        output_layer = current_layer

        loss = tf.square(input_placeholder - output_layer)
        mean_loss = tf.reduce_mean(loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=2e-3).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # for i in range(5):
            #     print(sess.run(gaussian_noise))
            # asdfadsf
            feed_dict = {
                input_placeholder: PPMI
            }
            self.losses = []
            for i in range(400):
                sess.run(optimizer, feed_dict=feed_dict)
                if i % 10 == 0:
                    loss_val = sess.run(mean_loss, feed_dict=feed_dict)
                    self.losses.append(loss_val)
                    print("step = {}\tloss = {}".format(i, loss_val))
            embeddings = sess.run(last_encoder_layer, feed_dict=feed_dict)

            self.vectors = {}
            look_back = self.g.look_back_list
            for i, embedding in enumerate(embeddings):
                self.vectors[look_back[i]] = embedding
            return



    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.plot([ii*10 for ii in list(range(len(self.losses)))], self.losses)
        plt.show()



