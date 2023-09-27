#encoding:utf-8

import numpy as np
import csv
import os
# os.add_dll_directory("${D:/Anaconda3/envs/pytorch/lib/site-packages/scipy/.libs/libbanded5x.7J4WS2QZKMXGIZDNNWWXUXE52PU2TOEI.gfortran-win_amd64.dll}")
import scipy
import xlrd
import torch as th
import dgl
import pickle
import pandas as pd
from scipy.io import loadmat

#EC_construct using matlab te_matlab_0.4    E:\project\te_matlab_0.4


def readxlsmatrix(file):
    table = xlrd.open_workbook(file).sheets()[0]
    row = table.nrows
    col = table.ncols

    matrix = np.zeros([row, col], np.float32)
    for i in range(row):
        for j in range(col):
            matrix[i][j] = table.cell(i, j).value

    return matrix

def mat_to_one_hot(bold_one_hot_path_root, norm_bold_path_root):
    filename = os.listdir(norm_bold_path_root)

    for name in filename:
        file_path = norm_bold_path_root + name
        # print("file_path", file_path)

        original_norm_bold_matrix = np.loadtxt(file_path, delimiter=",")


        """last 137"""
        norm_bold_matrix = original_norm_bold_matrix[:, -137:]

        """norm to one-hot"""
        mean = np.mean(np.array(norm_bold_matrix), axis=1)
        print("mean", np.shape(mean))
        print(mean)

        for i in range(len(norm_bold_matrix)):
            for j in range(len(norm_bold_matrix[0])):
                if norm_bold_matrix[i][j] >= mean[i]:
                    norm_bold_matrix[i][j] = 1
                else:
                    norm_bold_matrix[i][j] = 0

        series_name = name.replace('norm','one_hot')
        # print("series_name", series_name)
        bold_one_hot_path = bold_one_hot_path_root + series_name


        np.savetxt(bold_one_hot_path, norm_bold_matrix, fmt="%f", delimiter=",")



def EC_single_direction(dataset_root):

    filename_list = os.listdir(dataset_root + "EC/")
    for filename in filename_list:
        file_path = dataset_root + "EC/" + filename

        EC_matrix = readxlsmatrix(file_path)

        for row in range(len(EC_matrix)):
            for col in range(row, len(EC_matrix)):
                if EC_matrix[row][col] > EC_matrix[col][row]:
                    EC_matrix[col][row] = 0
                else:
                    EC_matrix[row][col] = 0

        if not os.path.exists(dataset_root + "EC_single_direction/"):
            os.makedirs(dataset_root + "EC_single_direction/")
        save_path = dataset_root + "EC_single_direction/" + filename
        save_path = save_path.replace("xls", "csv")
        np.savetxt(save_path, EC_matrix, delimiter=",", fmt="%f")




def heterogeneous_graph_construct(label_path, dataset_root, sample_num, node_num, pklname):

    label_matrix = np.loadtxt(label_path, delimiter=",")

    dgl_list = []

    for i in range(sample_num):
        FC_path = dataset_root + "FC/FC_adj_matrix_sample_" + str(i+1) + ".csv"
        FC_matrix = np.loadtxt(FC_path, delimiter=",")
        EC_path = dataset_root + "EC_single_direction/EC_adj_matrix_sample_" + str(i+1) + ".csv"
        EC_matrix = np.loadtxt(EC_path, delimiter=",")

        graph_data = {}

        # different types of nodes and edges have distinct ID

        FC_edge_s_list = []
        FC_edge_e_list = []
        EC_edge_s_list = []
        EC_edge_e_list = []
        Corr_edge_FE_s_list = [x for x in range(node_num)]
        Corr_edge_FE_e_list = [x for x in range(node_num)]
        Corr_edge_EF_s_list = [x for x in range(node_num)]
        Corr_edge_EF_e_list = [x for x in range(node_num)]



        for row in range(node_num):
            for col in range(node_num):
                if FC_matrix[row][col] == 1:
                    FC_edge_s_list.append(row)
                    FC_edge_e_list.append(col)
                else:
                    continue
                if EC_matrix[row][col] != 0:
                    EC_edge_s_list.append(row)
                    EC_edge_e_list.append(col)
                else:
                    continue



        graph_data[('FC_region', 'FC', 'FC_region')] = (th.tensor(FC_edge_s_list)), (th.tensor(FC_edge_e_list))
        graph_data[('EC_region', 'EC', 'EC_region')] = (th.tensor(EC_edge_s_list)), (th.tensor(EC_edge_e_list))
        graph_data[('FC_region', 'Corr_FE', 'EC_region')] = (th.tensor(Corr_edge_FE_s_list)), (th.tensor(Corr_edge_FE_e_list))
        graph_data[('EC_region', 'Corr_EF', 'FC_region')] = (th.tensor(Corr_edge_EF_s_list)), (th.tensor(Corr_edge_EF_e_list))

        hg = dgl.heterograph(graph_data)

        print("hg")
        print(hg)

        hg.nodes['FC_region'].data['feat'] = th.tensor(FC_matrix)
        hg.nodes['EC_region'].data['feat'] = th.tensor(EC_matrix)



        """label"""   #norm to 0,1
        label = label_matrix[i]
        label_tensor = th.tensor(label)

        signle_graph_tuple = (hg, label_tensor)

        dgl_list.append(signle_graph_tuple)

    dgl_list_pkl_path = dataset_root + pklname
    with open(dgl_list_pkl_path, "wb") as f:
        pickle.dump(dgl_list, f)



if __name__ == '__main__':

    # """3 class 1v1 change bold_matrix to one_hot matrix, prepared for EC computing using te_matlab_0.4/ADNI705_NC_LMCI.m"""
    # bold_one_hot_path_root = "E:/project/brain_functional_gnn/data/ADNI_EMCI_LMCI/one_hot_bold_matrix/"
    # norm_bold_path_root = "E:/project/brain_functional_gnn/data/ADNI_EMCI_LMCI/norm_bold_matrix/"
    #
    # mat_to_one_hot(bold_one_hot_path_root, norm_bold_path_root)


    """construct bi-layer heterogeneous graph"""
    # dataset_root = "E:/project/brain_functional_gnn/data/ADNI_NC_LMCI/"
    # label_path = "E:/project/brain_functional_gnn/data/ADNI_NC_LMCI/2_label_matrix.csv"
    # sample_num = 342
    # node_num = 90
    # pklname = "Input_ADNI_NC_LMCI_hetero.pkl"
    # # EC_single_direction(dataset_root)
    # heterogeneous_graph_construct(label_path, dataset_root, sample_num, node_num, pklname)



    # dataset_root = "E:/project/brain_functional_gnn/data/ADNI_NC_EMCI/"
    # label_path = "E:/project/brain_functional_gnn/data/ADNI_NC_EMCI/2_label_matrix.csv"
    # sample_num = 433
    # node_num = 90
    # pklname = "Input_ADNI_NC_EMCI_hetero.pkl"
    # # EC_single_direction(dataset_root)
    # heterogeneous_graph_construct(label_path, dataset_root, sample_num, node_num, pklname)



    dataset_root = "E:/project/brain_functional_gnn/data/ADNI_EMCI_LMCI/"
    label_path = "E:/project/brain_functional_gnn/data/ADNI_EMCI_LMCI/2_label_matrix.csv"
    sample_num = 389
    node_num = 90
    pklname = "Input_ADNI_EMCI_LMCI_hetero.pkl"
    # EC_single_direction(dataset_root)
    heterogeneous_graph_construct(label_path, dataset_root, sample_num, node_num, pklname)

