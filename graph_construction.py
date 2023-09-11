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


def FC_construct(mat_path, dataset_root):

    m = scipy.io.loadmat(mat_path)
    data = m.get('All_Bold')
    label = m.get('All_Group')

    """save label csv"""
    label_path = dataset_root + "label_4.csv"
    np.savetxt(label_path, label, fmt="%d", delimiter=",")


    sample_num = len(data)     #114
    FC_root_path = dataset_root + "FC/FC_adj_matrix_sample_"
    for i in range(sample_num):
        data_matrix = data[i][0].T
        norm_bold_matrix = data_matrix[:, -137:]            #116*137

        # person coefficient
        FC_matrix = np.corrcoef(norm_bold_matrix)      #116*116

        node_num = len(FC_matrix)          #116

        """threshold 0.5 value"""
        for row in range(node_num):
            FC_matrix[row][row] = 1
            for col in range(row+1, node_num):
                if (np.abs(FC_matrix[row][col]) < 0.5):
                    FC_matrix[row][col] = 0
                    FC_matrix[col][row] = 0
                else:
                    FC_matrix[row][col] = 1
                    FC_matrix[col][row] = 1

        FC_path = FC_root_path + str(i+1) + ".csv"
        np.savetxt(FC_path, FC_matrix, fmt="%d", delimiter=",")



def heterogeneous_graph_construct(dataset_root):
    FC_root = dataset_root + "FC/"
    filenames = os.listdir(FC_root)    #list, [filename1, filename2...]
    sample_num = len(filenames)

    FC_root_path = dataset_root + "FC/FC_adj_matrix_sample_"
    EC_root_path = dataset_root + "EC/EC_adj_matrix_sample_"
    HG_adj_path_root = dataset_root + "HG_adj/HG_adj_matrix_sample_"

    for i in range(sample_num):

        FC = FC_root_path + str(i+1) + ".csv"
        EC = EC_root_path + str(i+1) + ".xls"

        FC_matrix = np.loadtxt(FC, delimiter=",")
        EC_matrix = readxlsmatrix(EC)

        FC_edge_num = 0
        EC_edge_num = 0
        node_num = len(FC_matrix)

        """adj matrix"""
        HG_adj = FC_matrix
        for row in range(node_num):
            for col in range(row+1, node_num):
                if FC_matrix[row][col] == 1:
                    FC_edge_num += 1
                else:
                    if EC_matrix[row][col] > EC_matrix[col][row]:
                        HG_adj[row][col] = EC_matrix[row][col]
                    else:
                        HG_adj[col][row] = EC_matrix[col][row]
                    EC_edge_num += 1

        print("sample_%d, FC_edge_num:%d, EC_edge_num:%d"%(i+1, FC_edge_num, EC_edge_num))

        HG_adj_path = HG_adj_path_root + str(i+1) + ".csv"
        np.savetxt(HG_adj_path, HG_adj, fmt="%f", delimiter=",")


def heterogeneous_graph_feature(dataset_root, aal_rsn_path):
    FC_root = dataset_root + "FC/"
    filenames = os.listdir(FC_root)    #list, [filename1, filename2...]
    sample_num = len(filenames)

    FC_root_path = dataset_root + "FC/FC_adj_matrix_sample_"

    table = xlrd.open_workbook(aal_rsn_path).sheets()[0]
    row = table.nrows
    col = table.ncols

    node_type_90_list = []
    for i in range(row):
        if table.cell(i, 1).value == "Sensorimotor":
            node_type_90_list.append(1)
        elif table.cell(i, 1).value == "Default Mode":
            node_type_90_list.append(2)
        elif table.cell(i, 1).value == "Attention":
            node_type_90_list.append(3)
        elif table.cell(i, 1).value == "Subcortical":
            node_type_90_list.append(4)
        elif table.cell(i, 1).value == "Visual":
            node_type_90_list.append(5)
        else:
            print("Exception Value in line%d", i+1)


    #90 regions in cerebrum, 26 regions in cerebellum
    node_type_26_list = [0 for x in range(26)]
    node_type_116_list = node_type_90_list + node_type_26_list


    node_type_fea_matrix = np.eye(6)[node_type_116_list]          #116*6


    HG_fea_path_root = dataset_root + "HG_fea/HG_fea_matrix_sample_"

    for i in range(sample_num):
        FC = FC_root_path + str(i+1) + ".csv"
        FC_matrix = np.loadtxt(FC, delimiter=",")

        HG_fea_matrix_orginal = FC_matrix           #116*116

        HG_fea_matrix = np.concatenate((HG_fea_matrix_orginal, node_type_fea_matrix), axis=1)    #116*122

        HG_fea_path = HG_fea_path_root + str(i+1) + ".csv"
        np.savetxt(HG_fea_path, HG_fea_matrix, delimiter=",", fmt="%d")


def heterogeneous_graph_dgl(dataset_root, aal_rsn_path):
    HG_adj_root = dataset_root + "HG_adj/"
    filenames = os.listdir(HG_adj_root)  # list, [filename1, filename2...]
    sample_num = len(filenames)

    HG_adj_path = dataset_root + "HG_adj/HG_adj_matrix_sample_"
    HG_fea_path = dataset_root + "HG_fea/HG_fea_matrix_sample_"

    label_path = dataset_root + "label_4.csv"
    label_matrix = np.loadtxt(label_path, delimiter=",")

    Dataset_list = []

    table = xlrd.open_workbook(aal_rsn_path).sheets()[0]
    row = table.nrows
    col = table.ncols


    for i in range(sample_num):

        HG_adj_file_path = HG_adj_path + str(i+1) + ".csv"
        HG_adj = np.loadtxt(HG_adj_file_path, delimiter=",")

        node_num = len(HG_adj)

        graph_data = {}

        FC_edge_s_list = []
        FC_edge_e_list = []
        EC_edge_s_list = []
        EC_edge_e_list = []

        for row in range(node_num):
            for col in range(node_num):
                if HG_adj[row][col] == 1:
                    FC_edge_s_list.append(row)
                    FC_edge_e_list.append(col)
                elif HG_adj[row][col] !=0:
                    EC_edge_s_list.append(row)
                    EC_edge_e_list.append(col)

        graph_data[('region', 'FC', 'region')] = (th.tensor(FC_edge_s_list)), (th.tensor(FC_edge_e_list))
        graph_data[('region', 'EC', 'region')] = (th.tensor(EC_edge_s_list)), (th.tensor(EC_edge_e_list))


        hg = dgl.heterograph(graph_data)

        print("hg")
        print(hg)





        # graph_data_complex = {}
        #
        # SE_SE_FC_edge_s_list = []
        # SE_SE_FC_edge_e_list = []
        # SE_SE_EC_edge_s_list = []
        # SE_SE_EC_edge_e_list = []
        #
        # for row in range(node_num):
        #     for col in range(node_num):
        #         if HG_adj[row][col] == 1:
        #             if col>=90:
        #                 if row>=90:
        #                     SE_SE_FC_edge_s_list.append(row)
        #                     SE_SE_FC_edge_e_list.append(col)
        #                 elif
        #
        #
        #
        #             elif node_type(row) == "Sensorimotor" and node_type(col) == "Sensorimotor":
        #                 SE_SE_FC_edge_s_list.append(row)
        #                 SE_SE_FC_edge_e_list.append(col)
        #             elif
        #
        #
        #         elif HG_adj[row][col] != 0:
        #             EC_edge_s_list.append(row)
        #             EC_edge_e_list.append(col)
        #
        #
        #
        #
        # graph_data[('Sensorimotor', 'FC', 'Sensorimotor')] = (th.tensor(SE_SE_FC_edge_s_list)), (th.tensor(SE_SE_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Sensorimotor')] = (th.tensor(SE_SE_EC_edge_s_list)), (th.tensor(SE_SE_EC_edge_e_list))
        # graph_data[('Sensorimotor', 'FC', 'Default_Mode')] = (th.tensor(SE_DE_FC_edge_s_list)), (th.tensor(SE_DE_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Default_Mode')] = (th.tensor(SE_DE_EC_edge_s_list)), (th.tensor(SE_DE_EC_edge_e_list))
        # graph_data[('Sensorimotor', 'FC', 'Attention')] = (th.tensor(SE_AT_FC_edge_s_list)), (th.tensor(SE_AT_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Attention')] = (th.tensor(SE_AT_EC_edge_s_list)), (th.tensor(SE_AT_EC_edge_e_list))
        # graph_data[('Sensorimotor', 'FC', 'Subcortical')] = (th.tensor(SE_SU_FC_edge_s_list)), (th.tensor(SE_SU_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Subcortical')] = (th.tensor(SE_SU_EC_edge_s_list)), (th.tensor(SE_SU_EC_edge_e_list))
        # graph_data[('Sensorimotor', 'FC', 'Visual')] = (th.tensor(SE_VI_FC_edge_s_list)), (th.tensor(SE_VI_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Visual')] = (th.tensor(SE_VI_EC_edge_s_list)), (th.tensor(SE_VI_EC_edge_e_list))
        # graph_data[('Sensorimotor', 'FC', 'Cerebellum')] = (th.tensor(SE_CE_FC_edge_s_list)), (th.tensor(SE_CE_FC_edge_e_list))
        # graph_data[('Sensorimotor', 'EC', 'Cerebellum')] = (th.tensor(SE_CE_EC_edge_s_list)), (th.tensor(SE_CE_EC_edge_e_list))
        #
        # graph_data[('Default_Mode', 'FC', 'Sensorimotor')] = (th.tensor(DE_SE_FC_edge_s_list)), (th.tensor(DE_SE_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Sensorimotor')] = (th.tensor(DE_SE_EC_edge_s_list)), (th.tensor(DE_SE_EC_edge_e_list))
        # graph_data[('Default_Mode', 'FC', 'Default_Mode')] = (th.tensor(DE_DE_FC_edge_s_list)), (th.tensor(DE_DE_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Default_Mode')] = (th.tensor(DE_DE_EC_edge_s_list)), (th.tensor(DE_DE_EC_edge_e_list))
        # graph_data[('Default_Mode', 'FC', 'Attention')] = (th.tensor(DE_AT_FC_edge_s_list)), (th.tensor(DE_AT_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Attention')] = (th.tensor(DE_AT_EC_edge_s_list)), (th.tensor(DE_AT_EC_edge_e_list))
        # graph_data[('Default_Mode', 'FC', 'Subcortical')] = (th.tensor(DE_SU_FC_edge_s_list)), (th.tensor(DE_SU_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Subcortical')] = (th.tensor(DE_SU_EC_edge_s_list)), (th.tensor(DE_SU_EC_edge_e_list))
        # graph_data[('Default_Mode', 'FC', 'Visual')] = (th.tensor(DE_VI_FC_edge_s_list)), (th.tensor(DE_VI_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Visual')] = (th.tensor(DE_VI_EC_edge_s_list)), (th.tensor(DE_VI_EC_edge_e_list))
        # graph_data[('Default_Mode', 'FC', 'Cerebellum')] = (th.tensor(DE_CE_FC_edge_s_list)), (th.tensor(DE_CE_FC_edge_e_list))
        # graph_data[('Default_Mode', 'EC', 'Cerebellum')] = (th.tensor(DE_CE_EC_edge_s_list)), (th.tensor(DE_CE_EC_edge_e_list))






        """fea_dgl: h_dict"""
        """{'type1':tensor([[]]), 'type2':}"""
        HG_fea_file_path = HG_fea_path + str(i+1) + ".csv"
        HG_fea = np.loadtxt(HG_fea_file_path, delimiter=",")
        h_dict = {}

        type_Sensorimotor_index_list = []
        type_Default_Mode_index_list = []
        type_Attention_index_list = []
        type_Subcortical_index_list = []
        type_Visual_index_list = []
        type_Cerebellum_index_list = [range(90,116)]

        table = xlrd.open_workbook(aal_rsn_path).sheets()[0]

        for row2 in range(90):
            if table.cell(row2, 1).value == "Sensorimotor":
                type_Sensorimotor_index_list.append(row2)
            elif table.cell(row2, 1).value == "Default Mode":
                type_Default_Mode_index_list.append(row2)
            elif table.cell(row2, 1).value == "Attention":
                type_Attention_index_list.append(row2)
            elif table.cell(row2, 1).value == "Subcortical":
                type_Subcortical_index_list.append(row2)
            elif table.cell(row2, 1).value == "Visual":
                type_Visual_index_list.append(row2)
            else:
                print("Excepetion Value in 90_node type")

        # h_dict['Sensorimotor'] = th.tensor(HG_fea[type_Sensorimotor_index_list, :])
        # h_dict['Default_Mode'] = th.tensor(HG_fea[type_Default_Mode_index_list, :])
        # h_dict['Attention'] = th.tensor(HG_fea[type_Attention_index_list, :])
        # h_dict['Subcortical'] = th.tensor(HG_fea[type_Subcortical_index_list, :])
        # h_dict['Visual'] = th.tensor(HG_fea[type_Visual_index_list, :])
        # h_dict['Cerebellum'] = th.tensor(HG_fea[type_Cerebellum_index_list, :])


        """taken as single node type"""
        h_dict['region'] = th.tensor(HG_fea[type_Sensorimotor_index_list, :])
        h_dict['region'] = th.tensor(HG_fea[type_Default_Mode_index_list, :])
        h_dict['region'] = th.tensor(HG_fea[type_Attention_index_list, :])
        h_dict['region'] = th.tensor(HG_fea[type_Subcortical_index_list, :])
        h_dict['region'] = th.tensor(HG_fea[type_Visual_index_list, :])
        h_dict['region'] = th.tensor(HG_fea[type_Cerebellum_index_list, :])


        print("h_dict")
        print(h_dict)


        """label"""   #norm to 0,1,2,3
        label = label_matrix[i] - 1
        label_tensor = th.tensor(label)

        signle_graph_tuple = (hg, h_dict, label_tensor)

        Dataset_list.append(signle_graph_tuple)


    Input_ADNI114_path = dataset_root + "Input_ADNI114_single_nty.pkl"
    with open(Input_ADNI114_path, "wb") as f:
        pickle.dump(Dataset_list, f)




def heterogeneous_graph_dgl2(dataset_root, aal_rsn_path):
    HG_adj_root = dataset_root + "HG_adj/"
    filenames = os.listdir(HG_adj_root)  # list, [filename1, filename2...]
    sample_num = len(filenames)

    HG_adj_path = dataset_root + "HG_adj/HG_adj_matrix_sample_"
    HG_fea_path = dataset_root + "HG_fea/HG_fea_matrix_sample_"

    label_path = dataset_root + "label_4.csv"
    label_matrix = np.loadtxt(label_path, delimiter=",")

    Dataset_list = []

    table = xlrd.open_workbook(aal_rsn_path).sheets()[0]
    row = table.nrows
    col = table.ncols

    node_type_90_list = []
    for i in range(row):
        if table.cell(i, 1).value == "Sensorimotor":
            node_type_90_list.append(1)
        elif table.cell(i, 1).value == "Default Mode":
            node_type_90_list.append(2)
        elif table.cell(i, 1).value == "Attention":
            node_type_90_list.append(3)
        elif table.cell(i, 1).value == "Subcortical":
            node_type_90_list.append(4)
        elif table.cell(i, 1).value == "Visual":
            node_type_90_list.append(5)
        else:
            print("Exception Value in line%d", i + 1)

    # 90 regions in cerebrum, 26 regions in cerebellum
    node_type_26_list = [0 for x in range(26)]
    node_type_116_list = node_type_90_list + node_type_26_list

    node_type_fea_matrix = np.eye(6)[node_type_116_list]
    node_type_fea_matrix_tensor = th.tensor(node_type_fea_matrix)

    node_type_116_tensor = th.tensor(node_type_116_list)


    for i in range(sample_num):

        HG_adj_file_path = HG_adj_path + str(i+1) + ".csv"
        HG_adj = np.loadtxt(HG_adj_file_path, delimiter=",")

        node_num = len(HG_adj)

        graph_data = {}

        FC_edge_s_list = []
        FC_edge_e_list = []
        EC_edge_s_list = []
        EC_edge_e_list = []

        for row in range(node_num):
            for col in range(node_num):
                if HG_adj[row][col] == 1:
                    FC_edge_s_list.append(row)
                    FC_edge_e_list.append(col)
                elif HG_adj[row][col] !=0:
                    EC_edge_s_list.append(row)
                    EC_edge_e_list.append(col)

        graph_data[('region', 'FC', 'region')] = (th.tensor(FC_edge_s_list)), (th.tensor(FC_edge_e_list))
        graph_data[('region', 'EC', 'region')] = (th.tensor(EC_edge_s_list)), (th.tensor(EC_edge_e_list))


        hg = dgl.heterograph(graph_data)

        print("hg")
        print(hg)



        """fea_dgl: h_dict"""
        """{'type1':tensor([[]]), 'type2':}"""
        HG_fea_file_path = HG_fea_path + str(i+1) + ".csv"
        HG_fea = np.loadtxt(HG_fea_file_path, delimiter=",")
        h_dict = {}

        type_Sensorimotor_index_list = []
        type_Default_Mode_index_list = []
        type_Attention_index_list = []
        type_Subcortical_index_list = []
        type_Visual_index_list = []
        type_Cerebellum_index_list = range(90,116)

        table = xlrd.open_workbook(aal_rsn_path).sheets()[0]

        for row2 in range(90):
            if table.cell(row2, 1).value == "Sensorimotor":
                type_Sensorimotor_index_list.append(row2)
            elif table.cell(row2, 1).value == "Default Mode":
                type_Default_Mode_index_list.append(row2)
            elif table.cell(row2, 1).value == "Attention":
                type_Attention_index_list.append(row2)
            elif table.cell(row2, 1).value == "Subcortical":
                type_Subcortical_index_list.append(row2)
            elif table.cell(row2, 1).value == "Visual":
                type_Visual_index_list.append(row2)
            else:
                print("Excepetion Value in 90_node type")


        """taken as single node type"""
        h_dict['region'] = th.tensor(HG_fea[type_Sensorimotor_index_list, :])
        h_dict['region'] = th.cat((h_dict['region'], th.tensor(HG_fea[type_Default_Mode_index_list, :])), 0)
        h_dict['region'] = th.cat((h_dict['region'], th.tensor(HG_fea[type_Attention_index_list, :])), 0)
        h_dict['region'] = th.cat((h_dict['region'], th.tensor(HG_fea[type_Subcortical_index_list, :])), 0)
        h_dict['region'] = th.cat((h_dict['region'], th.tensor(HG_fea[type_Visual_index_list, :])), 0)
        h_dict['region'] = th.cat((h_dict['region'], th.tensor(HG_fea[type_Cerebellum_index_list, :])), 0)



        hg.ndata['feat'] = h_dict['region']
        hg.ndata['nodelabel'] = node_type_116_tensor


        """label"""   #norm to 0,1,2,3
        label = label_matrix[i] - 1
        label_tensor = th.tensor(label)

        signle_graph_tuple = (hg, label_tensor)

        Dataset_list.append(signle_graph_tuple)


    Input_ADNI114_path = dataset_root + "Input_ADNI114_2.pkl"
    with open(Input_ADNI114_path, "wb") as f:
        pickle.dump(Dataset_list, f)




if __name__ == '__main__':

    mat_path = "E:/project/dataset/data114.mat"
    dataset_root = "E:/project/brain_network_FCEC_graph_construction/data/ADNI114/"


    aal_rsn_path = "E:/project/brain_network_FCEC_graph_construction/data/aal_rsn_90.xls"


    """construct FC"""
    # FC_construct(mat_path, dataset_root)

    """construct Heterogeneous Graph"""
    # heterogeneous_graph_construct(dataset_root)         #adj

    # heterogeneous_graph_feature(dataset_root, aal_rsn_path)         #fea

    # heterogeneous_graph_dgl(dataset_root, aal_rsn_path)

    heterogeneous_graph_dgl2(dataset_root, aal_rsn_path)
