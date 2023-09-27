import numpy as np

import argparse
import pickle
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import copy
import visdom
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import os
from model import FunctionalGNN
from model_hetero import HeteroClassifier
from torch.distributions import Categorical, kl


def model_measurements(total_predict_label, total_predict_proba_label, total_test_label):

    total_test_accuracy = accuracy_score(total_test_label, total_predict_label)
    total_test_precision = metrics.precision_score(total_test_label, total_predict_label, average='macro')
    total_test_recall = metrics.recall_score(total_test_label, total_predict_label, average='macro')
    total_test_f1 = f1_score(total_test_label, total_predict_label, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(total_test_label, total_predict_label)
    TP = test_confusion_matrix[1, 1]
    TN = test_confusion_matrix[0, 0]
    FP = test_confusion_matrix[0, 1]
    FN = test_confusion_matrix[1, 0]

    test_specificity = TN / float(TN + FP)

    labels = np.zeros(shape=[len(total_test_label), 2], dtype=np.int32)

    for i, index in enumerate(total_test_label):
        labels[i, int(index)] = 1

    one_hot_total_test_label = np.array(labels)
    #
    # print(np.shape(one_hot_total_test_label))
    # print(total_predict_proba_label.size())

    """3 class"""
    # test_AUC = metrics.roc_auc_score(one_hot_total_test_label, total_predict_proba_label, multi_class='ovo')

    # y_one_hot = np.zeros((len(total_test_label), 2), np.float32)
    # for label_line in range(len(total_test_label)):
    #     if total_test_label[label_line].astype(int) == 0:
    #         y_one_hot[label_line][0] = 1.0
    #     if total_test_label[label_line].astype(int) == 1:
    #         y_one_hot[label_line][1] = 1.0
    fpr, tpr, thresholds = metrics.roc_curve(one_hot_total_test_label.ravel(), total_predict_proba_label.ravel())
    auc = metrics.auc(fpr, tpr)
    """2 class"""
    test_AUC = metrics.roc_auc_score(total_test_label, total_predict_proba_label[:,1].T)
    # fpr, tpr, thresholds = metrics.roc_curve(total_test_label, total_predict_proba_label[:,1].T)

    print("confusion matrix", test_confusion_matrix)

    # return total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, fpr, tpr, thresholds
    return total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, auc


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    # print("total test num", total)

    total_predict = []
    total_predict_prob = []
    total_true = []
    val_step = 0

    for x, y in loader:
        input_feature = x.ndata['x']
        # predict = model(batched_graph, h_dict)
        # x, y = x.to(device), y.to(device)
        with torch.no_grad():
            # logits, S_assign_matirx, h_1_list, attention_map = model(x.to(device),input_feature.float().to(device))

            logits, S_assign_matirx = model(x.to(device))


            # test_Y = torch.tensor(test_Y).float().view(-1, 1)
            # probs_Y = F.softmax(logits, 1)


            pred = logits.argmax(dim=1)


            # print("test predict", pred)
            # print("test predict prob", logits)
            # print("test true", y)

            total_predict.append(pred)
            total_predict_prob.append(logits)
            total_true.append(y)


        correct += torch.eq(pred.cpu(), y).sum().float().item()

    total_predict_label = torch.cat(total_predict, axis=0)
    total_predict_proba_label = torch.cat(total_predict_prob, axis=0)
    total_test_label = torch.cat(total_true, axis=0)

    print("total_predict_label", total_predict_proba_label)
    print("total_predict_label", total_predict_label)
    print("total_test_label", total_test_label)

    return correct / total, total_predict_label, total_predict_proba_label, total_test_label




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='ADNI_NC_EMCI', type=str, help='name of datasets')  # ADNI114, ADNI705, ABIDE, ADNI_NC_LMCI, ADNI_EMCI_LMCI, ADNI705
    parser.add_argument('--inputpkl', default='ADNI_NC_EMCI_hetero', type=str, help='name of input pkl')  # ADNI114_2, ADNI705_2, ABIDE_2, ADNI_NC_LMCI, ADNI_EMCI_LMCI, ADNI705_3
    parser.add_argument('--node_num', default='90', type=int, help='node num')  # 116,  90, 116, 90
    parser.add_argument('--cluster_num', default='7', type=int, help='cluster corresponding to node type')  # 7
    parser.add_argument('--in_dim', default='90', type=int, help='input feature dimenson (node num)')  # 116, 90, 116, 90
    parser.add_argument('--class_num', default='2', type=int, help='graph classification num')  # 4, 2

    parser.add_argument('--max_epoch', default='25', type=int, help='max epoch')
    parser.add_argument('--batch_size', default='32', type=int, help='batch size')
    parser.add_argument('--lr', default='1e-3', type=float, help='learning rate')

    parser.add_argument('--hid_layer', default='256', type=int, help='hidden layer feature dimension')  # 1024, 512
    parser.add_argument('--drop', default='0.01', type=float, help='dropout rate')  # 0.5, 0.5
    parser.add_argument('--weight_decay', default='0.09', type=float, help='L2 loss weight')  # 0.5, 0.5


    parser.add_argument('--n_heads', default='4', type=int, help='attention head num')  # 1, 8, 4, 4
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--attn_drop', default='1e-6', type=float, help='attention dropout rate')  # 0.9, 0.9

    parser.add_argument('--alpha', default='0.001', type=float, help='coef of assign matrix')  # 0.9, 0.9, 0.01

    args = parser.parse_args()

    viz = visdom.Visdom(env='model_hetero')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    Input_pkl_path = "./data/" + args.dataset + "/Input_" + args.inputpkl + ".pkl"

    print("Input_pkl", Input_pkl_path)
    File = open(Input_pkl_path, 'rb')
    dataset = pickle.load(File)


    best_acc_fold_list = []
    fold_num = 1

    total_result = []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(dataset):

        # print('train_index:%s , test_index: %s ' %(train_index,test_index))

        dataset_train = [dataset[i] for i in train_index]
        dataset_test = [dataset[i] for i in test_index]

        dataloader_train = GraphDataLoader(
            dataset_train,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True)

        dataloader_test = GraphDataLoader(
            dataset_test,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True)

        # model = FunctionalGNN(args.in_dim, args.hid_layer, args.class_num, args.node_num, args.cluster_num, args.drop, args.n_heads, args.attn_drop, args.residual).to(device)

        etypes = ['FC', 'EC', 'Corr_FE', 'Corr_EF']
        model = HeteroClassifier(args.in_dim, args.hid_layer, args.class_num, etypes, args.node_num, args.cluster_num, args.n_heads, args.attn_drop, args.drop).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)


        best_acc, best_epoch = 0, 0
        best_epoch_predict = torch.tensor([])
        best_epoch_predict_prob = torch.tensor([])
        epoch_y = torch.tensor([])


        best_model = None
        epoch_losses = []
        viz.line([0], [-1], win='epoch_loss', opts=dict(title='epoch loss'))
        viz.line([0], [-1], win='loss', opts=dict(title='loss'))
        viz.line([0], [-1], win='val_loss', opts=dict(title='val loss'))
        viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0], [-1], win='best_val_acc', opts=dict(title='best_val_acc'))
        viz.line([0], [-1], win='train_iter_acc', opts=dict(title='train_iter_acc'))
        global_step = 0



        for epoch in range(args.max_epoch):
            epoch_loss = 0


            train_correct = 0
            train_num = len(dataloader_train.dataset)

            epoch_labels = []
            epoch_train_predict = []

            for iter, (batched_graph, labels) in enumerate(dataloader_train):

                model.train()

                input_feature = batched_graph.ndata['x']
                # predict, S_assign_matirx, h_1_list, attention_map = model(batched_graph.to(device))
                predict, S_assign_matirx = model(batched_graph.to(device))


                # np.savetxt("./")

                # S_assign_matirx.cpu()
                Entropy = Categorical(S_assign_matirx[0]).entropy().item()

                # Entropy = -sum([S_assign_matirx[0][i] * np.log(S_assign_matirx[0][i]) for i in range(len(S_assign_matirx[0]))])
                # Entropy = Categorical(S_assign_matirx[S_assign_matirx_row])

                S_assign_matirx_row_num = len(S_assign_matirx)
                for S_assign_matirx_row in range(S_assign_matirx_row_num):
                    Entropy += Categorical(S_assign_matirx[S_assign_matirx_row]).entropy().item()
                loss_assign = Entropy/S_assign_matirx_row_num

                loss = F.cross_entropy(predict.float(), labels.to(device).long()) + args.alpha * loss_assign
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()


                # if epoch == 10:
                #     assign_matrix_path = "./data/"+ args.dataset +"/assignmatrix/assign_" + str(iter+1) + ".csv"
                #     np.savetxt(assign_matrix_path, S_assign_matirx.cpu().detach().numpy(), delimiter=",")

                    # h_1_list_path_root = "./data/" + args.dataset + "/h_1_list/list_iter_" + str(iter + 1) + "_class_"
                    # for h_1_list_row in range(len(h_1_list)):
                    #     h_1_list_path = h_1_list_path_root + str(h_1_list_row+1) + ".csv"
                    #     np.savetxt(h_1_list_path, h_1_list[h_1_list_row][:args.node_num,:].cpu().detach().numpy(), delimiter=",")



                viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1


                epoch_labels.extend(labels)
                epoch_train_predict.extend(torch.argmax(predict.cpu(), 1))
                train_accuracy = accuracy_score(labels, torch.argmax(predict.cpu(), 1))
                viz.line([train_accuracy], [global_step], win='train_iter_acc', update='append')
                print("iteration acc", train_accuracy)

                # scheduler.step()

            train_accuracy_epoch = accuracy_score(epoch_labels, epoch_train_predict)
            print("train epoch acc", train_accuracy_epoch)

            if epoch % 1 == 0:
                val_acc, total_predict_label, total_predict_proba_label, total_test_label = evalute(model, dataloader_test)

                viz.line([val_acc], [epoch], win='val_acc', update='append')

                val_loss = F.cross_entropy(total_predict_proba_label.float(), total_test_label.to(device).long())
                viz.line([val_loss.item()], [epoch], win='val_loss', update='append')


                if val_acc > best_acc:
                    best_epoch = epoch
                    best_acc = val_acc
                    best_model = copy.deepcopy(model)
                    best_epoch_predict = total_predict_label.cpu()
                    best_epoch_predict_prob = total_predict_proba_label.cpu()
                    epoch_y = total_test_label

                viz.line([best_acc], [epoch], win='best_val_acc', update='append')

            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

            viz.line([epoch_loss], [epoch], win='epoch_loss', update='append')


        print('each fold best acc:', best_acc, 'best epoch:', best_epoch)
        best_model_para_path = './data/' + args.dataset + '/parameters.pkl'
        torch.save(best_model.state_dict(), best_model_para_path)

        # total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, fpr, tpr, thresholds \
        #     = model_measurements(best_epoch_predict, best_epoch_predict_prob, epoch_y)

        total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, auc \
            = model_measurements(best_epoch_predict, best_epoch_predict_prob, epoch_y)

        print(
            ' Test acc: %.4f | Test pre: %.4f | Test rec: %.4f | Test F1: %.4f | Test specificity: %.4f | Test auc: %.4f | Test auc_one_hot: %.4f' %
            (total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_specificity,
             test_AUC, auc))

        # print("fpr", fpr)
        # print("tpr", tpr)
        # auc_prob = metrics.auc(fpr, tpr)
        # print('auc_prob：', auc_prob)




        fold_result_path = "./data/" + args.dataset + "/result_hetero/fold_" + str(fold_num) + "_result.csv"

        # print("best_epoch_predict.unsqueeze(1).T", best_epoch_predict.unsqueeze(1).T.size())
        # print("best_epoch_predict_prob.unsqueeze(1).T", best_epoch_predict_prob.T.size())
        # print("epoch_y.T", epoch_y.unsqueeze(1).T.size())

        result = torch.cat([best_epoch_predict.unsqueeze(1).T, best_epoch_predict_prob.T, epoch_y.unsqueeze(1).T], axis=0).detach().numpy()
        np.savetxt(fold_result_path, result, delimiter=",", fmt="%f")
        fold_num += 1

        best_acc_fold_list.append(best_acc)

        total_result.append(result)

    print(best_acc_fold_list)

    total_result_path = "./data/" + args.dataset + "/result_hetero/total_result.csv"
    total_result_matrix = np.concatenate(total_result, axis=1)
    np.savetxt(total_result_path, total_result_matrix, delimiter=",", fmt="%f")