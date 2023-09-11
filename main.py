import numpy as np
import openhgnn
import argparse
# from openhgnn.experiment import Experiment
import pickle
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from model import HeteroClassifier
from openhgnn import models
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

    test_AUC = metrics.roc_auc_score(total_test_label, total_predict_proba_label[:,1].T)

    fpr, tpr, thresholds = metrics.roc_curve(total_test_label, total_predict_proba_label[:,1].T)


    return total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, fpr, tpr, thresholds


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
        h_dict = x.ndata['feat']
        # predict = model(batched_graph, h_dict)
        # x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, val_reconstruction_loss = model(x.to(device), h_dict.to(device))

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


    print("total_predict_label", total_predict_label)
    print("total_test_label", total_test_label)

    return correct / total, total_predict_label, total_predict_proba_label, total_test_label



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', default='SimpleHGN', type=str, help='name of models')
    # parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')


    # parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    # parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    # parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')

    parser.add_argument('--dataset', '-d', default='ADNI705', type=str, help='name of datasets')   #ADNI114, ADNI705, ABIDE
    parser.add_argument('--inputpkl', default='ADNI705_2', type=str, help='name of input pkl')   #ADNI114_2, ADNI705_2, ABIDE
    parser.add_argument('--node_num', default='90', type=int, help='node num')    # 116,  90
    parser.add_argument('--cluster', default='7', type=int, help='cluster corresponding to node type') #3, 5, 7
    parser.add_argument('--in_dim', default='95', type=int, help='input feature dimenson (node num+ node type)')  # 122(116+6), 95(90+5)
    parser.add_argument('--class_num', default='2', type=int, help='graph classification num')  # 4, 2

    parser.add_argument('--max_epoch', default='150', type=int, help='max epoch')
    parser.add_argument('--batch_size', default='32', type=int, help='batch size')
    parser.add_argument('--lr', default='1e-4', type=float, help='learning rate')

    parser.add_argument('--hid_layer', default='512', type=int, help='hidden layer feature dimension')  # 1024, 512
    parser.add_argument('--drop', default='0.5', type=float, help='dropout rate in hgcn')  # 0.5, 0.5
    parser.add_argument('--weight_decay', default='0.1', type=float, help='L2 loss weight')  # 0.5, 0.5


    parser.add_argument('--n_heads', default='4', type=int, help='attention head num')  # 1, 8
    parser.add_argument('--attn_drop', default='0.9', type=float, help='attention dropout rate')  # 0.9, 0.9


    args = parser.parse_args()

    viz = visdom.Visdom(env='model_1')

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    Input_pkl_path = "./data/" + args.dataset + "/Input_" + args.inputpkl + ".pkl"
    File = open(Input_pkl_path, 'rb')
    dataset = pickle.load(File)


    best_acc_fold_list = []
    fold_num = 1

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(dataset):


        print('train_index:%s , test_index: %s ' %(train_index,test_index))

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

        etypes = ['FC', 'EC']
        model = HeteroClassifier(args.in_dim, args.hid_layer, args.class_num, etypes,
                                 args.node_num, args.cluster, args.n_heads, args.attn_drop, args.drop).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)


        best_acc, best_epoch = 0, 0
        best_epoch_predict = torch.tensor([])
        best_epoch_predict_prob = torch.tensor([])
        epoch_y = torch.tensor([])


        best_model = None
        epoch_losses = []
        viz.line([0], [-1], win='loss', opts=dict(title='loss'))
        viz.line([0], [-1], win='val_loss', opts=dict(title='val loss'))
        viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0], [-1], win='best_val_acc', opts=dict(title='best_val_acc'))
        viz.line([0], [-1], win='train_iter_acc', opts=dict(title='train_iter_acc'))
        global_step = 0
        for epoch in range(args.max_epoch):
            epoch_loss = 0
            scheduler.step()

            train_correct = 0
            train_num = len(dataloader_train.dataset)

            epoch_labels = []
            epoch_train_predict = []

            for iter, (batched_graph, labels) in enumerate(dataloader_train):
                h_dict = batched_graph.ndata['feat']
                predict, reconstruction_loss = model(batched_graph.to(device), h_dict.to(device))
                loss = F.cross_entropy(predict.float(), labels.to(device).long()) + 0.05 * reconstruction_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1


                epoch_labels.extend(labels)
                epoch_train_predict.extend(torch.argmax(predict.cpu(), 1))
                train_accuracy = accuracy_score(labels, torch.argmax(predict.cpu(), 1))
                viz.line([train_accuracy], [global_step], win='train_iter_acc', update='append')
                print("iteration acc", train_accuracy)


            #     train_pred = predict.argmax(dim=1)
            #     print("train_pred", train_pred)
            #     train_correct += torch.eq(train_pred, labels).sum().float().item()
            # train_acc = train_correct / train_num
            # print("train epoch acc", train_acc)

            # print("epoch_labels", epoch_labels)
            # print("epoch_train_predict", epoch_train_predict)
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


        print('each fold best acc:', best_acc, 'best epoch:', best_epoch)
        best_model_para_path = './data/' + args.dataset + '/parameters.pkl'
        torch.save(best_model.state_dict(), best_model_para_path)

        total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_confusion_matrix, test_specificity, test_AUC, fpr, tpr, thresholds \
            = model_measurements(best_epoch_predict, best_epoch_predict_prob, epoch_y)

        print(' Test acc: %.4f | Test pre: %.4f | Test rec: %.4f | Test F1: %.4f | Test specificity: %.4f | Test auc: %.4f' %
              (total_test_accuracy, total_test_precision, total_test_recall, total_test_f1, test_specificity, test_AUC))

        print("fpr", fpr)
        print("tpr", tpr)
        auc_prob = metrics.auc(fpr, tpr)
        print('auc_prob：', auc_prob)


        fold_result_path = "./data/" + args.dataset + "/result/fold_" + str(fold_num) + "_result.csv"

        # print("best_epoch_predict.unsqueeze(1).T", best_epoch_predict.unsqueeze(1).T.size())
        # print("best_epoch_predict_prob.unsqueeze(1).T", best_epoch_predict_prob.T.size())
        # print("epoch_y.T", epoch_y.unsqueeze(1).T.size())

        result = torch.cat([best_epoch_predict.unsqueeze(1).T, best_epoch_predict_prob.T, epoch_y.unsqueeze(1).T], axis=0).detach().numpy()
        np.savetxt(fold_result_path, result, delimiter=",", fmt="%f")
        fold_num += 1

        best_acc_fold_list.append(best_acc)


        # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % test_AUC)
        # plt.plot((0, 1), (0, 1), c='grey', lw=1, ls='--', alpha=0.7)
        # plt.xlim((-0.01, 1.02))
        # plt.ylim((-0.01, 1.02))
        # plt.xticks(np.arange(0, 1.1, 0.1))
        # plt.yticks(np.arange(0, 1.1, 0.1))
        # plt.xlabel('False Positive Rate', fontsize=13)
        # plt.ylabel('True Positive Rate', fontsize=13)
        # plt.grid(b=True, ls=':')
        # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        # plt.title(u'ROC and AUC', fontsize=17)
        # plt.show()


        # plt.title('loss averaged over minibatches')
        # plt.plot(epoch_losses)
        # plt.show()

        # torch.save(model.state_dict(), 'parameters.pkl')
        # model.load_state_dict(torch.load('parameters.pkl'))


    print(best_acc_fold_list)

