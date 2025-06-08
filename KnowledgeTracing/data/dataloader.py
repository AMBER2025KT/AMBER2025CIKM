import sys

sys.path.append('../')
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.preprocess import DataReader
from KnowledgeTracing.data.OneHot import OneHot,OneHotM


def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP)
    trainques, trainans = handle.getTrainData()
    dtrain = OneHot(trainques, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True)
    return trainLoader


def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP)
    testques, testans = handle.getTestData()
    dtest = OneHot(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False, drop_last=True)
    return testLoader



def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    heldLoaders = []
    if dataset == 'assist2009':
        trainLoader = getTrainLoader(C.Dpath + '/assist2009/assist2009_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2009/assist2009_pid_test.csv')
        testLoaders.append(testLoader)
        heldLoader = getTrainLoader(C.Dpath + '/assist2009/held.csv')
        heldLoaders.append(heldLoader)
    elif dataset == 'assist2017':
        trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_pid_test.csv')
        testLoaders.append(testLoader)
        heldLoader = getTrainLoader(C.Dpath + '/assist2017/held.csv')
        heldLoaders.append(heldLoader)
    elif dataset == 'assistednet':
        trainLoader = getTrainLoader(C.Dpath + '/assistednet/assistednet_pid_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/assistednet/assistednet_pid_test.csv')
        testLoaders.append(testLoader)
        heldLoader = getTrainLoader(C.Dpath + '/assistednet/held.csv')
        heldLoaders.append(heldLoader)

    return trainLoaders[0], testLoaders[0], heldLoaders[0]


def getTrainDataset(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP)
    trainques, trainans = handle.getTrainData()
    dtrain = OneHotM(trainques, trainans)
    return dtrain


def getTestDataset(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP)
    testques, testans = handle.getTestData()
    dtest = OneHotM(testques, testans)
    return dtest


def getHeldDataset(held_data_path):
    handle = DataReader(held_data_path, C.MAX_STEP)
    testques, testans = handle.getTrainData()
    dtest = OneHotM(testques, testans)
    return dtest
