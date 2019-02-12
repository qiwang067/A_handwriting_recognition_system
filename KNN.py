'''
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN) 用于分类的输入向量(测试集)
            dataSet: size m data set of known vectors (NxM) 用于训练的数据(训练集)
            labels: data set labels (1xM vector) 标签向量 分类标签
            k: number of neighbors to use for comparison 最近邻居的数量
            
Output:     最多的分类标签

@author: wq
'''
import numpy as np
import operator
from os import listdir

#KNN分类器，作用就是个inX进行分类
def classify0(inX, dataSet, labels, k):  #计算inX与当前点的距离
    #返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #生成dataSet的行数*1的矩阵(元素有inX构成)
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet 
    sqDiffMat = diffMat**2  #特征相减后平方
    sqDistances = sqDiffMat.sum(axis=1) #行相加  
    distances = sqDistances**0.5 #欧式距离
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):  #选择距离最小的k个点,并计算其中k个点钟标签出现的最大频次
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #排序
        #key=operator.itemgetter(1)根据字典的值进行排序
        #key=operator.itemgetter(0)根据字典的键进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回频次最高的类别,即要分类的类别
    return sortedClassCount[0][0]


  #将图形转换为测试向量  
#将32*32的文本(本质上就是特征值)，按行拼成1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1,1024)) 
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')   #加载训练集
    m = len(trainingFileList) #查看列表的长度
    trainingMat = np.zeros((m,1024))
    #range(m)生成0~m-1的列表
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #去掉每个文件名中的txt
        fileStr = fileNameStr.split('.')[0]     
        #提出每个文件中所表示的数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #训练的矩阵的第i行
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        
    testFileList = listdir('testDigits')        
    errorCount = 0.0    #错误计数
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i] 
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        #传入具体的文件来进行处理
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total accuracy rate is: %f" % (1.0-errorCount/float(mTest)))
handwritingClassTest()
