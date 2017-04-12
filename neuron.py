import sys
import csv
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from optparse import OptionParser

def changeWeights(nodes, rate, gradients, y, attributes):
    nodes_re = nodes[::-1]
    outputs = y[::-1]
    outputs.append(attributes)
    outputs = outputs[1:len(outputs)]
    newNodes = []
    
    level = 0
    for nodeLevel in nodes_re:
        row = 0
        newWeightsLevel = []
        for weights in nodeLevel:
            index = 0
            newWeights = []
            for weight in weights:
                deltaWeight = 0
                if (index < len(outputs[level])):
                    deltaWeight = rate * gradients[level][row] * outputs[level][index]
                else:
                    deltaWeight = rate * gradients[level][row]
                newWeights.append(weight - deltaWeight)
                index += 1
            row += 1
            newWeightsLevel.append(newWeights)
        level += 1
        newNodes.append(newWeightsLevel)

    return newNodes[::-1]

def calGradient(error, y):

    result = error * (y * (1 - y))

    return result

def sumBackProp():
    result = 0
    for i in range(len(inputs)):
        weight = weights[i]
        attr = inputs[i]
        result += weight * attr
    return result

def backPropagation(expectedResult, results, nodes):
    # print(nodes)
    outputs = results[::-1]
    nodesReverse = nodes[::-1]

    gradients = []

    #outputLayer
   
    outputLayerGradients = []
    for output in outputs[0]:
        error = expectedResult - output
        outputLayerGradients.append(calGradient(error, output))

    gradients.append(outputLayerGradients)
    #Hidden Layer
    level = 0
    for output in outputs[1:len(outputs)]:
        hiddenLayerGradients = []
        row = 0
        for y in output:
            arrayX = []
            for i in range(len(nodesReverse[level])):
                x = nodesReverse[level][i][row]
                arrayX.append(x)
            gradient = calGradient(sumOf(arrayX, gradients[level]), y)
            hiddenLayerGradients.append(gradient)
            row += 1
        level += 1
        gradients.append(hiddenLayerGradients)
    return gradients

def calSigmoid(v):
    return 1 / (1 + math.e ** (-1 * v))

def sumOf(weights, inputs):
    result = 0
    for i in range(len(inputs)):
        weight = weights[i]
        attr = inputs[i]
        result += weight * attr
    return result

def findOutput(inputs, weights):
    output = 0
    output += sumOf(weights, inputs)
    output += weights[len(weights) - 1]
    output = calSigmoid(output)
    return output

def runNeuronNetwork(nodes, attributes, expectedResult):
    weights = nodes[0][0]
    outputs = []
    level = 0

    # Input Nodes
    lvl1Output = []
    for weights in nodes[0]:
        output = findOutput(attributes, weights)
        lvl1Output.append(output)
    outputs.append(lvl1Output)

    # Hidden Layer Nodes
    for node in nodes[1:len(nodes)]:
        hiddenLayerOutput = []
        for weights in node:
            hiddenLayerOutput.append(findOutput(outputs[level], weights))
        outputs.append(hiddenLayerOutput)
        level += 1
    
    # All outputs from every nodes
    return outputs

def getAttributes(data):
    attributes = data[0:len(data)-1]
    return attributes

def getExpected(data):
    return data[len(data) - 1]

def convertToFloat(dataset):
    newDataset = []
    for data in dataset:
        obj = []
        for item in data:
            obj.append(float(item))
        newDataset.append(obj)
    return newDataset

def loadNodesWeight(fname):
    nodes = []

    nodesCount = []

    nodesRaw = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if not nodesCount:
                for count in row:
                    nodesCount.append(int(count))
            else:# [[2,3,1],[1,2,3]]
                weights = []
                for weight in row:
                    weights.append(float(weight))
                nodesRaw.append(weights)

    i = 0
    for count in nodesCount:
        level = nodesRaw[i:i+count]
        nodes.append(level)
        i += count

    return nodes, nodesCount

def writeToFile(fname, data, header):
    dataToWrite = []
    dataToWrite.append(header)

    for level in data:
        for node in level:
            dataToWrite.append(node)

    with open(fname, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(dataToWrite)
    print("done write file")

def calError(errors):
    goods = []
    bads = []
    for error in errors:
        if error < 0.5:
            goods.append(error)
        else:
            bads.append(error)
    print("Corrects: " + str(len(goods)))
    print("Wrong: " + str(len(bads)))
    percent = float(len(goods))/float(len(errors))
    print("Percent: " + str( 100 * percent))
    return percent


def dataFromFile(fname):
    dataset = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            item = []
            for attb in row:
                item.append(attb)
            dataset.append(item)
    return dataset

def train(datasetForTrain, nodes, threshold, count):

    count += 1

    avgError = 4

    trainNode = nodes

    errorsToPlot = []

    while avgError > threshold:
       
        count += 1

        randomIndex = random.sample(range(0, len(datasetForTrain)), len(datasetForTrain))

        errors = []

        for i in range(len(datasetForTrain)):

            data = datasetForTrain[randomIndex[i]]

            attributes = getAttributes(data)

            expectedResult = getExpected(data)

            outputs = runNeuronNetwork(trainNode, attributes, expectedResult)

            gradients = backPropagation(expectedResult, outputs, trainNode)

            trainNode = changeWeights(trainNode, learningRate, gradients, outputs, attributes)

            results = outputs[::-1][0]

            # e = 0.5(sqrt(X))
            for result in results:
                outputErrors = math.fabs(result - expectedResult)

            avgOupputErrors = np.average(outputErrors)
            
            errors.append(avgOupputErrors)

        avgError = sum(errors) / float(len(errors))

        # avgError = rootMeanSquare(errors)

        errorsToPlot.append(avgError)

    print('average error', avgError, count)
    return trainNode, errorsToPlot  

def rootMeanSquare(errors):
    return 0.5 * math.sqrt(np.sum(np.square(errors)))

def test(datasetForTest, nodes):
    errors = []
    for data in datasetForTest:

        attributes = getAttributes(data)

        expectedResult = getExpected(data)

        outputs = runNeuronNetwork(nodes, attributes, expectedResult)

        result = outputs[::-1][0][0]

        error = math.fabs(result - expectedResult)

        errors.append(error)

    return calError(errors)

if __name__ == "__main__":

        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                                dest='input',
                                help='filename containing csv',
                                default=None)
        optparser.add_option('-w', '--nodesFile',
                                dest='nodesInput',
                                help='filename containing csv',
                                default=None)
        optparser.add_option('-n', '--learn',
                                dest='learningRate',
                                help='learing rate',
                                default=-0.9,
                                type='float')
        optparser.add_option('-o', '--outputFile',
                                dest='outputWeights',
                                help='filename output containing csv',
                                default='new_weights.csv')
        optparser.add_option('-a', '--action',
                                dest='action',
                                help='train / test',
                                default='train')
        optparser.add_option('-t', '--threshold',
                                dest='threshold',
                                help='minimum error threshold for training',
                                default=1.0,
                                type='float')
        optparser.add_option('-l', '--length',
                                dest='length',
                                help='no. of fold to seperate train and test',
                                default=0.1,
                                type='float')

        (options, args) = optparser.parse_args()
        
        rawData = None
        if options.input is None:
                rawData = sys.stdin
        elif options.input is not None:
                rawData = dataFromFile(options.input)
        else:
                print 'No dataset filename specified, system with exit\n'
                sys.exit('System will exit')
        
        nodes = None
        nodesCount = None
        if options.nodesInput is None:
                nodes = sys.stdin
        elif options.nodesInput is not None:
                nodes, nodesCount = loadNodesWeight(options.nodesInput)
        else:
                print 'No dataset filename specified, system with exit\n'
                sys.exit('System will exit')

        learningRate = options.learningRate

        dataset = convertToFloat(rawData)

        foldPercent = options.length
        length = int(foldPercent * float(len(dataset)))

        accuracy = []
        nodesToWrite = []
        for start in range(0, len(dataset), length):
            print(length)
            datasetForTrain = dataset[start+length:len(dataset)]
            datasetForTrain += dataset[0:start]
            print("train data", len(datasetForTrain))
            datasetForTest = []
            if start+length < len(dataset):
                datasetForTest = dataset[start:start+length]
            else:
                datasetForTest = dataset[start:]
            print("test data", len(datasetForTest))
            print("start training")

            threshold = options.threshold
            # if options.action == 'train':
            newnodes, errorsToPlot = train(datasetForTrain, nodes, threshold, 0)
            
            plt.plot(errorsToPlot)

            nodesToWrite.append(newnodes)
            # elif options.action == 'test':
            accuracy.append(test(datasetForTest, newnodes))
        
        print(accuracy)
        print(np.mean(accuracy))

        count = 1
        for nodesWillWrite in nodesToWrite:
            writeToFile(options.outputWeights + str(count), nodesWillWrite, nodesCount)
            count += 1

        plt.show()