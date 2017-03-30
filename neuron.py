import sys
import csv
import math
from optparse import OptionParser

def changeWeights(nodes, rate, gradients, y, attributes):
    print("Start Change Weights, Rate: " + str(rate))
    nodes_re = nodes[::-1]
    print("outputs")
    outputs = y[::-1]
    outputs.append(attributes)
    outputs = outputs[1:len(outputs)]
    print(outputs)
    print("gradients")
    print(gradients)
    newNodes = []
    
    print("Weights")
    print(nodes)
    level = 0
    for nodeLevel in nodes_re:
        row = 0
        newWeightsLevel = []
        for weights in nodeLevel:
            index = 0
            newWeights = []
            for weight in weights:
                print(weight)
                print(gradients[level][row])
                deltaWeight = 0
                print("Level+row+index", level, row, index, len(outputs[level]))
                if (index < len(outputs[level])):
                    print(outputs[level][index])
                    deltaWeight = rate * gradients[level][row] * outputs[level][index]
                else:
                    deltaWeight = rate * gradients[level][row]
                print("new weight of ", weight ,weight - deltaWeight)
                print('-')
                newWeights.append(weight - deltaWeight)
                index += 1
            row += 1
            newWeightsLevel.append(newWeights)
        level += 1
        newNodes.append(newWeightsLevel)
    
    print("final")
    print(newNodes[::-1])
    print(nodes)

    return newNodes

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
    print(nodes)
    outputs = results[::-1]
    nodesReverse = nodes[::-1]

    error = expectedResult - outputs[0][0]
    
    gradients = []

    #outputLayer
    gradients.append([calGradient(error, outputs[0][0])])

    #Hidden Layer
    level = 0
    print(outputs)
    print("00000")
    for output in outputs[1:len(outputs)]:
        print(output)
        hiddenLayerGradients = []
        print("--------------")
        row = 0
        for y in output:
            print("y = " + str(y))
            
            arrayX = []
            for i in range(len(nodesReverse[level])):
                x = nodesReverse[level][i][row]
                arrayX.append(x)
                print("x = ", x)
            print("gradient = ", gradients[level])
            print("summation = ", sumOf(arrayX, gradients[level]))
            gradient = calGradient(sumOf(arrayX, gradients[level]), y)
            hiddenLayerGradients.append(gradient)
            row += 1
        level += 1
        gradients.append(hiddenLayerGradients)

    print("gradients--------------")
    print(gradients)

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

    #Input nodes
    lvl1Output = []
    for weights in nodes[0]:
        output = findOutput(attributes, weights)
        lvl1Output.append(output)
    outputs.append(lvl1Output)

    #Hidden Layer
    for node in nodes[1:len(nodes)]:
        hiddenLayerOutput = []
        for weights in node:
            hiddenLayerOutput.append(findOutput(outputs[level], weights))
        outputs.append(hiddenLayerOutput)
        level += 1

    return outputs

def PickOneFrom(dataset):
    data = dataset[0]

    return data

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


# def removeSecondCol(dataset):
#     newDataset = []
#     for data in dataset:
#         data.pop(1)
#         newDataset.append(data)
#     return newDataset
        

# def ordinalOutput(dataset):
#     newDataset = []

#     for data in dataset:
#         newData = data
#         lastElement = len(data) - 1
#         d = data[lastElement]
#         if (d == 'g'):
#             newData[lastElement] = 1
#         else:
#             newData[lastElement] = -1
#         newDataset.append(newData)
            
#     return newDataset

# def preprocess(dataset):
#     newDataset = ordinalOutput(dataset)
#     newDataset = removeSecondCol(newDataset)
#     newDataset = convertToFloat(newDataset)
#     return newDataset

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
            else:
                weights = []
                for weight in row:
                    weights.append(float(weight))
                nodesRaw.append(weights)

    i = 0
    for count in nodesCount:
        level = nodesRaw[i:i+count]
        nodes.append(level)
        i += count

    return nodes

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
        if options.nodesInput is None:
                nodes = sys.stdin
        elif options.nodesInput is not None:
                nodes = loadNodesWeight(options.nodesInput)
        else:
                print 'No dataset filename specified, system with exit\n'
                sys.exit('System will exit')
        print("init", nodes)
        learningRate = options.learningRate

        dataset = convertToFloat(rawData)
        
        data = PickOneFrom(dataset)

        attributes = getAttributes(data)

        expectedResult = getExpected(data)

        outputs = runNeuronNetwork(nodes, attributes, expectedResult)

        gradients = backPropagation(expectedResult, outputs, nodes)

        newNodes = changeWeights(nodes, learningRate, gradients, outputs, attributes)