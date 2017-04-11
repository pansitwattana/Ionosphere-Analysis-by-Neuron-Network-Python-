import sys
import csv
import random
from optparse import OptionParser

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

def generateWeight(nodesCount, attributeCount):
    nodes = []

    i = 0
    for count in nodesCount:
        level = []
        for node in range(count):
            print(node)
        nodes.append(level)
        i += count

    return nodes, nodesCount

if __name__ == "__main__":
    optparser.add_option('-o', '--outputFile',
                                dest='outputWeights',
                                help='filename output containing csv',
                                default='genweight.csv')
    
    outputFilename = options.outputWeights

    nodesCount = [4, 3, 1]

    attributeCount = 34

    nodes = generateWeight(nodesCount, attributeCount)

    writeToFile(outputFilename, nodes, nodesCount)