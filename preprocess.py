import csv
import sys
from optparse import OptionParser

def writeToFile(fname, data):
    dataToWrite = []
    dataToWrite = data

    with open(fname, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(data)
    print("done write file")

def convertToFloat(dataset):
    newDataset = []
    for data in dataset:
        obj = []
        for item in data:
            obj.append(float(item))
        newDataset.append(obj)
    return newDataset

def removeSecondCol(dataset):
    newDataset = []
    for data in dataset:
        data.pop(1)
        newDataset.append(data)
    return newDataset
        

def ordinalOutput(dataset):
    newDataset = []

    for data in dataset:
        newData = data
        lastElement = len(data) - 1
        d = data[lastElement]
        if (d == 'g'):
            newData[lastElement] = 1
        else:
            newData[lastElement] = -1
        newDataset.append(newData)
            
    return newDataset

def preprocess(dataset):
    newDataset = ordinalOutput(dataset)
    newDataset = removeSecondCol(newDataset)
    newDataset = convertToFloat(newDataset)
    return newDataset

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

        print('sdfasdf')
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                                dest='input',
                                help='filename containing csv',
                                default=None)
        optparser.add_option('-o', '--outputFile',
                                dest='output',
                                help='filename output containing csv',
                                default='proprocess.csv')

        (options, args) = optparser.parse_args()
        
        print('2')
        rawData = None
        if options.input is None:
                rawData = sys.stdin
        elif options.input is not None:
                rawData = dataFromFile(options.input)
        else:
                print 'No dataset filename specified, system with exit\n'
                sys.exit('System will exit')

        dataset = preprocess(rawData)
        
        writeToFile(options.output, dataset)