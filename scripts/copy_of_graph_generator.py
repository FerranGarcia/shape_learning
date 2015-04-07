# -*- coding: utf-8 -*-
"""
This script generates different kind of graphs outputs using reference shapes
and demo shapes recovered from log files

@author: ferran
"""

import string
import datetime
import argparse
import glob
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
from shape_learning.shape_modeler import ShapeModeler


parser = argparse.ArgumentParser(description='Plots the learning process');
parser.add_argument('symbols', action="store", nargs='?', default='[a-z]', 
                help='The list of symbols to be analysed. By default, [a-z]') 
parser.add_argument('init_time', action="store", nargs='?',
                help='The initial date for what to visualize it. By default first log');
parser.add_argument('end_time', action="store", nargs='?',
                help='The end date for what to visualize it. By default last log');
parser.add_argument('num_params', action="store", nargs='?', default=5, type=int,
                help='The number of parameters for which to have their effect visualised');
parser.add_argument('log_directory', action="store", nargs='?',
                help='The directory of the log to be analized');
parser.add_argument('dataset_directory', action="store", nargs='?',
                help='The directory of the dataset to find the shape dataset');


"""
Recovers the specified demo shapes provided by the children 
and stored in all log files based on specific period of time
""" 
def getLogShapes(logDir, initDate, endDate, regexp, num_params):
    
    initDate = datetime.datetime.strptime(initDate, "%d-%m-%Y")
    endDate = datetime.datetime.strptime(endDate, "%d-%m-%Y")        
    
    logset = glob.glob(logDir + '/*.log')
    logset.sort()
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    res = re.findall(alphabet, alphabetList)

    fields = {key: [] for key in res}

    for logfile in logset:
        print('Evaluating date ... ' + logfile)
        date = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", logfile)
        logDate = datetime.datetime.strptime(date.group(1), "%d-%m-%Y")  
        
        if (logDate >= initDate and logDate <= endDate): 
            print('Opening ... ' + logfile)                  
            #Process file
            with open(logfile, 'r') as inF:
                lineNumb = 1
                for line in inF:
                    found = re.search(r'...new\b.*?\bdemonstration\b', line)
                    if found is not None:                    
                        leter = found.group(0)[0]
                        if alphabet.match(leter):
                            print('Match found! ' + leter + ' in line ' + str(lineNumb))
                            strShape = line.partition('[')[-1].rpartition('].')[0]
                                                        
                            shape = strShape.split(', ')
                            shape = np.array(map(float, shape))

                            #Get the same number of parameters for all
                            if len(shape) > num_params: 
                                fields[leter].append(shape[0:num_params])
                            else:
                                fields[leter].append(shape)
                                
                    lineNumb = lineNumb + 1
        else:
            raise Exception("No log files among the entry dates")

    return fields
    

"""
Same function as getLogShapes but only returning the sahpes of a single Log
"""    
def testfunction(logfile, initDate, endDate, num_params, regexp):
     
    initDate = datetime.datetime.strptime(initDate, "%d-%m-%Y")
    endDate = datetime.datetime.strptime(endDate, "%d-%m-%Y")        
    
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    res = re.findall(alphabet, alphabetList)

    fields = {key: [0] * num_params for key in res}
    
    
    print('Evaluating date ... ' + logfile)
    date = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", logfile)
    logDate = datetime.datetime.strptime(date.group(1), "%d-%m-%Y")  
    
    if (logDate >= initDate and logDate <= endDate): 
        print('Opening ... ' + logfile)                  
        #Process file
        with open(logfile, 'r') as inF:
            lineNumb = 1
            for line in inF:
                found = re.search(r'...new\b.*?\bdemonstration\b', line)
                if found is not None:                    
                    leter = found.group(0)[0]
                    if alphabet.match(leter):
                        print('Match found! ' + leter + ' in line ' + str(lineNumb))
                        strShape = line.partition('[')[-1].rpartition('].')[0]
                                                    
                        shape = strShape.split(', ')
                        shape = np.array(map(float, shape))

                        #Get the same number of parameters for all
                        if len(shape) > num_params: 
                            fields[leter].append(shape[0:num_params])
                        else:
                            fields[leter].append(shape)
  
                lineNumb = lineNumb + 1
    else:
        raise Exception("No log files among the entry dates")    
    
    mean_vect = {key: [] for key in res}
    
    for leter in fields:
        num_occurences = len(fields[leter])
        it  = np.zeros(num_params)
        for i in range(0,num_occurences):
            it = fields[leter][i] + it
        
        mean = it/num_occurences
        mean_vect[leter].append(mean)
    print mean_vect
    
    return mean_vect
    
    
"""
Returns a list of ShapeModeler objects based on a regexp
"""         
def prepareShapesModel(datasetDirectory, regexp, num_params):

    shapes = {}

    nameFilter = re.compile(regexp)    
    datasets = glob.glob(datasetDirectory + '/*.dat')
    
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]
        
        if nameFilter.match(name):
            shapeModeler = ShapeModeler(init_filename = dataset, num_principle_components=num_params)
            shapes[name] = shapeModeler
            
    return shapes


"""
Projects the reference shape contained in ShapeModeler into the eigenspace
defined by the dataset for each of the shapes
"""
def projectRefShape(shapes, num_params):
    
    projRefShapes = np.zeros(shape=(1,num_params))
    
    for shape in shapes:
        refShape = np.reshape(shapes[shape].dataMat[0], (-1, 1))
        print refShape
        projRefShape = shapes[shape].decomposeShape(refShape)
        projRefShapes = np.vstack((projRefShapes,projRefShape[0][0:].T))
        #projRefShapes.append(projRefShape[0][0:].T)
    
    projRefShapes = np.delete(projRefShapes, 0, 0)
    return projRefShapes


"""
Calculates the distance between the demo shapes and its references
returned in a date/shape sorted structure
"""
def calcDistance(projRefShapes, projDemoShapes, regexp):
    
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    res = re.findall(alphabet, alphabetList)
    distances = {key: [] for key in res}
    
    i = 0
    
    for res in projDemoShapes:
        ref = projRefShapes[i]
        leter = projDemoShapes[res]
        for sample in range(0,len(leter)):
            demo = leter[sample]
                 
            dist = np.linalg.norm(demo-ref)         
            distances[res].append(dist)
            
        i = i+1
        
    return distances

   
"""
Receives the 'dist' structure and plots for each type of shape, a graph
"""
def plotDistShape(datasetDirectory, regexp, num_params, initDate, endDate):
    
    from pylab import *
    
    pltDistShape = plt.figure(1)
    refShapes = prepareShapesModel(datasetDirectory, regexp, num_params)
    projRefShapes = projectRefShape(refShapes, num_params)
    projDemoShapes = getLogShapes(logDir, initDate, endDate, regexp, num_params)
    distances = calcDistance(projRefShapes, projDemoShapes, regexp)
    
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    res = re.findall(alphabet, alphabetList)
    
    subplots_adjust(hspace=0.000)
    n=len(res)
    
#    for i,v in enumerate(xrange(number_of_subplots)):
#        v = v+1
#        
#        ax1 = subplot(number_of_subplots,1,v)            
#        ax1.plot(distances[res[i]])
#        plt.xlabel('Time progression')
#        plt.ylabel('Shape: ' + res[i] + ' wrt. Ref')

    a = np.floor(n**0.5).astype(int)
    b = np.ceil(1.*n/a).astype(int)
    print "a\t=\t%d\nb\t=\t%d\na*b\t=\t%d\nn\t=\t%d" % (a,b,a*b,n)
    fig = plt.figure(1, figsize=(2.*a,2.*b))
    for i in range(1,n+1):
        ax = fig.add_subplot(b,a,i)
        ax.plot(distances[res[i-1]])
        ax.set_title('Shape: ' + res[i-1])
        plt.xlabel('Time progression')
        plt.ylabel('dist. wrt. Ref')
    fig.set_tight_layout(True)            
       
    return fig
    

"""
Plots the distances of the shapes contained in each log
"""
def plotAlphabetLog(datasetDirectory, num_params, initDate, endDate, regexp):
    
    from pylab import *
    import collections
    
    pltAlphabetLog = plt.figure(2)
    refShapes = prepareShapesModel(datasetDirectory, regexp, num_params)
    projRefShapes = projectRefShape(refShapes, num_params)
    
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    res = re.findall(alphabet, alphabetList)
    
    logset = glob.glob(logDir + '/*.log')
    logset.sort()
    number_of_subplots = len(logset)    
    
    v = 0 
    i = 0
    
    for logfile in logset:
        projDemoShapes = testfunction(logfile, initDate, endDate, num_params, regexp)
        distances = calcDistance(projRefShapes, projDemoShapes, regexp)
        sort_distances = collections.OrderedDict(sorted(distances.items()))

        num_of_leters = len(sort_distances.keys())
        print sort_distances
        v = v+1
        
        ax1 = subplot(number_of_subplots,1,v)
        ind = np.arange(0,num_of_leters,1)
        plt.xticks(ind, sort_distances.keys())
        ax1.plot(sort_distances.values())
        
        plt.xlabel('Shape')
        plt.ylabel('dist wrt. Ref')
        i = i+1
    
    return pltAlphabetLog
    
     
     
shapes = None;
mainPlot = None;
fig = None;

if __name__ == "__main__":
    
    #parse arguments
    args = parser.parse_args()
    datasetDirectory = args.dataset_directory
    logDir = args.log_directory
    regexp = args.symbols
    initDate = args.init_time
    endDate = args.end_time
    num_params = args.num_params

    if(not initDate):
        initDate = '01-01-2000';

    if(not endDate):   
        endDate = '01-01-2100';
        
    if(not datasetDirectory):
        import inspect
        fileName = inspect.getsourcefile(ShapeModeler);
        installDirectory = fileName.split('/lib')[0];
        datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children';  

    if(not logDir):
        logDir = '../logs';  

    
    pltDistShape = plotDistShape(datasetDirectory, regexp, num_params, initDate, endDate)    
    pltDistShape.show()
    
    pltAlphabetLog = plotAlphabetLog(datasetDirectory, num_params, initDate, endDate, regexp)    
    pltAlphabetLog.show()
       
    raw_input()