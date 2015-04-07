# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:18:36 2015

@author: ferran
"""

"""
We wish to find the the best pose and shape parameters to match a model
instance X to new set of image points Y:
The expression to minimize is: |Y-T(x+Φb)|² where T is: T(Xt,Yt,Θ)
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
from numpy.linalg import inv


parser = argparse.ArgumentParser(description='Plots the learning process');
parser.add_argument('symbols', action="store", nargs='?', default='a', 
                help='The list of symbols to be analysed. By default, a')
parser.add_argument('dataset_directory', action="store", nargs='?',
                help='The directory of the dataset to find the shape dataset');
parser.add_argument('num_params', action="store", nargs='?', default=5, type=int,
                help='The number of parameters for which to have their effect visualised');
                

from recordtype import recordtype  #for mutable namedtuple (dict might also work)

SettingsStruct = recordtype('SettingsStruct',
                            ['shape',               #Initial shape captured from tablet consisting on 70 points - Y
                             'params',              #Parameters that represent the shape - b
                             'model_instance'])     #instance model - x
                             
                             
#TODO: Generate only for one                            
def generateEigenSpace(datasetDirectory, num_params):

    eigenSpace = {}

    nameFilter = re.compile(regexp)    
    datasets = glob.glob(datasetDirectory + '/*.dat')
    
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]

        if nameFilter.match(name):
            shapeModeler = ShapeModeler(init_filename = dataset, num_principle_components=num_params)
            eigenSpace[name] = shapeModeler
           
    return eigenSpace


def fitModel_to_NewPoints(self, eigenSpace, shape, num_params):
    initializeShape(self, shape, num_params)
    generateModelInstance(self, eigenSpace)
    poseParam = findPoseParameters(self, nb_points)      
    #invPoseParam = invertPoseParameters(poseParam)
    proj_y = project_to_coorFrame(self, poseParam, nb_points)
    proj_y_tanPlane = project_to_tangent(eigenSpace, proj_y)


def initializeShape(self, shape, num_params):
    self.shape = np.array(shape)
    self.params = np.zeros(num_params)


def generateModelInstance(self, eigenSpace):    
    #TODO: Not to hardcode the var 'a'
    self.model_instance = eigenSpace['a'].makeShape(self.params)   


def findPoseParameters(self, nb_points):
    #Getting a, b, tx and ty parameters in order to compose the matrix
    Sxx_p = (np.sum(self.model_instance[0:nb_points] * self.shape[0:nb_points]))/nb_points
    Sxy_p = (np.sum(self.model_instance[0:nb_points] * self.shape[nb_points:]))/nb_points
    Syx_p = (np.sum(self.model_instance[nb_points:] * self.shape[0:nb_points]))/nb_points
    Syy_p = (np.sum(self.model_instance[nb_points:] * self.shape[nb_points:]))/nb_points
    Sxx = (np.sum(self.model_instance[0:nb_points] * self.model_instance[0:nb_points]))/nb_points
    Syy = (np.sum(self.model_instance[nb_points:] * self.model_instance[nb_points:]))/nb_points
    
    a = (Sxx_p + Syy_p)/(Sxx + Syy)
    b = (Sxy_p - Syx_p)/(Sxx + Syy)
    
    tx = (np.sum(self.shape[0:nb_points]))/nb_points    
    ty = (np.sum(self.shape[nb_points:]))/nb_points
    s = a*a + b*b
    theta = np.arctan(b/a) 
       
    return tx,ty,s,theta
    

#def invertPoseParameters(poseParam):
#    
#    invPoseParam[0] = -poseParam[0]    #Xt
#    invPoseParam[1] = -poseParam[1]    #Yt 
#    invPoseParam[2] = 1/poseParam[2]   #1/s
#    invPoseParam[3] = -poseParam[3]    #-theta
#
#    return invPoseParam

   
def project_to_coorFrame(self, poseParam, nb_points):
    
    #Compose the matrix T and calculate its T⁻
    T = np.matrix([[poseParam[4], -poseParam[5]],
                   [poseParam[5],  poseParam[5]]])

    T_inv = np.linalg.inv(T)
    
    proj_y = np.array([])
    
    for i in range(nb_points):     
        Y = np.array([[shape[i], shape[i+nb_points]]]).T
        y = np.dot(T_inv,Y)
        np.append(proj_y, y)
        
    return proj_y

def project_to_tangent(eigenSpace, proj_y):
    
    
    
    return proj_y_tanPlane
    
    
def update():

    decomposeShape

shapes = None;
mainPlot = None;
fig = None;

if __name__ == "__main__":
    
    #parse arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_directory
    regexp = args.symbols
    num_params = args.num_params
        
    
    import inspect
    fileName = inspect.getsourcefile(ShapeModeler);
    installDirectory = fileName.split('/lib')[0];
        
    if(not dataset_dir):
        datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children';
       

    eigenSpace = generateEigenSpace(datasetDirectory, num_params)
    fitModel_to_NewPoints(eigenSpace, shape, num_params)
       
    raw_input()