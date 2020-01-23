#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 06:08:09 2020

@author: verisa
"""
import math
import numpy as np
import gzip # Entzipper für das Auslesen der Testdaten
import pickle #Einlesen der Testdaten in einen Array

#Testdaten einlesen
with gzip.open('mnist.pkl.gz', 'rb') as f: #Öffnet das Trainingsdatenset
        train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
        #train_set[0] ist eine 50.0000 x 784 Matrix --> Pixel Daten
        #train_set[1] ist eine 50.0000 x 1 Matrix --> Wahre Werte
        #encoding='iso-8859-1' wird benötigt, da ab python 3 ein anderer 
        #Kodierungsstandard Verwendung findet und das Trainingsdatenset in 
        #Python 2 geschrieben wurde 

train_x = train_set[0] #Pixeldatenset  
train_y = train_set[1] #Ziffern in Computerdarstellung
#Ziffern in Nullvektoren mit Eins an der i-ten Stelle
train_y_dec = np.zeros([len(train_y),10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1
   

#Sigmoidfunktion
def sig(x):
    return 1/(1+np.exp(-x))
     
#Ableitung der Sigmoidfunktion
def sigder(x):
    return -np.exp(-x)/(1+np.exp(-x))**2



class NeuralNetwork:
    #Inputs sind Tiefe der jew. Schicht und Groesse der Minibatchs
    def __init__(self, inp, lay1, outp, mbsize, schrittweite):
        self.input      = np.zeros([inp, mbsize])
        self.lay1       = np.zeros([lay1, mbsize])
        self.output     = np.zeros([outp, mbsize])
        
        self.mbsize = mbsize
        self.schrittweite = schrittweite
        
        self.weights1   = np.random.rand(lay1,inp)   
        self.weights2   = np.random.rand(outp,lay1)   
    
        #berechnet den output aus dem aktuellen input
    def feedforward(self):
        self.lay1 = sig(self.weights1.dot(self.input))
        self.output = sig(self.weights2.dot(self.lay1))
       

    def backprop(self, testy):
        #grad1 sind die Ableitungen nach den Gewichten der ersten Matrix
        grad2 = np.zeros(self.weights2.shape)
        grad1 = np.zeros(self.weights1.shape)
        
        #Hilfsmatrizen
        HM2 = self.weights2.dot(self.lay1)
        HM1 = self.weights1.dot(self.input)
 
#kann sein, dass bei den Formeln für die Ableitungen irgendwo ein Fehler ist       
        for i in range(len(self.output)):
            for j in range(len(self.lay1)):
                grad2[i][j] = np.sum(self.output[i,:]-testy[i,:])*np.sum(sigder(HM2[i,:]))*np.sum(self.lay1[j,:])
                
        for i in range(len(self.lay1)):
            for j in range(len(self.input)):
                grad1[i][j] = np.sum(self.output-testy)*np.sum(sigder(HM2))*np.sum(self.weights2[:,i])*np.sum(sigder(HM1[i,:]))*np.sum(self.input[j,:])
         
        #hier werden die Gewichte angepasst    
        self.weights1 = self.weights1-self.schrittweite*grad1
        self.weights2 = self.weights2-self.schrittweite*grad2
        
        #bekommt kompletten Trainingsdatensatz
    def train(self, pictures, numbers):
        #schneide pictures und numbers in teile der länge mbsize
    
        for i in range(0,len(pictures),self.mbsize):
      #um bei der Fehlersuche die Laufzeit zu verringern, die Funktionen 
      #nur für zwei minibatches ausführen mit dieser for-schleife:      
        #for i in range(0,12,10):  
 #irgendwas stimmt mit testy nicht           
            self.input = pictures[:,i:i+self.mbsize]
            testy = numbers[:,i:i+self.mbsize]
            
            self.feedforward()
            self.backprop(testy)
            

       
network = NeuralNetwork(784, 30, 10, 10, 0.5) 
#print(network.weights1)

network.train(train_x.transpose(),train_y_dec.transpose())
    
#network.input = train_x.transpose()[:,0]
#network.feedforward()
#print(network.input)
#print(network.weights1)
#print("output = ",network.output)