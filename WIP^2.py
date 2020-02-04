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
import os.path

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
test_x = test_set[0]
test_y = test_set[1]
#Ziffern in Nullvektoren mit Eins an der i-ten Stelle (in Dezimalsystem)
train_y_dec = np.zeros([len(train_y),10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1
    
test_y_dec = np.zeros([len(test_y),10])
for i in range(len(test_y)):
    test_y_dec[i][test_y[i]] = 1
  
    
bit = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]    

train_y_bit = np.zeros([len(train_y),4])
for i in range(len(train_y)):
    train_y_bit[i,:] = bit[train_y[i]]
    
test_y_bit = np.zeros([len(test_y),4])
for i in range(len(test_y)):
    test_y_bit[i,:] = bit[test_y[i]]


#Sigmoidfunktion
def sig(x):
    return 1/(1+np.exp(-x))
     
#Ableitung der Sigmoidfunktion
def sigder(x):
    return -np.exp(-x)/(1+np.exp(-x))**2



class NeuralNetwork:
    #Inputs sind Tiefe der jew. Schicht und Groesse der Minibatchs
    def __init__(self, inp, lay, outp, mbsize, schrittweite):
        self.bias      = np.ones([1, mbsize])       #1x10
        
        self.inputOB      = np.zeros([inp, mbsize]) #784x10
        self.input      = np.zeros([inp+1, mbsize]) #785x10
        
        self.layOB       = np.zeros([lay, mbsize])  #30x10
        self.lay       =  np.zeros([lay+1, mbsize]) #31x10
        
        self.output     = np.zeros([outp, mbsize])  #10x10
        
        self.mbsize = mbsize
        self.schrittweite = schrittweite
        
        if os.path.exists("Weights1.txt") == True:
            self.weights1 = np.loadtxt("Weights1.txt")
            self.weights2 = np.loadtxt("Weights2.txt")
        else:
            self.weights1 = np.random.rand(lay,inp+1) #30x785
            self.weights2 = np.random.rand(outp,lay+1) #10x31  
        
        #berechnet den output aus dem aktuellen input
    def feedforward(self):
        self.input = np.concatenate((self.inputOB,self.bias))
        self.layOB = sig(self.weights1.dot(self.input))
        self.lay = np.concatenate((self.layOB,self.bias))
        self.output = sig(self.weights2.dot(self.lay))
       

    def backprop(self, testy):
        #grad1 sind die Ableitungen nach den Gewichten der ersten Matrix
        grad2 = np.zeros(self.weights2.shape)   #30x785
        grad1 = np.zeros(self.weights1.shape)   #10x31
        
        #Hilfsmatrizen
        HM2 = self.weights2.dot(self.lay)       #10x10
        HM1 = self.weights1.dot(self.input)     #30x10
        delta = np.zeros(self.output.shape)     #10x10
        quasidelta = np.zeros(self.lay.shape)   #31x10
              
        for i in range(len(self.output)):
            delta[i,:] = (self.output[i,:]-testy[i,:])*sigder(HM2[i,:])
            for j in range(len(self.lay)):             
                grad2[i][j] = np.sum(delta[i,:]*self.lay[j,:])
          

        for i in range(len(self.lay)-1):
            quasidelta[i,:] = np.sum(delta*self.weights2[:,i],axis =0)#*sigder(HM1[i,:])                
            for j in range(len(self.input)):               
                grad1[i][j] = np.sum(quasidelta[i,:]*self.input[j,:])
         
        #hier werden die Gewichte angepasst    
        self.weights1 = self.weights1-self.schrittweite*grad1
        self.weights2 = self.weights2-self.schrittweite*grad2
        
        np.savetxt("Weights1.txt", self.weights1)
        np.savetxt("Weights2.txt", self.weights2)
        
        
        #bekommt kompletten Trainingsdatensatz
    def train(self, pictures, numbers):
        #schneide pictures und numbers in teile der länge mbsize
   
        for i in range(0,len(pictures),self.mbsize):
      #um bei der Fehlersuche die Laufzeit zu verringern, die Funktionen 
      #nur für zwei minibatches ausführen mit dieser for-schleife:      
        #for i in range(12,10):    
      
            self.inputOB = pictures[:,i:i+self.mbsize]
            testy = numbers[:,i:i+self.mbsize]
            
            self.feedforward()
            self.backprop(testy)
            
    def test_dec(self, pictures, numbers):
        self.bias      = np.ones([1, len(pictures.transpose())]) #1x10000
        self.inputOB = pictures     #784x10000
        
        self.feedforward()
        #für jedes Testdatum ein Eintrag im Vektor
        vec = np.zeros(len(pictures.transpose())) #1x10000
    
        for i in range(len(vec)):
            #Für alle Testdaten: berechnet Index des maximalen Output-Neurons, 
            #speichert Testdateneintrag an dieser Stelle in Vektor
            #(1 wenn richtig erkannt, 0 wenn falsch)
          vec[i] = numbers[np.argmax(self.output,axis = 0)[i],i]
        print(np.sum(vec)*100/len(vec),"Prozent erkannt")
        
    def test_bit(self, pictures, numbers):
        self.bias      = np.ones([1, len(pictures.transpose())]) #1x10000
        self.inputOB = pictures     #784x10000
        
        self.feedforward()
        
        
        
        
network = NeuralNetwork(784, 30, 10, 12, 200) 


network.train(train_x.transpose(),train_y_dec.transpose())

network.test_dec(test_x.transpose(),test_y_dec.transpose())
 


#print("output = ",network.output)