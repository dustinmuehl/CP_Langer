#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 18:21:17 2020

@author: verisa
"""

import numpy as np
import gzip # Entzipper für das Auslesen der Testdaten
import pickle #Einlesen der Testdaten in einen Array
import os.path


#Selbstdefinierte Funktionen

#Sigmoidfunktion
def sig(x):
    return 1/(1+np.exp(-x))
     
#Ableitung der Sigmoidfunktion
def sigder(x):
    return -np.exp(-x)/(1+np.exp(-x))**2



#Beginn Klasse

class NeuralNetwork:
    #Inputs sind Tiefe der jew. Schicht, Groesse der Minibatchs und Schrittweite für Gradientenverfahren
    def __init__(self, inp, lay, outp, mbsize, schrittweite):
        self.bias      = np.ones([1, mbsize])       #1x10
        #input (ohne und mit Bias)
        self.inputOB      = np.zeros([inp, mbsize]) #784x10
        self.input      = np.zeros([inp+1, mbsize]) #785x10
        #hidden layer (ohne und mit Bias)
        self.layOB       = np.zeros([lay, mbsize])  #30x10
        self.lay       =  np.zeros([lay+1, mbsize]) #31x10
        
        self.output     = np.zeros([outp, mbsize])  #10x10
        
        self.mbsize = mbsize
        self.schrittweite = schrittweite
        
  #Auskommentiertes hier kann verwendet werden, wenn Gewichte in Textdatei gespeichert wurden      
#        if os.path.exists("Weights1.txt") == True:
#            self.weights1 = np.loadtxt("Weights1.txt")
#            self.weights2 = np.loadtxt("Weights2.txt")
#        else:
  
        #weights1 sind die Gewichte, die vom Input zum Hidden layer verlaufen, weights2 die zwischen hidden layer und output  
        self.weights1 = np.random.rand(lay,inp+1)*10 #30x785
        self.weights2 = np.random.rand(outp,lay+1)*10 #10x31  
        
    #berechnet den output aus dem aktuellen input
    def feedforward(self):
        
        self.input = np.concatenate((self.inputOB,self.bias))
        self.layOB = sig(self.weights1.dot(self.input))
        self.lay = np.concatenate((self.layOB,self.bias))
        self.output = sig(self.weights2.dot(self.lay))
       

    def backprop(self, testy):
        #grad1 sind die Ableitungen nach den Gewichten der ersten Matrix
        #grad2 nach denen der zweiten Matrix
        grad2 = np.zeros(self.weights2.shape)   #30x785
        grad1 = np.zeros(self.weights1.shape)   #10x31
        
        #Hilfsmatrizen:
        
        #diese werden hier berechnet, um Aufwand zu sparen
        HM2 = self.weights2.dot(self.lay)       #10x10
        HM1 = self.weights1.dot(self.input)     #30x10
        
        #entsprechen den Deltas aus dem Skript
        delta2 = np.zeros(self.output.shape)     #10x10
        delta1 = np.zeros(self.lay.shape)   #31x10
        
        #Berechnung des Gradienten für die zweite Gewichtmatrix
        for i in range(len(self.output)):
            delta2[i,:] = (self.output[i,:]-testy[i,:])*sigder(HM2[i,:])
            for j in range(len(self.lay)):             
                grad2[i][j] = np.sum(delta2[i,:]*self.lay[j,:])
          
        #Berechnung des Gradienten für die erste Gewichtmatrix   
        for i in range(len(self.lay)-1):        
            delta1[i,:] = np.sum((delta2.transpose()*self.weights2[:,i]).transpose(),axis =0)*sigder(HM1[i,:])                
            for j in range(len(self.input)):               
                grad1[i][j] = np.sum(delta1[i,:]*self.input[j,:])
         
        #hier werden die Gewichte angepasst    
        self.weights1 = self.weights1-self.schrittweite*grad1
        self.weights2 = self.weights2-self.schrittweite*grad2
  
#Auskommentiertes hier dient dem Speichern der Gewichte in Textdatei      
        #np.savetxt("Weights1.txt", self.weights1)
        #np.savetxt("Weights2.txt", self.weights2)
        
        
        #funktion zum trainieren des Netzwerkes
        #bekommt kompletten Trainingsdatensatz, pictures enthält x-Werte in einer 784x50000 Matrix, 
        #numbers die y-Werte in einer 10x50000 Matrix
    def train(self, pictures, numbers):
        self.bias = np.ones([1,self.mbsize])
        #die Schleife läuft in Schritten der Länge der Minibatches einmal über den Trainingsdatensatz
        for i in range(0,len(pictures),self.mbsize):
      #um bei der Fehlersuche die Laufzeit zu verringern, die Funktionen 
      #nur für zwei minibatches ausführen mit dieser for-schleife:      
        #for i in range(12,10):    
      
            #schneide pictures und numbers in teile der länge mbsize
            #x-werte werden sofort als neuer input festgelegt
            self.inputOB = pictures[:,i:i+self.mbsize]
            testy = numbers[:,i:i+self.mbsize]
            
            self.feedforward()
            self.backprop(testy)
         
       #Funktion zum Testen, wie gut das Netzwerk gelernt hat
       #bekommt kompletten Testdatensatz, pictures enthält x-Werte in einer 784x10000 Matrix, 
        #numbers die y-Werte in einer 10x10000 Matrix
    def test_dec(self, pictures, numbers):
        #der Bias muss angepasst werden, da hier keine Minibatches mehr verwendet werden
        self.bias      = np.ones([1, len(pictures.transpose())]) #1x10000
        #Alle x-werte der Testdaten werden gleichzeitig als Input gesetzt
        self.inputOB = pictures     #784x10000
        
        self.feedforward()
        
        #für jedes Testdatum ein Eintrag im Vektor
        vec = np.zeros(len(pictures.transpose())) #1x10000
    
        for i in range(len(vec)):
            #Für alle Testdaten: berechnet Index des maximalen Output-Neurons, 
            #speichert Testdateneintrag an dieser Stelle in Vektor
            #(1 wenn richtig erkannt, 0 wenn falsch)
          vec[i] = numbers[np.argmax(self.output,axis = 0)[i],i]
        #der Vektor wird aufsummiert, dadurch wird gezählt, wieviele Zahlen richtig erkannt wurden. 
        #Anzahl wird in Prozent umgerechnet und in der Konsole ausgegeben
        print(np.sum(vec)*100/len(vec),"Prozent erkannt")
        
        #Diese Testfunktion ist noch nicht fertiggestellt, soll später verwendet werden, 
        #um das Netzwerk zu testen, wenn der Output aus vier Neuronen besteht
    def test_bit(self, pictures, numbers):
        self.bias      = np.ones([1, len(pictures.transpose())]) #1x10000
        self.inputOB = pictures     #784x10000        
        self.feedforward()
        
        #Output wird gerundet (auf 0 oder 1)
        gerundet = np.around(self.output) #4x10000
        vec = np.zeros(len(pictures.transpose())) #1x10000
        
        #vergleicht für alle Testdaten, ob gerundeter Output mit korrektem Vektor übereinstimmt, schreibt je nachdem 0 oder 1 in den Vektor
        for i in range(len(vec)):
            if((gerundet[:,i]==numbers[:,i]).all()):
                vec[i]=1
            else:
                vec[i]=0
        #zählt richtig erkannte Zahlen zusammen, rechnet in Prozent um
        print(np.sum(vec)*100/len(vec),"Prozent erkannt")
#Ende der Klasse   


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
#Ziffern werden in Nullvektoren mit Eins an der i-ten Stelle (in Dezimalsystem) umgewandelt
train_y_dec = np.zeros([len(train_y),10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1
#analog für die Testdaten    
test_y_dec = np.zeros([len(test_y),10])
for i in range(len(test_y)):
    test_y_dec[i][test_y[i]] = 1
  

#Ziffern werden in Vektoren entsprechend der Bit-Schreibweise umgewandelt    
bit = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]]    
train_y_bit = np.zeros([len(train_y),4])
for i in range(len(train_y)):
    train_y_bit[i,:] = bit[train_y[i]]
#analog für die Testdaten    
test_y_bit = np.zeros([len(test_y),4])
for i in range(len(test_y)):
    test_y_bit[i,:] = bit[test_y[i]]        



#Netzwerk wird erstellt        
network = NeuralNetwork(784, 30, 10, 10, 3) 
network2 = NeuralNetwork(784,30,4,10,3)


#ein Schleifendurchlauf entspricht einer Epoche
for i in range(20):
    #Um nicht in jeder epoche dieselben Minibatches zu verwenden, werden die Trainingsdaten gemischt
    #x- und y-Werte müssen dabei zuerst zu einer Matrix verklebt, dann gemischt und dann wieder getrennt werden
    trainges = np.concatenate((train_x.transpose(),train_y_dec.transpose())).transpose()
    np.random.shuffle(trainges)
    shuffle_x = trainges[:,0:784]
    shuffle_y = trainges[:,784:794]
    
    network.train(shuffle_x.transpose(),shuffle_y.transpose())
    network.test_dec(test_x.transpose(),test_y_dec.transpose())
 
for i in range(20):
    #Um nicht in jeder epoche dieselben Minibatches zu verwenden, werden die Trainingsdaten gemischt
    #x- und y-Werte müssen dabei zuerst zu einer Matrix verklebt, dann gemischt und dann wieder getrennt werden
    trainges = np.concatenate((train_x.transpose(),train_y_bit.transpose())).transpose()
    np.random.shuffle(trainges)
    shuffle_x = trainges[:,0:784]
    shuffle_y = trainges[:,784:794]
    
    network2.train(shuffle_x.transpose(),shuffle_y.transpose())
    network2.test_bit(test_x.transpose(),test_y_bit.transpose())
