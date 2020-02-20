# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:58:22 2020

@author: Verisa
"""

import NeuronalesNetz as net
import Hyperparameterstudie as hyp
import numpy as np
from random import randint
import gzip # Entzipper fÃ¼r das Auslesen der Testdaten
import pickle #Einlesen der Testdaten in einen Array



#Testdaten auslesen

with gzip.open('mnist.pkl.gz', 'rb') as f: #Ãffnet das Trainingsdatenset
        train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
        #train_set[0] ist eine 50.0000 x 784 Matrix --> Pixel Daten
        #train_set[1] ist eine 50.0000 x 1 Matrix --> Wahre Werte
        #encoding='iso-8859-1' wird benÃ¶tigt, da ab python 3 ein anderer 
        #Kodierungsstandard Verwendung findet und das Trainingsdatenset in 
        #Python 2 geschrieben wurde         
        
        
train_x = train_set[0] #Pixeldatenset  
train_y = train_set[1] #Ziffern in Computerdarstellung
test_x = test_set[0]
test_y = test_set[1]



#Ã¼berfÃ¼hre Ziffern in Vektoren der passenden Form
train_y_dec = np.zeros([len(train_y),10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1
    
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
    




#Netzwerkvergleich 10 vs 4 Outputs

n=net.NeuralNetwork([784,30,10],0.01)
nbit = net.NeuralNetwork([784,30,4],0.01)

m=1
print("Training über",m,"Epochen")
for i in range(m):
    nbit.train(train_x.T ,train_y_bit.T , 10, 5000, 0.3)
    n.train(train_x.T ,train_y_dec.T , 10, 5000, 0.3)
    
    print("Epoche:",i)
    print(n.test1(test_x.T, test_y_dec.T),"% Erfolg (dec)")
    print(nbit.test1(test_x.T, test_y_bit.T),"% Erfolg (bit)")
    print(n.test2(test_x.T, test_y.T),"/",test_x.shape[0],"erkannt (dec)")
    print(nbit.test3(test_x.T, test_y_bit),"/",test_x.shape[0],"erkannt (bit)")
    print("")

print("Stichprobe:")
j=randint(1,10000)
n.erkenne(test_x[j,:],test_y[j])
for i in range(10):
    print(np.around(n.V[n.layercount-1][i],decimals=2),"%   ",i)
nbit.erkenne(test_x[j,:],test_y[j])
for i in range(4):
    print(np.around(nbit.V[nbit.layercount-1][i],decimals=2),"*",2**(3-i))

print("")




#Hyperparameterstudie   (nur für Netz mit 1 Hidden Layer, Dezimal-Output) 
print("Hyperparameterstudie:")  
e = 2  # Anzahl Epochen, zb e=4 damit Epoche 0,1,2,3 berechnet wird
S = [1,2]  # Schrittweite, Optimum 2.5 ?
B = [10,20] # Batch-Size, Optimum ~15
H = [30]  # Hidden-Layer Size, Optimum bei ~50 aber bei 30 vgl und schneller
a =2 #Anzahl an Durchläufen
hyp.txt_weight(H)   #auskommentieren falls Gewichte erstellt wurden und H nicht mehr verändert wird.
print("")
pMat = hyp.hypP(S, B, H, e)
for l in range(a-1):
    pMat_2 = hyp.hypP(S, B, H, e)
    pMat = pMat + pMat_2

pMat_name = "pMatrix_"+"S_"+str(S)+"_B_"+str(B)+"_H_"+str(H)+".txt"

print("")
print(["Epoche","Schrittweite","Minibatchsize","Hiddenlayersize","erkannte Zahlen","Rechenzeit pro Epoche","Rechenzeit mehrere Epochen"])
print(np.around(pMat,decimals=2))
np.savetxt(pMat_name, pMat)