import math 
import numpy as np
import gzip # Entzipper für das Auslesen der Testdaten
import pickle #Einlesen der Testdaten in einen Array

# Wird nur benötigt, wenn das Bild nochmals wiedergegeben werden soll 
import matplotlib.cm as cm
import matplotlib.pyplot as plt

###############################-To Do- ########################################
#- Backwardspropagation Algorithmus stochastisches Gradientenverfahren
#- 
###############################################################################


###############################-Fragen - ########################################
#- Sigmuid Funktion auch angewertet auf die Ausgabeschicht
#- Genaue Implementierung der SGD
#- Unterschied der Backwardspropagation
#- Welche Verlustfunktion genau --> (x-y)**2 ?
###############################################################################

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
print(train_x[0])
#Sigmuidfunktion für Vektoren 
def sig(z): 
    i = z.shape[0]
    y = np.zeros(z.shape[0])
    for j in range(0, i): 
        y[j] = (1+ math.exp(-z[j]))**(-1)
    return y 

def diff_sig(z):
    y = (-1)*(1+ math.exp(-z[j]))**(-2)
    return y


#def lossfunction(h, y, k): #h Vektor der ermittelten Werte, y der wahre Wert, Neuron k 
#    r = (h[k]-y)
#    return r

#Klasse für Neuronales Netzwerk
class NeuronalNetwork:
    def __init__(self, x, ql2, ql3):
        # x als Vektor, ql1, ql2, ql3 sind jeweils die Layergrößen 
        #Bias definieren
        self.y              = np.array([1]) # Bias 1 
        self.z              = np.array([1]) # Bias 2
        #Layergrößen definieren
        self.Shape_in_l     = x.shape[0] #Größe des input Layers + 1 Bias 
        self.Shape_h_l      = ql2 #Größe des hidden Layers 
        self.Shape_out_l    = ql3 #Größe des output Layers 
        self.input          = np.concatenate((x, self.y)) #Input Array aus den Trainingsdaten, erweitert um einen Bias
        self.w_l_1          = np.random.rand(self.Shape_in_l+1, self.Shape_h_l) #erste Gewichtsmatrix
        self.w_l_2          = np.random.rand(self.Shape_h_l+1, self.Shape_out_l) #zweite Gewichtsmatrix (30+1Bias)x10
        self.output         = np.zeros(self.Shape_out_l)
    
    def forward(self): 
        self.layer1 = np.concatenate((sig(np.dot(self.input, self.w_l_1)), self.z)) #Matrixmultiplikation, Sigmuid funktion, Erweiterung um Bias
        self.output = sig(np.dot(self.layer1, self.w_l_2))
        return self.output
    
def sto_grad():
    #Wähle zufällig (x, y) also zuf. Tupel von Eingangswerten und endwerten
    k = np.random.randint(0, 50.000)
    n = NeuronalNetwork(train_x[k], 30, 10)
    x = n.forward()
    y = train_y[k]
        #unterscheidung zwischen den Gewichten des letzten Layers 
        #und dem Layer zuvor
        
        
            #Berechne Gradienten g(j) --> Backpropagation
            #
            #Setze Gewichte(j+1) = Gewichte(j) - s(j)*g(j)
  

















      
#Forward Ergebnisse gesammelt in der Matrix A(50.000x10)
#A = np.zeros((train_x.shape[0], 10))
#for i in range(0, train_x.shape[0]):
#    
#    A[i, :] = n.forward()

#print(A[49999, :])             #Überprüfung 
    









