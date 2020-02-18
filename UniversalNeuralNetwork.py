import numpy as np
import gzip # Entzipper für das Auslesen der Testdaten
import pickle #Einlesen der Testdaten in einen Array
import os.path

with gzip.open('mnist.pkl.gz', 'rb') as f: #Öffnet das Trainingsdatenset
        train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
        #train_set[0] ist eine 50.0000 x 784 Matrix --> Pixel Daten
        #train_set[1] ist eine 50.0000 x 1 Matrix --> Wahre Werte
        #encoding='iso-8859-1' wird benötigt, da ab python 3 ein anderer 
        #Kodierungsstandard Verwendung findet und das Trainingsdatenset in 
        #Python 2 geschrieben wurde 

train_x = train_set[0]
train_y = train_set[1] #Ziffern in Computerdarstellung

test_x = test_set[0]
test_y = test_set[1]


#überführe Ziffern in Vektoren der passenden Form
train_y_dec = np.zeros([len(train_y),10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1
    
test_y_dec = np.zeros([len(test_y),10])
for i in range(len(test_y)):
    test_y_dec[i][test_y[i]] = 1

#Sigmoid-Funktion
def sig(x):
    return 1/(1 + np.exp(-x))
    

#Ableitung der Sigmoid-Funktion
def sigder(x):
    return sig(x)*(1-sig(x))


class UniversalNeuralNetwork:
    def __init__(self,L,c):
        #L ist eine Liste, die die Größen der einzelnen Layers enthält
        self.layercount = len(L)        #Anzahl der Layers
        
        
        K = []          #Größe der Layers mit Bias
        for i in range(self.layercount -1):
            K.append(L[i]+1)
        K.append(L[self.layercount-1])
        self.layersize = K
        
        
        J1 = []
        for i in range(self.layercount):
            J1.append(np.zeros(self.layersize[i]))
        self.V = J1
        
        
        self.bias = np.array([1])   #Bias
    
    
        J2 = []
        for i in range(self.layercount -2):
            J2.append(np.zeros(self.layersize[i+1] -1))
        J2.append(np.zeros(self.layersize[self.layercount -1]))
        self.A = J2
    

        J3 = []         #Liste der Gewichtsmatrizen
        for i in range(self.layercount -1):
            J3.append(c*np.random.rand(L[i+1],self.layersize[i]))
        self.weights = J3    
        
        
        
    def feedforward(self,x):
        K1 = []
        K2 = []
        
        K1.append(np.concatenate((x,self.bias)))
        for i in range(self.layercount-2):
            K1.append(np.concatenate((sig(np.dot(self.weights[i], K1[i])),self.bias)))
        K1.append(sig(np.dot(self.weights[self.layercount -2], K1[self.layercount -2])))
        self.V = K1
                  
        for i in range(self.layercount -1):
            K2.append(np.dot(self.weights[i],self.V[i]))
        self.A = K2
        
        
        
    def backprop(self, y):
        
        W = []
        for i in range(self.layercount -2):
            W.append(np.delete(self.weights[i+1], self.weights[i+1].shape[1]-1, 1))
        
        a=[]
        for item in self.A:
            a.append(np.diag(sigder(item)))    
        
        h = self.V[self.layercount -1]    
        
        o = []
        for i in range(self.layercount-1):
            o.append(self.V[i])
        
        delta = []
        delta.insert(0,np.dot(a[self.layercount -2],(h-y)))
        for i in range(self.layercount -2):
            delta.insert(0, np.dot(a[self.layercount -3-i],np.dot(delta[0].T, W[self.layercount-3-i]).T))
        
        gradw = []
        for i in range(self.layercount -1):
            gradw.append(np.outer(delta[i],o[i]))
        output = gradw
        return output
    

    
    def train(self, x, y, m, n, c):
        N = np.arange(y.shape[1])
        np.random.shuffle(N)
        for k in range(n):
            gradw = []
            for i in range(self.layercount-2):
                gradw.append(np.zeros((self.layersize[i+1]-1, self.layersize[i])))
            gradw.append(np.zeros((self.layersize[self.layercount-1],self.layersize[self.layercount-2])))
            
            i = k*m
            for l in range(m):
                self.feedforward(x[:,N[i]])
                backprop=self.backprop(y[:,N[i]])
                for j in range(self.layercount -1):
                    gradw[j] = gradw[j] + backprop[j]
                i = i +1
            
            for j in range(self.layercount-1):
                self.weights[j] = self.weights[j] -c*(1/m)*gradw[j]



    def test1(self, x, y):
        abstand=0
        for j in range(y.shape[1]):
            self.feedforward(x[:,j])
            summe=0
            for i in range (self.layersize[self.layercount-1]):
                summe=summe+abs(y[i][j]-self.V[self.layercount-1][i])
            summe=(1/self.layersize[self.layercount-1])*summe
            abstand=abstand+summe
        P_erfolg=100*(1-(1/y.shape[1])*abstand)
        print(P_erfolg,"% Erfolg")



    def test2(self, x, y):
        summe=0
        for j in range(x.shape[1]):
            self.feedforward(x[:,j])
            if np.argmax(self.V[self.layercount-1])==y[j]:
                summe=summe+1
        print(summe,"/",x.shape[1],"erkannt")

###################################################################################################################
g=[784,30,10]

n=UniversalNeuralNetwork(g,0.01)

n.feedforward(test_x.T[:,4001])

print("output = ", n.V[n.layercount-1])
print("richtig = ", test_y[4001])

for i in range(5):
    #c=1/(0.01*(25+i))
    n.train(train_x.T ,train_y_dec.T , 10, 5000, 0.3)
    print("Epoche",i)
    #n.test1(test_x.T, test_y_dec.T)
    n.test2(test_x.T, test_y.T)

n.feedforward(test_x.T[:,4001])

print("output = ", n.V[n.layercount-1])
