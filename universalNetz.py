import numpy as np


#Sigmoid-Funktion
def sig(x):
    output=1/(1 + np.exp(-x))
    return output

#Ableitung der Sigmoid-Funktion
def sigder(x):
    output=sig(x)*(1-sig(x))
    return output



class UniversalNeuralNetwork:
    
    #L: Liste mit den Größen der einzelnen Layer
    #c: Konstante, um die Gewichte anzupassen
    def __init__(self,L,c):
        self.layercount = len(L)        #Anzahl der Layers
        self.bias = np.array([1])       #Bias
        
        self.layersize = []          #Liste mit den Größen der Layer mit Bias
        for i in range(self.layercount -1):
            self.layersize.append(L[i]+1)
        self.layersize.append(L[self.layercount-1])

        
        
        self.V = []                 #Liste mit Vektoren, die die Outputs der einzelnen Layer enthalten
        for i in range(self.layercount):
            self.V.append(np.zeros(self.layersize[i]))
        
    
        self.A = []      #Liste mit Vektoren, die die Summen der Produkte von Gewichten und Outputs enthalten 
        for i in range(self.layercount -1):
            self.A.append(np.zeros(L[i+1]))


        self.weights = []         #Liste der Gewichtsmatrizen
        for i in range(self.layercount -1):
            self.weights.append(np.random.rand(L[i+1],self.layersize[i])*c) 
        
    #führt Feedforward-Berechnung mit übergebenem Input durch    
    #x: Vektor, der als Input gesetzt wird    
    def feedforward(self,x):
        KV = []
        KA = []
        
        KV.append(np.concatenate((x,self.bias)))    #Input + Bias
        
        for i in range(self.layercount-2):          #berechnet mittlere Schichten 
            KA.append(np.dot(self.weights[i],KV[i]))
            KV.append(np.concatenate((sig(KA[i]),self.bias)))
            
        KA.append(np.dot(self.weights[self.layercount -2],KV[self.layercount -2])) #berechnet Output
        KV.append(sig(KA[self.layercount -2]))
        
        self.V = KV
        self.A = KA
        
    #führt die Backpropagation mit derzeitigen Gewichten und Output durch    
    #y: Vektor, mit korrektem Wert, mit dem der Output verglichen werden soll 
    def backprop(self, y):
        
        W = []              #Liste der Gewichtsmatrizen ohne Biaswerte
        for i in range(self.layercount -2):
            W.append(np.delete(self.weights[i+1], self.weights[i+1].shape[1]-1, 1))
        
        sigdera=[]      #Hilfsmatrizen
        for item in self.A:
            sigdera.append(np.diag(sigder(item)))    
        
        h = self.V[self.layercount -1]   #Output des Outputlayers, zur besseren Übersicht umbenannt 
        
        o = []                          #Liste mit Outputs der übrigen Layer
        for i in range(self.layercount-1):
            o.append(self.V[i])
        
        delta = []                  #Berechnung der Deltas
        delta.insert(0,np.dot(sigdera[self.layercount -2],(h-y)))
        for i in range(self.layercount -2):
            delta.insert(0, np.dot(sigdera[self.layercount -3-i],np.dot(delta[0].T, W[self.layercount-3-i]).T))
        
        gradw = []                  #Berechnung der Gradienten
        for i in range(self.layercount -1):
            gradw.append(np.outer(delta[i],o[i]))
            
        output = gradw
        return output
    
    #trainiert das Netz anhand des übergebenen Datensatzes
    #x: Trainierdaten Bildwerte: Matrix der Dimension Inputlänge x Anzahl Daten
    #y: Trainierdaten Vergleichswerte: Matrix der Dimension Outputlänge x Anzahl Daten
    #mbsize: Größe der Minibatches
    #n: Anzahl, wie oft Verfahren angewendet werden muss, um alle Trainierdaten zu verwenden (abhängig von Anzahl Daten und Größe Minibatches)
    #schrittweite: Schrittweite des Gradientenverfahrens
    def train(self, x, y, mbsize, n, schrittweite):
        N = np.arange(y.shape[1])   #Indizes werden erstellt und zufällig sortiert
        np.random.shuffle(N)
        for k in range(n):
            gradw = []          #Liste von Nullmatrizen, die später durch Gradienten ersetzt werden
            for i in range(self.layercount-2):
                gradw.append(np.zeros((self.layersize[i+1]-1, self.layersize[i])))
            gradw.append(np.zeros((self.layersize[self.layercount-1],self.layersize[self.layercount-2])))
            
            i = k*mbsize
            
            for l in range(mbsize): #Stochastisches Gradientenverfahren: in jedem Minibatch werden die Gradienten berechnet und aufaddiert
                self.feedforward(x[:,N[i]])
                backprop = self.backprop(y[:,N[i]])
                for j in range(self.layercount -1):
                    gradw[j] = gradw[j] + backprop[j]
                i = i +1
            
            for j in range(self.layercount-1):  #Anpassung der Gewichte nach jedem Minibatch
                self.weights[j] = self.weights[j] -schrittweite*(1/mbsize)*gradw[j]
              
    #erhälte Werte eines Bildes, gibt Zahl, die das Netz erkennt, in der Konsole aus            
    #x: Vektor mit Werten des Bildes
    #y: Korrekte, zu erkennende Zahl als Vergleich
    def erkenne(self,x,y):
        self.feedforward(x)
        if(len(self.V[self.layercount-1])==10): #Falls das Netz Dezimal-Output hat
            print("Erkannt wurde (dec): ",np.argmax(self.V[self.layercount-1]))
            print("Richtig ist: ",y)
        elif(len(self.V[self.layercount-1])==4): #Falls das Netz Bit-Output hat
            vec = np.around(self.V[self.layercount-1])
            print("Erkannt wurde (bit): ",vec[0]*8+vec[1]*4+vec[2]*2+vec[3])
            print("Richtig ist: ",y)

    #bewertet durchschnittlichen Abstand der Werte des Outputs mit Vergleichswerten, verwendet ganzen Testdatensatz
    #x: Testdaten Bildwerte: Matrix der Dimension Inputlänge x Anzahl Daten
    #y: Testdaten Vergleichswerte: Matrix der Dimension Outputlänge x Anzahl Daten
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


    #zählt richtig erkannte Zahlen in übergebenem Testdatensatz (nur für Netz mit Dezimal-Output)
    #x: Testdaten Bildwerte: Matrix der Dimension Inputlänge x Anzahl Daten
    #y: Testdaten Vergleichswerte: Vektor der Länge Anzahl Daten
    def test2(self, x, y):
        summe=0
        for j in range(x.shape[1]):
            self.feedforward(x[:,j])
            if np.argmax(self.V[self.layercount-1])==y[j]:
                summe=summe+1
        print(summe,"/",x.shape[1],"erkannt (dec)")
     
        
    #zählt richtig erkannte Zahlen in übergebenem Testdatensatz (nur für Netz mit Bit-Output)
    #x: Testdaten Bildwerte: Matrix der Dimension Inputlänge x Anzahl Daten
    #y: Testdaten Vergleichswerte: Matrix der Dimension Outputlänge (=4) x Anzahl Daten  
    def test3(self, x, y): 
        summe = 0
        for j in range(x.shape[1]):
            self.feedforward(x[:,j])
            if np.array_equal(np.around(self.V[self.layercount-1]),y[j,:]):
                summe = summe+1
        print(summe,"/",x.shape[1],"erkannt (bit)")      
   