import numpy as np

class UniversalNeuralNetwork:
    def __init__(self,L):
        self.layercount = len(L)
        
        K=[]
        for i in range(self.layercount -1):
            K.append(L[i]+1)
        K.append(L[self.layercount-1])
        self.layersize = K
        
        
        
        J1 = []
        for i in range(self.layercount):
            J1.append(np.zeros(self.layersize[i]))
        self.V = J1
        
        self.bias = np.array([1])
        
        J2 = []
        for i in range(self.layercount -1):
            J2.append(np.random.rand(L[i+1],self.layersize[i]))
        self.weights = J2
        
        J3 = []
        for i in range(self.layercount -2):
            J3.append(np.zeros(self.layersize[i+1] -1))
        J3.append(np.zeros(self.layersize[self.layercount -1]))
        self.A = J3
        
        
        
        
    def feedforward(self,x):
        K1 = []
        K2 = []
        
        K1.append(np.concatenate((x,self.bias)))
        for i in range(self.layercount-2):
            K1.append(np.concatenate((sig(np.dot(self.weights[i], self.V[i])),self.bias)))
        K1.append(sig(np.dot(self.weights[self.layercount -2], self.V[self.layercount -2])))
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
            a.append(np.diag(item))    
        
        h = self.V[self.layercount -1]    
        
        o = []
        for i in range(self.layercount-1):
            o.append(self.V[i])
        
        delta = []
        delta.insert(0,np.dot(sigder(a[self.layercount -2]),(h-y)))
        for i in range(self.layercount -2):
            delta.insert(0, np.dot(sigder(a[self.layercount -3-i]),np.dot(delta[0].T, W[self.layercount-3-i]).T))
        
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
                for j in range(self.layercount -1):
                    gradw[j] = gradw[j] + self.backprop(y[:,N[i]])[j]
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
