import numpy as np
import gzip  # Entzipper für das Auslesen der Testdaten
import pickle  # Einlesen der Testdaten in einen Array
import os.path
import time

# Wird nur benötigt, wenn das Bild nochmals wiedergegeben werden soll
import matplotlib.cm as cm
import matplotlib.pyplot as plt

with gzip.open('mnist.pkl.gz', 'rb') as f:  # Öffnet das Trainingsdatenset
    train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
    # train_set[0] ist eine 50.0000 x 784 Matrix --> Pixel Daten
    # train_set[1] ist eine 50.0000 x 1 Matrix --> Wahre Werte
    # encoding='iso-8859-1' wird benötigt, da ab python 3 ein anderer
    # Kodierungsstandard Verwendung findet und das Trainingsdatenset in
    # Python 2 geschrieben wurde

# überführe train_x in np Array
train_x = np.zeros((len(train_set[0]), 784))
for i in range(len(train_set[0])):  # Pixeldatenset
    train_x[i, :] = np.asarray(train_set[0][i])

train_y = train_set[1]  # Ziffern in Computerdarstellung

# überführe test_x in np Array
test_x = np.zeros((len(test_set[0]), 784))
for i in range(len(test_set[0])):  # Pixeldatenset
    test_x[i, :] = np.asarray(test_set[0][i])

test_y = test_set[1]

# überführe Ziffern in Vektoren der passenden Form
train_y_dec = np.zeros([len(train_y), 10])
for i in range(len(train_y)):
    train_y_dec[i][train_y[i]] = 1

test_y_dec = np.zeros([len(test_y), 10])
for i in range(len(test_y)):
    test_y_dec[i][test_y[i]] = 1


# Sigmoid-Funktion
def sig(x):
    return 1 / (1 + np.exp(-x))


# Ableitung der Sigmoid-Funktion
def sigder(x):
    return sig(x) * (1 - sig(x))


class NeuralNetwork:
    def __init__(self, V0, V1, V2):
        self.V0size = V0 + 1  # |V_0|
        self.V1size = V1 + 1  # |V_1|
        self.V2size = V2  # |V_2|
        # erschaffe leere Layer
        self.V0 = np.zeros(self.V0size)
        self.A1 = np.zeros(self.V1size - 1)
        self.V1 = np.zeros(self.V1size)  # Auch V_0_O, also Output
        self.A2 = np.zeros(self.V2size)
        self.V2 = np.zeros(self.V2size)  # Auch V_1_O, also Output
        # bias als numpy Array, zum rankleben an x nachher
        self.bias = np.array([1])
        # Gewichtsmatrizen konstruieren
        self.W1 = 0.01 * np.random.rand(self.V1size - 1, self.V0size)
        self.W2 = 0.01 * np.random.rand(self.V2size, self.V1size)

    # Feed-Forward Funktion, eingabe: ein einziger Bildvektor x
    def feedforward(self, x):
        self.V0 = np.concatenate((x, self.bias))

        self.A1 = np.dot(self.W1, self.V0)
        self.V1 = np.concatenate((sig(self.A1), self.bias))

        self.A2 = np.dot(self.W2, self.V1)
        self.V2 = sig(self.A2)

    # Back-Propagation Funktion, berechne Gradienten, eingabe: Zahl y
    def backprop(self, y):
        W2 = np.delete(self.W2, self.W2.shape[1] - 1, 1)
        # definiere Variablen wie im Skript
        h = self.V2
        sigdera1 = np.diag(sigder(self.A1))
        sigdera2 = np.diag(sigder(self.A2))
        o1 = self.V1
        o0 = self.V0
        # Berechne delta_T (T=2)
        delta2 = np.dot(sigdera2, (h - y))
        # Berechne delta1
        delta1 = np.dot(sigdera1, np.dot(delta2.T, W2).T)
        gradw2 = np.outer(delta2, o1)
        gradw1 = np.outer(delta1, o0)
        # ...und gebe diese zurück
        output = [gradw1, gradw2]
        return output

    # Trainigsfunktion, eine Implementierung des SGV
    # Eingabe: Datensatz x und y (Matrizen), mb-größe,
    # Anzahl Durchgänge pro Epoche, Konstante für Schrittweite c
    def train(self, x, y, m, n, c):
        # Indexmenge, normal 0,...,49999
        N = np.arange(y.shape[1])
        # Permutation von Indexmenge für Epoche
        np.random.shuffle(N)
        # führe n Anpassungen durch
        for k in range(n):
            gradw1 = np.zeros((self.V1size - 1, self.V0size))
            gradw2 = np.zeros((self.V2size, self.V1size))
            i = k * m
            for l in range(m):
                self.feedforward(x[:, N[i]])
                prop = self.backprop(y[:, N[i]])
                gradw1 = gradw1 + prop[0]
                gradw2 = gradw2 + prop[1]
                i = i + 1
            # führe Anpassung des Gewichts durch
            self.W1 = self.W1 - c * (1 / m) * gradw1
            self.W2 = self.W2 - c * (1 / m) * gradw2
        # speichere Gewichte
        # np.savetxt("Weights1.txt", self.W1)
        # np.savetxt("Weights2.txt", self.W2)

    def traindumm(self, x, y, c):
        for k in range(y.shape[1]):
            self.feedforward(x[:, k])
            prop = self.backprop(y[:, k])
            self.W1 = self.W1 - c * prop[0]
            self.W2 = self.W2 - c * prop[1]

    # Testfunktion, die die Treffergenauigkeit misst
    # Eingabe: Datensatz x und y
    def test1(self, x, y):
        abstand = 0
        for j in range(y.shape[1]):
            self.feedforward(x[:, j])
            summe = 0
            # bestimme summe der Abweichungen eines Datenpaares
            for i in range(self.V2size):
                summe = summe + abs(y[i][j] - self.V2[i])
            # Bilde Durchscnitt der Summe
            summe = (1 / self.V2size) * summe
            # Addiere Durchscnittliche Abstände auf
            abstand = abstand + summe
        # Erfolgs-wk ist 1-(normierte Abstandssumme)
        P_erfolg = 100 * (1 - (1 / y.shape[1]) * abstand)
        #print(P_erfolg, "% Erfolg")

    # Testfunktion, die Index der maximalen Ausgabe mit richtiger Ziffer
    # vergleicht und Anzahl übereinstimmungen printet
    def test2(self, x, y):
        summe = 0
        for j in range(x.shape[1]):
            self.feedforward(x[:, j])
            if np.argmax(self.V2) == y[j]:
                summe = summe + 1
        print(summe/x.shape[1]*100, "% erkannt")


# Los geht's mit dem Rechenspaß

e = 4       # Anzahl Epochen, zb e=4 damit Epoche 0,1,2,3 berechnet wird
T = np.zeros((1,7), float)
S = []
H = [10, 20, 30, 40, 50]
for h in range(len(H)):          #Für Hyperparameter HiddenLayer Größe
    n = NeuralNetwork(784, H[h], 10)
    name1 = "Weights1" + str(h) + ".txt"
    name2 = "Weights2" + str(h) + ".txt"
    print(H[h])

    #n = NeuralNetwork(784,30,10)

    C = [1, 1.5, 2, 2.5, 3]  # 0.05 nach epoche 2: 69%
    K = [8, 10, 12, 14, 16, 18, 20]  # bei 15 am besten, 5, 10, 20 aber vergleichbar gut

    for k in range(1):          #für Hyperparamter Schrittweite und MB-Size
        #p = int(50000 / K[k])  # nur damit Anzahl der Durchgänge pro Epoche als Integer
        if os.path.exists(name1) == True:
            n.W1 = np.loadtxt(name1)
            n.W2 = np.loadtxt(name2)
        # print( K[k])

        for j in range(1):      #um computerbedingte Laufzeitdifferenzen zu glätten, zb range(3)
            start_proc = time.process_time()

            n.feedforward(test_x.T[:, 4000])

            # print("output = ", n.V2)
            # print("richtig = ", test_y[4000])

            # print(timeit.timeit(number=10))
            for i in range(2):      #Epochen Durhlauf, Standard=20, für Studie=5
                # print(n.W1, n.W2)
                # c=1/(0.01*(25+i))
                S.append(e)
                start_proc_epoch = time.process_time()
                n.train(train_x.T, train_y_dec.T, 10, 5000, 2)
                # n.train(train_x.T, train_y_dec.T, 10, 5000, C[k])  # Schrittweite C[j]
                # n.train(train_x.T, train_y_dec.T, K[k], p, 2)  # Mini-Batch-Size K[j]
                S.append(20, 10, h) #Eintrag 2,3,4 in jeweiliger Zeile : Schrittweite, MB-Size, Größe Hidden Layer
                # n.traindumm(train_x.T ,train_y_dec.T , 0.3)
                print("Epoche", i)
                n.test1(test_x.T, test_y_dec.T)
                n.test2(test_x.T, test_y.T)
                end_proc_epoch = time.process_time()
                # print("Dauer der Berechunung von Epoche ", str(i), ":{:5.3f}".format(end_proc_epoch - start_proc_epoch))
                S.append(end_proc_epoch - start_proc_epoch)         #Durchlaufzeit pro Epoche
                n.feedforward(test_x.T[:, 4000])


            end_proc = time.process_time()
            print("Dauer der Berechunung von 3 Epochen:{:5.3f}sek".format(end_proc - start_proc))
            S.append(end_proc - start_proc)         #Durchlaufzeit für alle e Epochen
            print(S)
            np.append(T, S, axis=0)
            S = []
        # np.savetxt("Liste_S_Schritteweite.txt", S)
        # np.savetxt("Liste_S_Mini-Batch-Size.txt", S)
        # np.savetxt("Liste_S_Hidden_Layer_Größe.txt", S)