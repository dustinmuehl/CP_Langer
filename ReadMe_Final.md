**Entwickler:** Diesen Code haben *Moritz Amann*, *Isabell Giers*, *Benedikt Leibinger*, *Dustin Mühlhäuser* und *Ilya Shapiro* erstellt.

---------------

## Allgemeines ##
Ziel des Programmes ist es, mithilfe eines neuronalen Netzes die vorgegebenen MNIST-Daten als Zahlen zu interpretieren. Dabei soll das Netz trainiert werden um die Aufgabe genauer und schneller zu lösen.

Es werden hauptsächlich drei Module verwendet, *NeuronalesNetz*, welches die Klasse *NeuralNetwork* enthält (s.u.), *Hyperparameterstudie* (s.u.) und *Rechenskript*. Letztere Datei enthält sämtlichen auszuführenden Code.

Diese Klasse *NeuralNetwork* benötigt zwei Parameter, einmal die Anzahl der Knoten in jedem Layer (ohne Bias) in Form einer Liste *L* und einmal ein Anpassungsfaktor *c* der Startgewichte. Die Gewichte des Netzes werden dabei in der ersten Epoche gleichverteilt zufällig ausgewählt aus dem Intervall *c* * [0,1). Die Klasse enthält einige Funktionen:



#### *feedforward* ####
Diese Funktion benötigt als Input *x* einen Vektor mit der selben Größe des Inputlayers (ohne Bias) des Netzes. Dieser Inputvektor wird dann mit den entsprechenden Gewichten durch das Netz interpretiert, das heißt aus der Inputschicht werden iterativ die nächsten Schichten berechnet.

#### *backprob* ####
Diese Funktion benötigt als Input *y* einen Vektor mit der selben Größe des Outputlayers des Netzes. Diese Funktion berechnet dann die Gradienten, die notwendig sind für das Training und gibt diese zurück.

#### *train* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias), eine Matrix *y* mit Spalten der Länge des Outputlayers und der gleichen Spaltenanzahl wie die der Matrix *x*, die Anzahl der Durchgänge pro Epoche *n*, die Größe der Minibatches *m* und die Schrittweite *c*. Diese Funktion trainiert, also passt die Gewichte des Netzes mit den Trainingsdaten *x* und *y* an.

#### *test1* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias) und eine Matrix *y* mit Spalten der Länge des Outputlayers und der gleichen Spaltenanzahl wie die der Matrix *x*. Diese Funktion summiert die Beträge der Differenzen der Einträge des Outputs des neuronalen Netzes zu einer Spalte der Matrix *x* und der entsprechenden Spalte der Matrix *y*, mittelt diese, summiert diese Mittelwerte auf und mittelt diese erneut. Dies ergibt die prozentuale Übereinstimmung der Outputs der Spalten von *x* und der dazugehörigen korrekten Werte aus der Matrix *y*.


#### *test2* ####
**Hinweis:** *test2* bezieht sich auf ein neuronales Netz mit einem Outputlayer mit zehn Neuronen.
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias) und einen Zeilenvektor, wobei die Länge des Vektors der Anzahl der Spalten von *x* entspricht. Diese Funktion vergleicht die Stelle des Maximums im Output zu den Spalten von *x* mit dem entsprechenden Eintrag in *y*. Bei Übereinstimmung wird ein Erfolg verzeichnet. Die Funktion gibt daraufhin die Anzahl der Erfolge zurück.


#### *test3* ####
**Hinweis:** *test3* bezieht sich auf ein neuronales Netz mit einem Outputlayer mit vier Neuronen.
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias) und eine Matrix *y* mit Spalten der Länge des Outputlayers und der gleichen Spaltenanzahl wie die der Matrix *x*. Diese Funktion rundet den finalen Output und vergleicht diesen mit der entsprechenden Spalte von *y*. Bei Übereinstimmung wird ein Erfolg verzeichnet. Die Funktion gibt daraufhin die Anzahl der Erfolge zurück.


#### *erkenne* ####
Diese Funktion benötigt als Inputs einen Vektor der Größe, die der Anzahl der Neuronen des Inputlayers entspricht, sowie das tatsächliche Ergebnis, das ein perfektes Netz erkennen sollte. Die Funktion gibt die Zahl, die das Netz erkannt hat (nach dem Vorgehen von *test2*, bzw. *test3*), sowie die korrekte Zahl aus.

#### *Hyperparameterstudie (HPS)* #####
Die HPS besteht im wesentlichen aus der Funktion *hypP*. Diese Funktion benötigt als Inputs eine Liste *S*, die  verschiedene Schrittweiten enthält, eine Liste *B*, die verschiedene Minibatch-Größen enthält, eine lIste *H*, die verschiedene Neuronenanzahlen des Hiddenlayers enthält und die Anzahl der zu durchlaufenden Epochen *e*. Für jede Kombination der Parameter aus den Listen wird ein Trainiervorgang durchgeführt und die Ergebnisse (nach *test2*) und die benötigte Zeit (für eine Epoche und für *e*-viele Epochen) in der Matrix *pMat* festgehalten. Dabei gehören je *e*-Zeilen zu einer Parameterkombination und in den Spalten stehen (von links nach rechts): die aktuelle Epoche, die verwendete Schrittweite, die verwendete Minibatch-Größe, die verwendete Anzahl an Neuronen im Hiddenlayer, die Anzahl richtig erkannter Zahlen, die benötigte Zeit für die aktuelle Epoche und die benötigte Zeit für *e* Epochen. Das Modul Plot_Studie.py ist nur um Plots zu erstellen und bedarf keiner größeren Aufmerksamkeit.


## Wichtig ##
Der Code benötigt die Datei *mnist.pkl.gz*. Diese Datei darf **nicht** gelöscht werden. 
(Sollte die Datei verloren gehen, kann sie [hier](https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_fold_1764278.html) heruntergeladen werden)