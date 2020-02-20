**Entwickler:** Diesen Code haben *Moritz Amann*, *Isabell Giers*, *Benedikt Leibinger*, *Dustin Mühlhäuser* und *Ilya Shapiro* erstellt.

---------------

## Allgemeines ##
Ziel des Programmes ist es, mithilfe eines neuronalen Netzes die vorgegebenen MNIST-Daten als Zahlen zu interpretieren. Dabei soll das Netz trainiert werden um die Aufgabe genauer und schneller zu lösen.

Dazu wurde die Klasse *UniversalNeuralNetwork* erstellt. Diese Klasse benötigt zwei Parameter, einmal die Anzahl der Knoten in jedem Layer (ohne Bias) in Form einer Liste *L* und einmal ein Anpassungsfaktor *c* der Startgewichte. Die Gewichte des Netzes werden dabei in der ersten Epoche gleichverteilt zufällig ausgewählt aus dem Intervall *c* * [0,1). Die Klasse enthält fünf Funktionen:

#### *feedforward* ####
Diese Funktion benötigt als Input *x* einen Vektor mit der selben Größe des Inputlayers (ohne Bias) des Netzes. Dieser Inputvektor wird dann mit den entsprechenden Gewichten durch das Netz interpretiert, das heißt aus der Inputschicht werden iterativ die nächsten Schichten berechnet.

#### *backprob* ####
Diese Funktion benötigt als Input *y* einen Vektor mit der selben Größe des Outputlayers des Netzes. Diese Funktion berechnet dann die Gradienten, die notwendig sind für das Training und gibt diese zurück.

#### *train* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias), eine Matrix *y* mit Spalten der Länge des Outputlayers und der gleichen Spaltenanzahl wie die der Matrix *x*, die Anzahl der Durchgänge pro Epoche *n*, die Größe der Minibatches *m* und die Schrittweite *c*. Diese Funktion trainiert, also passt die Gewichte des Netzes mit den Trainingsdaten *x* und *y* an.

#### *test1* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias) und eine Matrix *y* mit Spalten der Länge des Outputlayers und der gleichen Spaltenanzahl wie die der Matrix *x*. Diese Funktion summiert die Beträge der Differenzen der Einträge des Outputs des neuronalen Netzes zu einer Spalte der Matrix *x* und der entsprechenden Spalte der Matrix *y*, mittelt diese, summiert diese Mittelwerte auf und mittelt diese erneut. Dies ergibt die prozentuale Übereinstimmung der Outputs der Spalten von *x* und der dazugehörigen korrekten Werte aus der Matrix *y*.


#### *test2* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit Spalten der Länge der Inputschicht (ohne Bias) und einen Zeilenvektor, wobei die Länge des Vektors der Anzahl der Spalten von *x* entspricht. Diese Funktion vergleicht die Stelle des Maximums im Output zu den Spalten von *x* mit dem entsprechenden Eintrag in *y*. Bei Übereinstimmung wird ein Erfolg verzeichnet. Die Funktion druckt daraufhin die Anzahl der Erfolge im Vergleich zur Anzahl der Testdaten.


#### *Hyperparameterstudie (HPS)* #####
Die HPS besteht im wesentlichen aus der Funktion hypP(S, B, H, e). Diese erstellt aus den in den Arrays Schrittweite S, MiniBatch-Size B, Hidden-Layer Größe H und Anzahl der zu durchlaufenden Epochen e eine Matrix pMat, welche alle wesentlichen Daten enthält. Diese enthält in den Zeilen je eine Epoche, e-viele aufeinander folgende Zeilen gehören immer zu einem Durchlauf eines Parametertupels (s,b,h). In den Spalten stehen von links nach rechts: die jeweilige Epoche im Durchlauf, die Schrittweite, die MB-Size, die Hidden-Layer-Größe, die Anzahl der erkannten Ziffern nach der jeweiligen Epoche in Prozent, die benötiogte Zeit für die jeweilige Epoche sowie die Zeit für alle Epochen zusammen je Durchlauf. 
Das Modul Plot_Studie.py ist nur um die in der Präsenation gezeigten Plots zu erstellen und bedarf keiner größeren Aufmerksamkeit.


## Wichtig ##
Der Code benötigt die Datei *mnist.pkl.gz*. Diese Datei darf **nicht** gelöscht werden. 
(Sollte die Datei verloren gehen, kann sie [hier](https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_fold_1764278.html) heruntergeladen werden)