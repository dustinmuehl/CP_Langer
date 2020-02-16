# ReadMe: Neurale Netze #

**Entwickler:** Diesen Code haben *Moritz Amann*, *Isabell Giers*, *Benedikt Leibinger*, *Dustin Mühlhäuser* und *Ilya Shapiro* erstellt.

---------------

## Allgemeines ##
Ziel des Programmes ist es, mithilfe eines dreischichtigen neuronalen Netzes die vorgegebenen MNIST-Daten als Zahlen zu interpretieren. Dabei soll das Netz trainiert werden um die Aufgabe genauer und schneller zu lösen.

Dazu wurde die Klasse *UniversalNeuralNetwork* erstellt. Diese Klasse benötigt zwei Parameter, einmal die Anzahl der Knoten in jedem Layer in Form einer Liste *L* und einmal die Schrittweite *c* (später dazu mehr). Die Gewichte des Netzes werden dabei in der ersten Epoche gleichverteilt zufällig ausgewählt. Die Klasse enthält fünf Funktionen:

#### *feedforward* ####
Diese Funktion benötigt als Input *x* einen Vektor mit der selben Größe des Inputlayers des Netzes. Dieser Inputsvektor wird dann mit den entsprechenden Gewichten durch das Netz angepasst.

#### *backprob* ####
Diese Funktion benötigt als Input *y* einen Vektor mit der selben Größe des Outputlayers des Netzes. Diese Funktion berechnet dann die Gradienten, die notwendig sind für das Training.

#### *train* ####
Diese Funktion benötigt als Inputs eine Matrix *x* mit der Größe des Inputlayers, eine Matrix *y* mit der Größe des Outputlayers, die Anzahl der Epochen *n*, die Anzahl der Durchgänge pro Epoche *m* und die Schrittweite *c*. Diese Funktion trainiert das Netz mit den Trainingsdaten *x* und *y*.


## Wichtig ##
Der Code benötigt die Datei *mnist.pkl.gz*. Diese Datei darf **nicht** gelöscht werden. 
(Sollte die Datei verloren gehen, kann sie [hier](https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_fold_1764278.html) heruntergeladen werden)


