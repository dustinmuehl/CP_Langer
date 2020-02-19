import matplotlib
import matplotlib.pyplot as plt
import numpy as np

pmat = np.loadtxt("pMatrix_S_[1, 2.5, 3, 3.5, 5]_B_[10]_H_[30].txt")
print(pmat)

labels = ['1', '2.5', '3', '3.5', '5']

prozent_erkannt_0 = pmat[[0,5,10,15,20],4]
prozent_erkannt_1 = pmat[[1,6,11,16,21],4]
prozent_erkannt_2 = pmat[[2,7,12,17,22],4]
prozent_erkannt_3 = pmat[[3,8,13,18,23],4]
prozent_erkannt_4 = pmat[[4,9,14,19,24],4]

x = np.arange(len(labels))  # the label locations
width = 0.12  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - 2*width, prozent_erkannt_0, width, label='Epoche 0', color=(0,0,1,0.25)) #rgb
rects2 = ax.bar(x - 1*width, prozent_erkannt_1, width, label='Epoche 1', color=(0,0,1,0.45))
rects3 = ax.bar(x + 0*width, prozent_erkannt_2, width, label='Epoche 2', color=(0,0,1,0.65))
rects4 = ax.bar(x + 1*width, prozent_erkannt_3, width, label='Epoche 3', color=(0,0,1,0.80))
rects5 = ax.bar(x + 2*width, prozent_erkannt_4, width, label='Epoche 4', color=(0,0,1,0.90))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel("Schrittweite")
ax.set_ylabel('% erkannt')
ax.set_title('Variation der Schrittweite, B=10, H=30')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(90,96)
ax2 = ax.twinx()
color = 'red'
ax2.set_ylabel("Benötigte Zeit für 4 Epochen", color=color)
ax2.plot(x, pmat[[4,9,14,19,24],6], color = color)
ax2.set_ylim(90,130)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()

plt.show()