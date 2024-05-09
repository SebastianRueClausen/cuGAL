import numpy as np
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
data = np.load(path)

if len(sys.argv) > 2:
    option = data[sys.argv[2]]

if option == 'gradient':
    data = data[:, :, 0, :, 0]

#print(data[0, :, 0 :, 0])
#print(data[0, :, 0, 0, 0, 0])

fig, ax = plt.subplots()

labels = ['fw000', 'fw005', 'fw010', 'fw020', 'fw050', 'fw100']
noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

[ax.plot(noises, data[0, :, 0, i, 0], label=labels[i]) for i in range(len(labels))]
#[ax.plot(range(1, 6), data[0, 0, :, i, 0, 0], label=labels[i]) for i in range(3)]
#ax.plot(range(0, 16, 5), data[0, :, 0, 0, 0, 0])#, label=labels[i]) for i in range(3)]

#ax.set_yscale('log')
ax.set_xlabel('Noise')


#add gridlines
ax.grid(True, which='both')


ax.legend()

#set y label to accuracy in percent and tick labels to percentage and limit y axis to 0-1
ax.set_ylabel('Accuracy (%)')
vals = ax.get_yticks()
#ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
#ax.set_ylim(0, 1)
plt.show()