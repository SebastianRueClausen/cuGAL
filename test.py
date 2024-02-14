import numpy as np
import official.pred as p
import networkx as nx
import official.metrics as metrics

f1 = open("./official/data/yeast0_Y2H1.txt", "r")
f2 = open("./official/data/yeast10_Y2H1.txt", "r")
nodes1 = [list(map(int, n.split())) for n in f1.read().split("\n") if n]
nodes2 = [list(map(int, n.split())) for n in f2.read().split("\n") if n]
n = max([max(n) for n in nodes1])
n2 = max([max(n) for n in nodes2])

assert(n == n2)

A1 = np.zeros((n, n))
for i, j in nodes1:
    A1[i - 1, j - 1] = 1
    A1[j - 1, i - 1] = 1

A2 = np.zeros((n, n))
for i, j in nodes2:
    A2[i - 1, j - 1] = 1
    A2[j - 1, i - 1] = 1

print("A1:\n", type(A1))
print("A2:\n", type(A2[0][0]))

mapping, _ = p.predict_alignment([nx.from_numpy_array(A1)], [nx.from_numpy_array(A2)])

mapping = [x for _, x in mapping[0]]

print("ICS:", metrics.ICS(A1, A2, np.arange(n), mapping))

f1.close()
f2.close()