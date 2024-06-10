import sys

file = sys.argv[1]
out = file.replace('.csv', '_t.csv')

with open(file) as f:
  lis = [x.split(',') for x in f]

for x in zip(*lis):
  for y in x:
    print(y+'\t', end='')
  print('\n')

#write to file
with open(out, 'w+') as f:
  for x in zip(*lis):
    for y in x:
      f.write(y+',\t')
    f.write('\n')