from decimal import Decimal

n = 256
m = 256

a = -0.5
b = -1.0
c = 0.5

h = (1 / 128)
fout = open('internalCloud.txt', 'wt')

for i in range(n):
   for j in range(m):
      print("(%lf %lf %lf)" %(a, b, c), file=fout)
      a += h
   b += h
   a = -0.5

fout.close()










