'''
Created on Jun 30, 2012

@author: leiding
'''
import sys
import numpy

file_name, eps_string = sys.argv[1:3]
eps = float(eps_string)
func = lambda x: x[0] + x[1]
diff = []
diff2 = []
with open(file_name) as the_file:
    for i, line in enumerate(the_file):
        list = line.strip().split()
        data = [float(num) for num in list[0:-1]]
        label = float(list[-1])
        diff += [abs(func(data) - label)]
        diff2 += [pow((func(data) - label), 2)]
 
the_file.close()
sigma2 = pow(numpy.std([1, 2]), 2)
M = max(diff)
numData = -numpy.log(0.05 / 2) * 2 * ( sigma2 + pow(M, 2) * eps / 3 ) / pow(eps, 2)
print "You may need this many data points to have confidence in your regressor: " + str(int(round(numData)))