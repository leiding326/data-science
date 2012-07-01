'''
Created on Jun 30, 2012

@author: leiding
'''
import sys
import numpy
import random

file_name, eps_string, conf_string = sys.argv[1:4]
eps = float(eps_string)
confidence = float(conf_string)
func = lambda x: x[0] + x[1] ## this is an example function to simulate data
noise_level = 1 ## play with the noise level to see how this impacts the results

## Simulate data
random.seed()
the_file_output = open (file_name, "w")
for i in range(10000):
    x1 = random.random()
    x2 = random.random()
    y = func([x1, x2]) + random.random() * noise_level
    the_file_output.write(str(x1) + " " + str(x2) + " " + str(y) + "\n")
the_file_output.close()

## Calculate bounds
diff = []
diff2 = []
with open(file_name) as the_file:
    for i, line in enumerate(the_file):
        list_data = line.strip().split()
        data = [float(num) for num in list_data[0:-1]]
        label = float(list_data[-1])
        diff += [abs(func(data) - label)]
        diff2 += [pow((func(data) - label), 2)]
the_file.close()
sigma2 = pow(numpy.std(diff2), 2)
M = max(diff)
numData = -numpy.log((1-confidence) / 2) * 2 * ( sigma2 + pow(M, 2) * eps / 3 ) / pow(eps, 2)
print "You may need this many data points to have confidence in your regression: " + str(int(round(numData)))