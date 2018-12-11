import sys
dataset = open(sys.argv[1], 'r')
import pyfpgrowth

objectItemsets = {}
subjectItemsets = {}

for line in dataset:
     if len(line.split()) == 3:
        s,o,p = line.split()
        if o not in objectItemsets:
            objectItemsets[o] = set()
        objectItemsets[o].add(str(p))
        #subjectitemsets
        if s not in subjectItemsets:
            subjectItemsets[s] = set()
        subjectItemsets[s].add(str(p))
transactions = []
for key, value in objectItemsets.items():
    transactions.append(list(value))


#transactions = transactions[0:1000]

from pyspark import SparkContext
sc = SparkContext("local", "Simple App")

from pyspark.mllib.fpm import FPGrowth
data = transactions
rdd = sc.parallelize(data, 10)
model = FPGrowth.train(rdd, sys.argv[3])
patterns = model.freqItemsets().collect()
#print(patterns)
#for fi in patterns:
#    for i in fi.items:
#        print(i)
print('finished frequent item set mining')
print('found ' + str(len(patterns)) + 'items')

import math
synonyms = open(sys.argv[2], 'w')

def support(p1, p2=None):
    counter = 0
    if p2 is None:
        for key, value in subjectItemsets.items():
            if p1 in value:
                counter += 1
        return counter
    else:
        for key, value in subjectItemsets.items():
            if p1 in value and p2 in value:
                counter += 1
        return counter

def eCoeff(p1,p2):
    n = len(subjectItemsets)
    sup1 = support(p1)
    sup2 = support(p2)
    sup12 = support(p1,p2)
    return (n* sup12 - (sup1*sup2)) / math.sqrt(sup2 * (n-sup2) * sup1 * (n-sup1))
    

results = {}
alreadyVisited = set()
for pattern in patterns:
    if len(pattern.items) > 1:
        for p1 in pattern.items:
            for p2 in pattern.items:
                if p1 != p2 and (p1+p2) not in alreadyVisited:
                    alreadyVisited.add(p1+p2)
                    alreadyVisited.add(p2+p1)
                    if (str(p2) + "\t" + str(p1)) not in results:
                        results[str(p1) + "\t" +  str(p2)] = eCoeff(p1,p2)
for w in sorted(results, key=results.get, reverse=False):
    print(w, results[w])
    synonyms.write(w + "\n")
synonyms.close()
