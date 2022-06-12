import pandas as pd
import numpy as np
from itertools import combinations
from queue import Queue
'''
This is simple implementation of CN2 algorithm for discrete values of parameters
'''

def is_covered(row, complex):
	#returns if row of training data is covered by complex k
	for i, arg in enumerate(row):
		if arg not in complex[i]:
			return False
	return True	

def covered_df(data, complex):
	#returns dataframe covered by complex
	covered_data = [row for row in data.iloc if is_covered(list(row)[:-1],complex)]
	return pd.DataFrame(covered_data)

def rule_entropy(covered_data):	
	#returns measure of relevance of complex calculeted from covered df - the higher the better
	if len(covered_data) == 0: return -1
	class_series = covered_data[covered_data.columns[-1]]
	num_instances = len(class_series)
	class_counts = class_series.value_counts()
	class_probabilities = class_counts.divide(num_instances)
	log2_of_classprobs = np.log2(class_probabilities)
	plog2p = class_probabilities.multiply(log2_of_classprobs)
	entropy = plog2p.sum()
	return entropy

def atomic_combos(wildcards):
	atomset = []#list of all possible combinations (complexity 2^n-2)
	for i, wildcard in enumerate(wildcards):
		combos = [] 
		for l in range(1,len(wildcard)):
			x=combinations(wildcard, l)
			combos+=[set(y) for y in x]
		atomset.append(combos)
	return atomset
	
def calculate_prim(star,combos):
	#returns calculated intersection of atomic complexes and star, deletes empty complexes and those that are in star
	prim = []
	for row in star:
		tmp = list(row)
		for i, col in enumerate(combos):
			for atom in col:
				tmp[i]=row[i].intersection(atom)
				prim.append(tuple(tmp))

	temp = [x for x in prim if x not in star]
	prim = [y for y in temp if all(map(lambda x: not len(x)==0, y))]
	return prim

def compare_complex(trainer, prim, best, significance):
	#uses heuristics to calculate best complex from set
	newbest = best
	bestEntropy = rule_entropy(covered_df(trainer,newbest))
	for complex in prim:
		if len(covered_df(trainer,complex)) <= significance:
			continue
		complexEntropy = rule_entropy(covered_df(trainer,complex))
		if complexEntropy == 0.0:
			return complex
		if complexEntropy > bestEntropy: 
			bestEntropy = complexEntropy
			newbest = complex
	return newbest

def best_complex(trainer,prim):
	if len(prim)==0: return []
	best=prim[0]
	bestEntropy = rule_entropy(covered_df(trainer,best))
	for complex in prim:
		coveredData = covered_df(trainer,complex)
		complexEntropy = rule_entropy(coveredData)
		if complexEntropy == 0.0:
			return [complex]
		if complexEntropy > bestEntropy:
			best=complex
			bestEntropy = complexEntropy
	return [best]

def print_ruleset(ruleset,inputdata,classColName):
	while not ruleset.empty():
		rule = ruleset.get()
		for i, selector in enumerate(rule[0]):
			print(f"{inputdata.columns[i]} : {selector} ", end='')
		print(f" -> {classColName}:{rule[1]}") 

def rules_cn2(inputdata,significance,n):
	ruleSet = Queue()
	trainingSet = pd.DataFrame(inputdata)
	wildcards = tuple(set(trainingSet[col].unique()) for col in inputdata.columns[0:-1])
	atomicCombos = atomic_combos(wildcards)
	while len(trainingSet) >0:
		star = [wildcards]
		bestComplex=tuple(wildcards)
		while len(star) > 0:
			prim = calculate_prim(star,atomicCombos)
			bestComplex=compare_complex(trainingSet,prim,bestComplex,significance)
			star = best_complex(trainingSet,prim)
		category = covered_df(trainingSet, bestComplex)[inputdata.columns[-1]].value_counts().idxmax()
		ruleSet.put([bestComplex,category])
		excluded=list(covered_df(trainingSet,bestComplex).index.values)
		trainingSet.drop(excluded, axis=0, inplace=True)	
	return ruleSet
	