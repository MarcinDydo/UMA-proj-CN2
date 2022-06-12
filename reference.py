import pandas as pd
import numpy as np
from itertools import combinations
'''
This is simple implementation of CN2 algorithm for discrete values of parameters
'''
def is_covered(row, complex):
	#returns if row of training data is covered by complex k
	for i, arg in enumerate(row):
		if i >= len(complex): 
			break
		if arg not in complex[i]:
			return False
	return True	

def covered_df(data, complex):
	#returns dataframe covered by complex
	covered_data=[] 
	for row in data.iloc:
		if is_covered(row, complex):	
			covered_data.append(row)
	return pd.DataFrame(covered_data)

def rule_entropy(covered_data):	
	#returns measure of relevance of complex calculeted from covered df - the higher the better
	if len(covered_data)==0: return -1
	class_series = covered_data[ClassColumnName]
	num_instances = len(class_series)
	class_counts = class_series.value_counts()
	class_probabilities = class_counts.divide(num_instances)
	log2_of_classprobs = np.log2(class_probabilities)
	plog2p = class_probabilities.multiply(log2_of_classprobs)
	entropy = plog2p.sum()
	return entropy

def atomic_complexes(wildcards):
	#returns all atomic complexes for input data using wildcards
	atomset = []
	for i, wildcard in enumerate(wildcards):
		combos = [] #list of all possible combinations
		for l in range(1,len(wildcard)):
			x=combinations(wildcard, l)
			combos+=[set(y) for y in x]
		for c in combos:
			complex = list(wildcards) #assign wildcards
			complex[i]=c #assign combination
			atomset.append(complex)
	return atomset

def calculate_prim(star,atomic,n):
	#returns calculated intersection of atomic complexes and star, deletes empty complexes and those that are in star
	prim=[]
	for complex in star: 
		for atomcomplex in atomic: 
			temp=[]
			for i in range(0,n):
				temp.append(atomcomplex[i].intersection(complex[i]))
			prim.append(temp)
            

		
	j=0
	while j < len(prim):
		if prim[j] in star: 
			del prim[j]
			continue
		for i in range(len(prim[j])):
			if len(prim[j][i])==0:
				del prim[j]
				j-=1
				break
		j+=1	

	return prim

def compare_complex(trainer, prim, best, significance):
	#uses heuristics to calculate best complex from set
	newbest=best
	for complex in prim:
		if len(covered_df(trainer,complex))>significance and rule_entropy(covered_df(trainer,complex)) > rule_entropy(covered_df(trainer,newbest)): 
			newbest = complex
	return newbest

def best_complex(trainer,prim):
	if len(prim)==0: return []
	best=prim[0]
	for complex in prim:
		if rule_entropy(covered_df(trainer,complex)) > rule_entropy(covered_df(trainer,best)):
			best=complex
	return [best]

def print_ruleset(ruleset,inputdata):
	for rule in ruleset:
		for i, selector in enumerate(rule[0]):
			print(f"{inputdata.columns[i]} : {selector} ", end='')
		print(f" -> {ClassColumnName}:{rule[1]}") 

def rules_cn2(inputdata,significance,n):
	wildcards, star, ruleSet = ([] for i in range(3))
	for col in inputdata.columns[0:-1]:
		wildcards.append(set(inputdata[col].unique()))
	atomicComplexes = atomic_complexes(wildcards)
	trainingSet = pd.DataFrame(inputdata)
	while len(trainingSet) > 0:
		bestComplex=list(wildcards)
		star.append(list(wildcards))
		while len(star) > 0:
			prim = calculate_prim(star,atomicComplexes,n)
			bestComplex=compare_complex(trainingSet,prim,bestComplex,significance)
			star = best_complex(trainingSet,prim)
		category = covered_df(trainingSet, bestComplex)[ClassColumnName].value_counts().idxmax()
		ruleSet.append([bestComplex,category])
		excluded=list(covered_df(trainingSet,bestComplex).index.values)
		trainingSet.drop(excluded, axis=0, inplace=True)
	return(ruleSet)
	

InputData = pd.read_csv("basic.csv",sep = ';')
NumberOfAttributes = len(InputData.columns)-1
Significance=1
ClassColumnName='Class'
Rules=rules_cn2(InputData,Significance,NumberOfAttributes)

print_ruleset(Rules,InputData)
