from numpy import empty
import pandas as pd
from statistics import quantiles
import json
import ast
'''
Pobiera las regul z pliku oraz dane do predykcji, a następnie przewiduje i dostarcza informacji o błędach
'''

rules_path = "generated_trees/rules-wine.txt" #input("Config file path ->")
predict_data = "datasets/predict-wine.csv"

InputData = pd.read_csv(predict_data,sep = ';')
GlobalAttributes = list(InputData.columns)[0:-1]
ClassName = InputData.columns[-1]

Bins=dict()
with open(rules_path,"r") as trees:
    for attr in GlobalAttributes:
        Bin = str(trees.readline())
        Bin = '{"'+Bin.replace(":",'":')[:-1]+'}'
        tmp = json.loads(Bin)
        Bins[attr]=tmp[attr]
    for col in InputData.columns[:-1]:
        InputData[col]=pd.cut(InputData[col],bins=Bins[col],labels=Bins[col][1:],right=False)
    Trees = trees.readlines()


treeRules = [[]]
treeArg = []
treeN=-1
ruleN=[]
for line in range(0,len(Trees)):
    if(Trees[line].startswith("tree")):
        treeN+=1
        treeRules.append([])
        subline = Trees[line].find(":")
        substr = Trees[line]
        substr = substr[subline+1:].strip()
        treeArg.append(ast.literal_eval(substr))
        ruleN.append(0)
        continue
    treeRules[treeN].append(ast.literal_eval(Trees[line]))
    ruleN[treeN]+=1

#Dokonanie predykcji za pomocą wygenerowanych zbiorów regul
#Zastosowanie zasad każdego z losowo stworzonych zbiorów reguł do wartości atrybutów ze zbioru testowego
#i przewidzenie wyniku. Zachowanie przewidywanych wyników.

depth = len(treeArg[0]) 
rowTreePred=[[]]
rowN=0
for row in InputData.itertuples(index=False):
    rowTreePred.append([])
    for tree in range(0, treeN+1):
        for rule in range(0,ruleN[tree]):
            flag = False
            for t in range(0,depth):
                arg = treeArg[tree][t]
                last = t == depth-1
                if(getattr(row,arg) in treeRules[tree][rule][0][t] and last):
                    rowTreePred[rowN].append(treeRules[tree][rule][1])
                    flag = True
                elif(getattr(row,arg) not in treeRules[tree][rule][0][t]):
                    break
            if flag: break
    rowN+=1

#Obliczenie liczby głosów na każdy przewidywany wynik.
votes=[[]]
classes=InputData[ClassName].unique()
for i in range(0,len(rowTreePred)):
    for clas in classes:
        votes[i].append(rowTreePred[i].count(clas))
    votes.append([])
    
#Uznanie wyniku z największą liczbą głosów jako ostateczną predykcję algorytmu lasu regul
rowPred=[]
for i in range(0, len(votes)-1):
    max_val=max(votes[i])
    max_id=votes[i].index(max_val)
    rowPred.append(classes[max_id])

#print(rowPred)
trueCount=0
falseCount=0
i=0
for row in InputData.itertuples(index=False):
    if(row[-1]==rowPred[i]):
        trueCount+=1
    else:
        falseCount+=1
    i+=1

print("Poprawna predykcja: ", trueCount)
print("Zła predykcja: ", falseCount)
print("Wynik procentowy: ", trueCount/(trueCount+falseCount)*100, "%")