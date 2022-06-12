from asyncore import read
from numpy import empty
import pandas as pd
from statistics import quantiles
import json
import ast
'''
Pobiera las regul z pliku oraz dane do predykcji, a następnie przewiduje i dostarcza informacji o błędach
'''

config_path = "config.txt" #input("Config file path ->")
with open(config_path) as f:
    data = f.read()
config = json.loads(data)

InputData = pd.read_csv("datasets/Rice_Cammeo_Osmancik_calk.csv",sep = ';', dtype=config["DataFormat"])
GlobalAttributes = list(InputData.columns)[0:-1]
ClassName = InputData.columns[-1]

Bins=dict()
with open("generated_trees/rules.txt","r") as trees:
    for attr in GlobalAttributes:
        Bin = str(trees.readline())
        Bin = '{"'+Bin.replace(":",'":')[:-1]+'}'
        tmp = json.loads(Bin)
        Bins[attr]=tmp[attr]
    for col in InputData.columns[:-1]:
        InputData[col]=pd.cut(InputData[col],bins=Bins[col],labels=Bins[col][1:],right=False)
    Trees = trees.readlines()


treeRules = [[]]
treeN=-1
ruleN=[]
for line in range(0,len(Trees)):
    if(Trees[line].startswith("tree")):
        treeN+=1
        treeRules.append([])
        ruleN.append(0)
        continue
    treeRules[treeN].append(ast.literal_eval(Trees[line]))
    ruleN[treeN]+=1


#lista z argumentami ktore zostaly wylosowane dla kazdego drzewa
treeArg=[["Minor_Axis_Length", "Convex_Area", "Extent"],
["Area", "Eccentricity", "Convex_Area"],
["Perimeter", "Major_Axis_Length", "Eccentricity"],
["Perimeter", "Major_Axis_Length", "Convex_Area"],
["Extent", "Eccentricity",  "Major_Axis_Length"],
["Convex_Area", "Extent", "Area"],
["Perimeter", "Major_Axis_Length", "Eccentricity"],
["Extent", "Perimeter", "Convex_Area"],
["Area", "Major_Axis_Length", "Perimeter"],
["Convex_Area", "Major_Axis_Length", "Perimeter"],
["Convex_Area", "Major_Axis_Length", "Minor_Axis_Length"],
["Convex_Area", "Eccentricity", "Perimeter"],
["Minor_Axis_Length", "Area", "Extent"],
["Perimeter", "Major_Axis_Length", "Eccentricity"],
["Extent", "Major_Axis_Length", "Minor_Axis_Length"],
["Area", "Eccentricity", "Convex_Area"],
["Eccentricity", "Convex_Area", "Minor_Axis_Length"],
["Minor_Axis_Length", "Extent", "Area"],
["Perimeter", "Convex_Area", "Eccentricity"],
["Perimeter", "Minor_Axis_Length", "Major_Axis_Length"],
["Area", "Extent", "Minor_Axis_Length"],
["Perimeter", "Convex_Area", "Major_Axis_Length"],
["Convex_Area", "Perimeter", "Eccentricity"],
["Major_Axis_Length", "Extent", "Eccentricity"],
["Extent", "Area", "Convex_Area"]]

#Dokonanie predykcji za pomocą wygenerowanych zbiorów regul
#Zastosowanie zasad każdego z losowo stworzonych zbiorów reguł do wartości atrybutów ze zbioru testowego
#i przewidzenie wyniku. Zachowanie przewidywanych wyników.
rowTreePred=[[]]
rowN=0
for row in InputData.itertuples(index=False):
    rowTreePred.append([])
    for tree in range(0, treeN+1):
        arg0 = treeArg[tree][0]
        arg1 = treeArg[tree][1]
        arg2 = treeArg[tree][2]
        for rule in range(0,ruleN[tree]):
            if(getattr(row,arg0) in treeRules[tree][rule][0][0]):
                if(getattr(row,arg1) in treeRules[tree][rule][0][1]):
                    if(getattr(row,arg2) in treeRules[tree][rule][0][2]):
                        rowTreePred[rowN].append(treeRules[tree][rule][1])
                        break
    rowN+=1

#Obliczenie liczby głosów na każdy przewidywany wynik.
votes=[[]]
classes=['Cammeo','Osmancik']
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
    if(row.Class==rowPred[i]):
        trueCount+=1
    else:
        falseCount+=1
    i+=1

print("Poprawna predykcja: ", trueCount)
print("Zła predykcja: ", falseCount)
print("Wynik procentowy: ", trueCount/(trueCount+falseCount)*100, "%")