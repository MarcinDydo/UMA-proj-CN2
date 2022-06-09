import pandas as pd
import cn_two
import random
'''
Program na ume, wykorzystuje cn2 do generowania zbiorów reguł
'''
Path = input("Specify path to training data (csv) ->")
Significance = int(input("Specify significance (minimum coverage) ->"))
NumberOfTrees = int(input("Specify number of trees ->"))
NumberOfAttributes = int(input("Specify number of attributes per tree (optimal is sqrt[n]) ->"))

InputData = pd.read_csv(Path,sep = ';')
GlobalAttributes = list(InputData.columns)[0:-1]
ClassColumnName = list(InputData.columns)[-1]
for i in range(0,NumberOfTrees):
    localAttributes = random.sample(GlobalAttributes,NumberOfAttributes)
    localAttributes.append(ClassColumnName)
    localDataFrame = pd.DataFrame(InputData[localAttributes]).sample(len(InputData),replace=True)
    rules=cn_two.rules_cn2(localDataFrame,Significance,NumberOfAttributes,ClassColumnName)
    print(f"Tree {i}")
    cn_two.print_ruleset(rules,InputData,ClassColumnName)
