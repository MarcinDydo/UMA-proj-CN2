import pandas as pd
from statistics import quantiles
import cn_two
import random
import json
'''
Program na ume, wykorzystuje cn2 do generowania zbiorów reguł
'''
config_path = "config.txt" #input("Config file path ->")
output_path = "rules-test.txt" #input("Output file path ->")

with open(config_path) as f:
    data = f.read()
config = json.loads(data)

InputData = pd.read_csv(config["Path"],sep = ';', dtype=config["DataFormat"])
GlobalAttributes = list(InputData.columns)[0:-1]
ClassName = InputData.columns[-1]

f = open(output_path,"w")
for col in GlobalAttributes:
        wildcard = list(InputData[col].unique())
        if len(wildcard) > config["MaxNumOfIntervals"]:
            bins=[0]
            bins+=[int(q) for q in quantiles(wildcard,n=config["MaxNumOfIntervals"]+1)]
            bins[1]-=1
            bins[-1]=max(wildcard)+1
            f.write(col+": "+str(bins)+'\n')
            InputData[col]=pd.cut(InputData[col],bins=bins,labels=bins[1:],right=False)
f.close()   

for i in range(0,config["NumberOfTrees"]):
    localAttributes = random.sample(GlobalAttributes,config["NumberOfAttributes"])
    localAttributes.append(ClassName)
    localDataFrame = pd.DataFrame(InputData[localAttributes]).sample(config["TrainerSize"],replace=True)
    print(f"Working on tree {i}...")
    rules=cn_two.rules_cn2(localDataFrame,config["Significance"],config["NumberOfAttributes"])
    cn_two.write_rules(rules,output_path,i,localDataFrame)
    #cn_two.print_ruleset(rules,localDataFrame,ClassName)