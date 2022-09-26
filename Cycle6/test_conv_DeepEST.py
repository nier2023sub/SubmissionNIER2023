import pandas as pd
import numpy as np
import sys

rep = sys.argv[1]
df = pd.read_csv("rep"+rep+"/test.csv")
df["confidence"] = df[["PredictedLabel0","PredictedLabel1","PredictedLabel2","PredictedLabel3","PredictedLabel4","PredictedLabel5","PredictedLabel6","PredictedLabel7","PredictedLabel8","PredictedLabel9"]].max(axis=1)
df["pred"] = df["label"]-df["SUT"]
df["pred"] = np.select([df["pred"]==0, df["pred"]!=0], ["Pass","Fail"], default=np.nan)
df["ID"] =  df.index

df_2 = pd.DataFrame(columns = ["ID","outcome","SUT","confidence"])
df_2["ID"] = df["ID"]
df_2["SUT"] = df["SUT"]
df_2["outcome"] = df["pred"]
df_2["confidence"] = df["confidence"]

df_2.to_csv("rep"+rep+"/test_DeepEST.csv", index=False)