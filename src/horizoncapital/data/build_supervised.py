import pandas as pd

def build_supervised(df, target, horizons):
    out=[]
    for h in horizons:
        tmp=df.copy()
        tmp["horizon"]=h
        tmp["y"]=tmp[target].shift(-h)
        out.append(tmp)
    return pd.concat(out, ignore_index=True).dropna()
