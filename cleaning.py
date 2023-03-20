#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd

# In[ ]:

def fill_area(row):
    if row["adresse"] == "":
        return row["area"]
    else:
        return row["adresse"]

# In[ ]:

def rent_bin(row):
    if row["rent"] < 600:
        return "below 600"
    elif (row["rent"] >= 600) & (row["rent"] < 1000):
        return "600 to 1000"
    elif (row["rent"] >= 1000) & (row["rent"] < 1500):
        return "1000 to 1500"
    elif (row["rent"] >= 1500) & (row["rent"] < 2000):
        return "1500 to 2000"
    elif (row["rent"] >= 2000) & (row["rent"] < 3000):
        return "2000 to 3000"
    else:
        return "over 3000"

# In[ ]:

def cleaning():
    import pandas as pd
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df = pd.read_csv("data raw.csv")
    df["adresse"] = df["adresse"].str.replace("berlin","Berlin")
    adresse,area = [],[]
    for i in df["adresse"]:
        adresse.append(i.split("Berlin ")[0].strip(" "))
        try:
            area.append(i.split("(")[1].strip(" "))
        except:
            area.append("")
    size = []
    for i in df["size"]:
        size.append(i.split(" m")[0])
    room = []
    for i in df["rooms"]:
        room.append(i.split(" Z")[0])
    df["rooms"] = room
    df["rooms"] = df["rooms"].astype(float)
    df["size"] = size
    df["size"] = df["size"].astype(float)
    df["adresse"] = adresse
    df["area"] = area
    df["adresse"] = df["adresse"].str.replace(",","")
    df["rent"] = df["rent"].str.replace("€","")
    df["rent"] = df["rent"].str.replace(".","")
    df["rent"] = df["rent"].str.replace(",",".")
    df["rent"] = df["rent"].astype(float)
    df["area"] = df["area"].str.replace("(","").str.replace(")","").str.replace("/ ","")
    df["adresse"] = df.apply(fill_area, axis = 1)
    df["rent_range"] = df.apply(rent_bin, axis = 1)
    df.to_csv("data clean.csv", index = False)

