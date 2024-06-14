import pandas as pd
import json

df = pd.read_csv("data/annotated.csv")

# Get unique objects
df = df.drop_duplicates("object_id")
df["next_to"] = df.next_to.apply(json.loads)

# For each unique object, check if the next_to relationship is symmetric.
for _, obj in df.iterrows():
    neighbors = obj["next_to"]
    for neighbor in neighbors:
        if not obj.object_id in df.loc[df["object_id"] == neighbor]["next_to"].iloc[0]:
            print(f"{obj.object_id} has neighbor {neighbor}")
            print(f"but {neighbor} only has {df.loc[df['object_id'] == neighbor]['next_to'].iloc[0]}")
