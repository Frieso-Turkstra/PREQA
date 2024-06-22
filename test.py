from collections import Counter
import pandas as pd


# Dataframe with all tilts.
df = pd.read_csv("data/annotated.csv")

def occlusion_sort(obj):
    occlusion_levels = {'none': 0, 'partial': 1, 'severe': 2}
    object_area = (obj.x2 - obj.x1) * (obj.y2 - obj.y1)
    return (occlusion_levels[obj.occlusion], -object_area)

def find_optimal_view(df, object_id):
    # Find all images that display the object.
    images = df[df.object_id == object_id]

    # Sort them by level of occlusion: none < partial < severe.
    # If same level of occlusion, choose the largest bounding box.
    sorted_images = sorted(images.iterrows(), key=lambda x: occlusion_sort(x[1]))
    optimal_view = sorted_images[0][1].filename
    return optimal_view

# Dataframe with no tilt.
df["tilt"] = df["filename"].apply(lambda x: x.split('-')[2].split(".")[0])
df_1 = df[df.tilt == "1"]


num_unique_objects = len(df.object_id.unique())
num_unique_objects_1 = len(df_1.object_id.unique())
print(Counter(df.occlusion))
print(Counter(df_1.occlusion))

optimal_views = []
for i in range(90):
    optimal_views.append(find_optimal_view(df, i))

print(sum([1 for view in optimal_views if view.split("-")[2] == "1.jpg"]))
exit()

