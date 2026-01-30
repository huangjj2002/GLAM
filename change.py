import pandas as pd

csv_in = "/mnt/f/new_embed_data/embed_data.csv"
csv_out = "/mnt/f/new_embed_data/embed_data_fixed.csv"

df = pd.read_csv(csv_in)

def fix_image_id(x):
    if isinstance(x, str) and x.endswith(".jpg"):
        return x.replace(".jpg","")
    return x

df["image_id"] = df["image_id"].apply(fix_image_id)

df.to_csv(csv_out, index=False)
print("Saved to:", csv_out)
