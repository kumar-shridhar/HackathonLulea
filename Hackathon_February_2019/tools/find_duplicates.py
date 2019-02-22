import glob
import tqdm
import os

rvl_cdip_path = "/home/koelscha/Downloads/rvl-cdip/"
tobacco_path = "/home/koelscha/Downloads/Tobacco/"

rvl_cdip_files = glob.glob(rvl_cdip_path + "**/*.tif", recursive=True)
tobacco_files = glob.glob(tobacco_path + "**/*.tif", recursive=True)

duplicates = []

for f in tqdm.tqdm(tobacco_files):
    name = os.path.basename(f)
    for f_ in rvl_cdip_files:
        if name == os.path.basename(f_):
            duplicates.append(name)
            break

out = open(rvl_cdip_path + "duplicates.txt", "w")

for dup in duplicates:
    out.write(dup + "\n")
out.close()
