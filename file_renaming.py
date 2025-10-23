#This file is for renaming the files so there won't be trouble when putting them all into an array later on
#Might have to doublecheck if this is allowed first (i don't see why it wouldn't be)

#ONLY RUN THIS ONCE, otherwise we get big long names

import os

parent_dir = "data/Bird_Species_Dataset"

#in each directory
for class_dir in os.listdir(parent_dir):
    class_path = os.path.join(parent_dir, class_dir)
    if os.path.isdir(class_path):
        #for each file
        for img_file in os.listdir(class_path):
            if img_file.endswith(".jpg"):
                #rename to (bird type)_(old file name).jpg
                new_name = f"{class_dir}_{img_file}"

                old_path = os.path.join(class_path, img_file)
                new_path = os.path.join(class_path, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

print("All files renamed successfully!")