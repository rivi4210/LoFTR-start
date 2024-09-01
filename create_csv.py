import os
import pandas as pd
def generate_image_pairs(directory_path, distance, output_csv):
    files = sorted([f for f in os.listdir(directory_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))])
    
    pairs = []
    num_files = len(files)
    for i in range(num_files - distance):
        # print(num_files-5)
        # for j in range(i + distance, num_files):
        print(i)
        img1 = files[i]
        img2 = files[i + distance]
        pairs.append([os.path.join(directory_path, img1), os.path.join(directory_path, img2)])
    df = pd.DataFrame(pairs, columns=['query_name', 'ref_name'])
    df.to_csv(output_csv, index=False)
    print(f"CSV file with image pairs has been saved to {output_csv}")

directory_path = 'C:\בוטקמפ\project\offset_0_None'
distance = 5  
output_csv = 'image_pairs.csv'

generate_image_pairs(directory_path, distance, output_csv)


