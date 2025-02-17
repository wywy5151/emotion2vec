import os
import pandas as pd

# 指定要读取的文件夹路径
folder_path = r'C:\Users\ROG\Desktop\音频数据\0822\split_0827_1'
file_name = []
root_path = []
# 使用os.walk遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(file)
        file_name.append(file)
        path = os.path.join(root, file)
        root_path.append(path)
df = pd.DataFrame({'File Name': file_name, 'Path': root_path})
excel_path = r'C:\Users\ROG\Desktop\音频数据\0822\split_0827_1\file_names.xlsx'
df.to_excel(excel_path, index=False)