import sys
import os
import random

# 存放原始图片地址
data_base_dir = r"seg2/JPEG"
# 建立列表，用于保存图片信息
file_list = []
# 读取图片文件，并将图片地址、图片名和标签写到txt文件中
write_file_name = r'train.txt'
# 以只写方式打开write_file_name文件
# write_file = open(write_file_name, "w", encoding='utf-8')
write_file = open(write_file_name, "a", encoding='utf-8')
for file in os.listdir(data_base_dir):  # file为current_dir当前目录下图片名
    if file.endswith(".jpg"):  # 如果file以jpg结尾
        write_name = file  # 图片路径 + 图片名 + 标签
        file_list.append(write_name)  # 将write_name添加到file_list列表最后
        sorted(file_list)  # 将列表中所有元素随机排列
# 去掉后面的“.jpg”
i = 0
file_write = []
for item in file_list:
    file_write.append(file_list[i][:-4])
    i += 1
number_of_lines = len(file_write)  # 列表中元素个数
# 将图片信息写入txt文件中，逐行写入
print(file_write)
for current_line in range(number_of_lines):
    print(file_write, current_line)
    write_file.write(file_write[current_line] + '\n')
    # write_file.write(file_list[current_line] + ',' + 'malignant' + '\n')
# 关闭文件
write_file.close()
print('写入完成！')