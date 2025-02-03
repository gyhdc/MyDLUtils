import random
from shutil import copyfile
import shutil
import os 

def split_data(data_dir,target_dir,label,sep=[0.7,0.25,0.05]):
    os.makedirs(os.path.join(target_dir, 'train',label), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val',label), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test',label), exist_ok=True)
    import glob

    image_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    random.shuffle(image_files)

    # 计算分割点
    total_images = len(image_files)
    train_split = int(sep[0] * total_images)
    val_split = int(sep[1] * total_images)

    # 将图片复制到对应的目标文件夹
    for i, image_file in enumerate(image_files):
        if i < train_split:
            dest_dir = os.path.join(target_dir, 'train',label)
        elif i < train_split + val_split:
            dest_dir = os.path.join(target_dir, 'val',label)
        else:
            dest_dir = os.path.join(target_dir, 'test',label)
        
        # 获取图片文件名并复制到目标文件夹
        image_filename = os.path.basename(image_file)
        copyfile(image_file, os.path.join(dest_dir, image_filename))



def clear_folder(folder_path):
    # 检查目标文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历目标文件夹内的所有文件和子文件夹
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # 如果是文件，则直接删除
            if os.path.isfile(item_path):
                os.remove(item_path)
            # 如果是子文件夹，则递归清除子文件夹内的内容
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
if __name__ == '__main__':
    total_normal_path=r"D:\Desktop\深度学习\7.道路坑洞识别\data\normal"
    total_p_path=r"D:\Desktop\深度学习\7.道路坑洞识别\data\potholes"
    target_dir=r'D:\Desktop\深度学习\7.道路坑洞识别\data\images'
    clear_folder(target_dir)
    split_data(total_normal_path,"normal")
    split_data(total_p_path,"potholes")
    