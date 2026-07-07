# import torch
# # a = torch.load("models/loss_val1.pth", weights_only=True)
# # print(a)

# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Исходная папка, которую вы скачали
# source_dir = "dataset" # внутри лежат папки 'cats' и 'dogs'
# # Куда мы хотим разложить структуру
# target_dir = "data"

# classes = ["no", "yes"]

# for cls in classes:
#     # Собираем все файлы из исходной папки класса
#     cls_dir = os.path.join(source_dir, cls)
#     all_images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     # Делим список файлов на train и val (80% на 20%)
#     train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
#     # Создаем новые папки назначения
#     os.makedirs(os.path.join(target_dir, "train", cls), exist_ok=True)
#     os.makedirs(os.path.join(target_dir, "val", cls), exist_ok=True)
    
#     # Копируем файлы в папку train
#     for img in train_imgs:
#         shutil.copy(os.path.join(cls_dir, img), os.path.join(target_dir, "train", cls, img))
        
#     # Копируем файлы в папку val
#     for img in val_imgs:
#         shutil.copy(os.path.join(cls_dir, img), os.path.join(target_dir, "val", cls, img))

# print("Данные успешно разделены на train и val!")