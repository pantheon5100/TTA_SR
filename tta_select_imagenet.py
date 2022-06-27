
import os
from PIL import Image
import tqdm
image_dir = "/media/philipp/ssd2_4tb/data/imagenet_pytorch/train"
image_save_dir = '/media/philipp/ssd1_2tb/nax_projects/super_resolution/dataset/select_imagenet'


for image_file_dir in tqdm.tqdm(os.listdir(image_dir)):
    image_names_file_path = os.path.join(image_dir, image_file_dir)
    for image_name in tqdm.tqdm(os.listdir(image_names_file_path)):
        # print(image_name)
        image_path = os.path.join(image_dir, image_file_dir, image_name)
        img = Image.open(image_path)
        h = img.height
        w = img.width
        # import ipdb;ipdb.set_trace();
        save_path = os.path.join(image_save_dir, image_name)
        if h>=500 and w>=500:
            # print(save_path)
            img.save(save_path)
            
    
    