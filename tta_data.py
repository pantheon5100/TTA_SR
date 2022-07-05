import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imresize import imresize
from PIL import Image
from tta_util import create_gradient_map, im2tensor, create_probability_map, nn_interpolation

def create_dataset(conf):
    dataset = DataGenerator(conf)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader

def read_image(path: str) -> np.array:
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    return im


class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def __init__(self, conf):
        np.random.seed(0)
        self.conf = conf
        
        print('*' * 60 + '\nPreparing data ...')
        
        # Default shapes
        # self.g_input_shape = conf.input_crop_size
        # self.d_input_shape = int(conf.input_crop_size * conf.scale_factor_downsampler)
        self.g_input_shape = conf.g_input_shape
        self.d_input_shape = conf.d_input_shape
        
        # Read input image
        input_image = read_image(conf.input_image_path)
        self.input_image = input_image / 255. if conf.source_model=="swinir" else input_image
        
        self.shave_edges(scale_factor=conf.scale_factor_downsampler, real_image=False)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return self.conf.num_iters * self.conf.batch_size

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        g_in = self.next_crop(for_g=True, idx=idx)
        d_in = self.next_crop(for_g=False, idx=idx)
        d_bq = imresize(im=d_in, scale_factor=int(1/self.conf.scale_factor_downsampler), kernel='cubic')
        
        # import ipdb; ipdb.set_trace()
        return {'HR':im2tensor(g_in).squeeze(), 'LR':im2tensor(d_in).squeeze(), 'LR_bicubic':im2tensor(d_bq).squeeze()}

    def next_crop(self, for_g, idx):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        #top = np.random.randint(0, self.in_rows - size)
        #left = np.random.randint(0, self.in_cols - size)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        #if not for_g:  # Add noise to the image for d
        #    crop_im += np.random.randn(*crop_im.shape) / 255.0
        return crop_im

    def make_list_of_crop_indices(self, conf):
        iterations = conf.num_iters * conf.batch_size
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor_downsampler)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, scale_factor):
        # Create loss maps for input image and downscaled one
        loss_map_big = create_gradient_map(self.input_image)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        # Create corresponding probability maps
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        return top - top % 2, left - left % 2

# TODO:
# 1. make all thing to be a list


def create_dataset_for_image_agnostic_gdn(conf):
    dataset = DataGenerator_ALLIMG(conf)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, num_workers=3)
    return dataloader

class DataGenerator_ALLIMG(Dataset):
    """
    The data generator loads all image at once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """

    def __init__(self, conf):
        np.random.seed(0)
        self.conf = conf
        
        print('*' * 60 + '\nPreparing data ...')
        
        # Default shapes
        # self.g_input_shape = conf.input_crop_size
        # self.d_input_shape = int(conf.input_crop_size * conf.scale_factor_downsampler)
        self.g_input_shape = conf.g_input_shape
        self.d_input_shape = conf.d_input_shape
        
        # Read input image
        img_dir_list = os.listdir(conf.input_dir)
        self.num_imgs = len(img_dir_list)
        assert self.conf.each_batch_img_size <= self.num_imgs
        
        self.all_img = []
        self.in_rows, self.in_cols = [], []
        self.crop_indices_for_g, self.crop_indices_for_d = [], []
        
        

        data_dir = "imagenet_data.pt"
        if os.path.exists(data_dir):
            image_data = torch.load(data_dir)
            self.all_img = image_data["all_img"]
            self.in_rows = image_data["in_rows"]
            self.in_cols = image_data["in_cols"]
            self.crop_indices_for_g = image_data["crop_indices_for_g"]
            self.crop_indices_for_d = image_data["crop_indices_for_d"]
            # import ipdb; ipdb.set_trace()
            return
        
        for img_dir in img_dir_list:
            print(1)
            # import ipdb; ipdb.set_trace()
            img = read_image(os.path.join(conf.input_dir, img_dir))
            img = img / 255. if conf.source_model=="swinir" else img
            
            # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
            img = img[10:-10, 10:-10, :]
            # Crop pixels for the shape to be divisible by the scale factor
            sf = int(1 / conf.scale_factor_downsampler)
            shape = img.shape
            img = img[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else img
            img = img[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else img


            self.all_img.append(img)
            self.in_rows.append(img.shape[0])
            self.in_cols.append(img.shape[1])
            
            crop_indices_for_g, crop_indices_for_d = self.make_list_of_crop_indices(conf=conf, img=img)
            self.crop_indices_for_g.append(crop_indices_for_g)
            self.crop_indices_for_d.append(crop_indices_for_d)

        image_data = {
            "all_img": self.all_img,
            "in_rows": self.in_rows,
            "in_cols": self.in_cols,
            "crop_indices_for_g": self.crop_indices_for_g,
            "crop_indices_for_d": self.crop_indices_for_d,
        }
        torch.save(image_data, "imagenet_data.pt")
        # Create prob map for choosing the crop
        # self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)
        

        # self.input_image = read_image(conf.input_image_path) / 255.
        # self.shave_edges(scale_factor=conf.scale_factor_downsampler, real_image=False)

        # self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # # Create prob map for choosing the crop
        # self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(conf=conf)

    def __len__(self):
        return self.conf.num_iters * self.conf.batch_size

    def __getitem__(self, idx):
        """Get a crop for both G and D """
        
        # generate seletion img
        img_list = [i for i in range(self.num_imgs)]
        img_selection = random.sample(img_list, self.conf.each_batch_img_size)
        
        g_in = self.next_crop(for_g=True, idx=idx, img_selection=img_selection)
        d_in = self.next_crop(for_g=False, idx=idx, img_selection=img_selection)
        # d_bq = imresize(im=d_in, scale_factor=int(1/self.conf.scale_factor_downsampler), kernel='cubic')
    
        # return {'HR':im2tensor(g_in).squeeze(), 'LR':im2tensor(d_in).squeeze(), 'LR_bicubic':im2tensor(d_bq).squeeze()}
        
        return {
            'HR':torch.FloatTensor(np.transpose(g_in, (0, 3, 1, 2))), 
            'LR':torch.FloatTensor(np.transpose(d_in, (0, 3, 1, 2)))
            }

    def next_crop(self, for_g, idx, img_selection):
        """Return a crop according to the pre-determined list of indices. Noise is added to crops for D"""
        size = self.g_input_shape if for_g else self.d_input_shape
        
        all_crop_img = []
        # for n in range(self.num_imgs):
        for n in range(self.conf.each_batch_img_size):
            top, left = self.get_top_left(size, for_g, idx, img_selection[n])
            crop_im = self.all_img[img_selection[n]][top:top + size, left:left + size, :]
            if crop_im.shape != (size, size, 3):
                import ipdb; ipdb.set_trace()
            all_crop_img.append(crop_im)
        # import ipdb; ipdb.set_trace()
        all_crop_img = np.array(all_crop_img)
        
        return all_crop_img

    def make_list_of_crop_indices(self, conf, img):
        iterations = conf.num_iters * conf.batch_size
        prob_map_big, prob_map_sml = self.create_prob_maps(scale_factor=conf.scale_factor_downsampler, img=img)
        crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
        crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, scale_factor, img):
        # Create loss maps for input image and downscaled one
        # loss_map_big = create_gradient_map(self.input_image)
        # loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'))
        loss_map_big = create_gradient_map(img)
        loss_map_sml = create_gradient_map(imresize(im=img, scale_factor=scale_factor, kernel='cubic'))
        # Create corresponding probability maps
        prob_map_big = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml = create_probability_map(nn_interpolation(loss_map_sml, int(1 / scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        """Shave pixels from edges to avoid code-bugs"""
        # Crop 10 pixels to avoid boundaries effects in synthetically generated examples
        if not real_image:
            self.input_image = self.input_image[10:-10, 10:-10, :]
        # Crop pixels for the shape to be divisible by the scale factor
        sf = int(1 / scale_factor)
        shape = self.input_image.shape
        self.input_image = self.input_image[:-(shape[0] % sf), :, :] if shape[0] % sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(shape[1] % sf), :] if shape[1] % sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx, img_idx):
        """Translate the center of the index of the crop to it's corresponding top-left"""
        
        center = self.crop_indices_for_g[img_idx][idx] if for_g else self.crop_indices_for_d[img_idx][idx]
        row, col = int(center / self.in_cols[img_idx]), center % self.in_cols[img_idx]
        top, left = min(max(0, row - size // 2), self.in_rows[img_idx] - size), min(max(0, col - size // 2), self.in_cols[img_idx] - size)
        # Choose even indices (to avoid misalignment with the loss map for_g)
        # import ipdb; ipdb.set_trace()

        top = top - top % 2
        left = left - left % 2

        if (self.in_rows[img_idx] - top < size) or (self.in_cols[img_idx] - left < size):
            import ipdb; ipdb.set_trace()
        

        return top, left


