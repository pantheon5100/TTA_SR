import argparse
import torch
import os


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DualSR')

        # Paths
        self.parser.add_argument('--input_dir', '-i', type=str, default='test/LR', help='path to image input directory.')
        self.parser.add_argument('--output_dir', '-o', type=str, default='results' , help='path to image output directory.')
        self.parser.add_argument('--kernel_dir', '-k', type=str, default='', help='path to grand-truth kernel directory.')
        self.parser.add_argument('--gt_dir', '-g', type=str, default='', help='path to grand-truth image.')
        
        self.parser.add_argument('--input_image_path', default='', help='path to one specific image file')
        self.parser.add_argument('--kernel_path', default=None, help='path to one specific kernel file')
        self.parser.add_argument('--gt_path', default=None, help='path to one specific ground truth file')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=128, help='crop size for HR patch')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
        self.parser.add_argument('--scale_factor', type=int, default=2, help='The upscaling scale factor')
        self.parser.add_argument('--scale_factor_downsampler', type=float, default=0.5, help='scale factor for downsampler')
        
        #Lambda Parameters
        self.parser.add_argument('--lambda_cycle', type=int, default=5, help='lambda parameter for cycle consistency loss')
        self.parser.add_argument('--lambda_interp', type=int, default=2, help='lambda parameter for masked interpolation loss')
        self.parser.add_argument('--lambda_regularization', type=int, default=2, help='lambda parameter for downsampler regularization term')
        
        # Learning rates
        self.parser.add_argument('--lr_G_UP', type=float, default=0.001, help='initial learning rate for upsampler generator')
        self.parser.add_argument('--lr_G_DN', type=float, default=0.0002, help='initial learning rate for downsampler generator')
        self.parser.add_argument('--lr_D_DN', type=float, default=0.0002, help='initial learning rate for downsampler discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')
        
        # Iterations
        self.parser.add_argument('--num_iters', type=int, default=2000, help='number of training iterations')
        self.parser.add_argument('--switch_iters', type=int, default=1000, help='number of training iterations')
        self.parser.add_argument('--eval_iters', type=int, default=100, help='for debug purpose')
        self.parser.add_argument('--plot_iters', type=int, default=200, help='for debug purpose')
        self.parser.add_argument('--debug', action='store_true', help='plot intermediate results')
        self.parser.add_argument('--model_save_iter', type=int, default=500, help='for debug purpose')

        self.parser.add_argument('--test_only', action="store_true")
        
        
        self.parser.add_argument('--training_strategy', type=str, default='000')
        
        SUPPORT_TRAIN_MODE=[
            "original",
            "bicubic",
            "dual_path",
            ]
        self.parser.add_argument('--training_mode', type=str, choices=SUPPORT_TRAIN_MODE, default='original')
        
        self.parser.add_argument(
            '--each_batch_img_size', type=int, default=5, help='batch size for training')
        # Source model
        SUPPORT_SOURCE_MODEL = [
            "swinir",
            "rcan",
            "edsr"
        ]
        self.parser.add_argument('--source_model', default='swinir',
                                 choices=SUPPORT_SOURCE_MODEL, help='path to one specific image file')
        
        # for swinir source model
        SUPPORT_SWINIR_MODE=[
            "classicalSR_s1",
            "lightweightSR",
            # "realSR", # not complete
        ]
        self.parser.add_argument('--swinir_task', type=str, default='classicalSR_s1', choices=SUPPORT_SWINIR_MODE)
        
        


        # for pretrain gdn
        self.parser.add_argument('--pretrained_gdn_only', action='store_true')
        self.parser.add_argument('--pretrained_gdn', type=str, default='')
        self.parser.add_argument('--pretrained_gdn_with_imgenet', action='store_true')
        self.parser.add_argument('--pretrained_gdn_num_iters', type=int, default=3000)

        # for gup updating
        self.parser.add_argument('--finetune_gdn', action='store_true')
        

        self.conf = self.parser.parse_args()
        
        # if not os.path.exists(self.conf.output_dir):
        #     os.makedirs(self.conf.output_dir)



    def get_config(self, img_name):
        self.conf.abs_img_name = os.path.splitext(img_name)[0]
        self.conf.input_image_path = os.path.join(self.conf.input_dir, img_name)
        self.conf.kernel_path = os.path.join(self.conf.kernel_dir, self.conf.abs_img_name + '.mat') if self.conf.kernel_dir != '' else None
        # self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None

        if "Set5" in self.conf.gt_dir:
            # self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name[:-6]+".png") if self.conf.gt_dir != '' else None
            self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name)
        
        elif "Manga109" in self.conf.gt_dir:
            self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name[:-6]+".png")
        elif "my_RealSR" in self.conf.gt_dir:
            self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) 
        
        else:
            # for set14 and other dataset
            self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) 
            

        # self.conf.gt_path = os.path.join(self.conf.gt_dir, img_name) if self.conf.gt_dir != '' else None

        print('*' * 60)
        print('input image: \'%s\'' %self.conf.input_image_path)
        print('grand-truth image: \'%s\'' %self.conf.gt_path)
        # print('grand-truth kernel: \'%s\'' %self.conf.kernel_path)
        return self.conf




