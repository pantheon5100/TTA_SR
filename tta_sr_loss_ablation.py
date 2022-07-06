import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.io import loadmat

import loss
import tta_util as util
import tta_util_calculate_psnr_ssim as util_calculate_psnr_ssim
from tta_model import networks
from tta_model.get_model import get_model


def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)

    # im = imageio.imread(path)

    return im


class TTASR:

    def __init__(self, conf):
        # Fix random seed
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True  # slightly reduces throughput

        # Acquire configuration
        self.conf = conf

        # Define the networks
        # 1. Define and Load the pretrained swinir
        self.G_UP = get_model(conf)
        self.D_DN = networks.Discriminator_DN().cuda()
        # 2. Define the down sample network
        self.G_DN = networks.Generator_DN(downsample_stride=conf.scale_factor, first_layer_padding="same").cuda()
        # self.G_DN = networks.Generator_DN(
        #     downsample_stride=conf.scale_factor, first_layer_padding=5).cuda()

        # Losses
        self.criterion_gan = loss.GANLoss().cuda()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_interp = torch.nn.L1Loss()
        self.regularization = loss.DownsamplerRegularization(
            conf.scale_factor_downsampler, self.G_DN.G_kernel_size)

        # Initialize networks weights
        self.D_DN.apply(networks.weights_init_D_DN)

        # Optimizers
        self.optimizer_G_UP = torch.optim.Adam(
            self.G_UP.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
        self.optimizer_D_DN = torch.optim.Adam(
            self.D_DN.parameters(), lr=conf.lr_D_DN, betas=(conf.beta1, 0.999))
        self.optimizer_G_DN = torch.optim.Adam(
            self.G_DN.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))

        # TODO: below need to rewrite
        # Read input image
        self.read_image(self.conf)

        # if self.gt_kernel is not None:
        #     self.gt_kernel = np.pad(self.gt_kernel, 1, 'constant')
        #     self.gt_kernel = util.kernel_shift(self.gt_kernel, sf=conf.scale_factor)
        #     self.gt_kernel_t = torch.FloatTensor(self.gt_kernel).cuda()

        #     self.gt_downsampled_img_t = util.downscale_with_kernel(self.in_img_cropped_t, self.gt_kernel_t)
        #     self.gt_downsampled_img = util.tensor2im(self.gt_downsampled_img_t)

        # debug variables
        self.debug_steps = []
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None

        if self.conf.debug:
            self.loss_GANs = []
            self.loss_cycle_forwards = []
            self.loss_cycle_backwards = []
            self.loss_interps = []
            self.loss_Discriminators = []

        self.iter = 0

        self.train_G_DN_switch = False
        self.train_G_UP_switch = False
        self.train_D_DN_switch = True

        self.reshap_train_data = False

        # train strategy
        # basicly we use backward loss by default
        # [gan loss, train gdn forward, train gup forward, train gdn backward, train gup backward]

        train_strategy = list(self.conf.training_strategy)
        self.train_strategy = [int(i) for i in train_strategy]

        if len(self.train_strategy) == 3:
            self.train_strategy.append(1)
            self.train_strategy.append(1)

    def read_image(self, conf):
        if conf.input_image_path:

            self.in_img = read_image(conf.input_image_path)
            if conf.source_model == "edsr" or conf.source_model == "rcan":

                self.in_img_t = torch.FloatTensor(np.transpose(
                    self.in_img, (2, 0, 1))).unsqueeze(0).cuda()
                pass
            else:

                self.in_img_t = util.im2tensor(self.in_img).cuda()

            b_x = self.in_img_t.shape[2] % conf.scale_factor
            b_y = self.in_img_t.shape[3] % conf.scale_factor
            self.in_img_cropped_t = self.in_img_t[..., b_x:, b_y:]

        self.gt_img = read_image(
            conf.gt_path) if conf.gt_path is not None else None
        self.gt_kernel = loadmat(conf.kernel_path)[
            'Kernel'] if conf.kernel_path is not None else None
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None

    def reset_ddn(self):
        self.G_DN = networks.Generator_DN(
            downsample_stride=self.conf.scale_factor).cuda()
        self.optimizer_D_DN = torch.optim.Adam(
            self.D_DN.parameters(), lr=self.conf.lr_D_DN, betas=(self.conf.beta1, 0.999))

    def train(self, data):
        self.G_UP.train()

        self.set_input(data)

        loss = {}
        loss_train_G_DN = self.train_G_DN()
        loss_train_G_UP = self.train_G_UP()
        loss_train_D_DN = self.train_D_DN()

        loss.update(loss_train_G_DN)
        loss.update(loss_train_G_UP)
        loss.update(loss_train_D_DN)

        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1

        return loss

    def set_input(self, data):
        self.real_HR = data['HR'].cuda()
        self.real_LR = data['LR'].cuda()
        # self.real_LR_bicubic = data['LR_bicubic']
        # import ipdb; ipdb.set_trace()
        if self.reshap_train_data:
            self.real_HR = self.real_HR.reshape([self.real_HR.size(
                0)*self.real_HR.size(1), self.real_HR.size(2), self.real_HR.size(3), self.real_HR.size(4)])
            self.real_LR = self.real_LR.reshape([self.real_LR.size(
                0)*self.real_LR.size(1), self.real_LR.size(2), self.real_LR.size(3), self.real_LR.size(4)])

    def train_G_DN(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_GAN = 0
        self.loss_regularization = 0

        if self.train_G_DN_switch:

            # Turn off gradient calculation for G_UP
            util.set_requires_grad([self.G_UP], False)
            # Turn on gradient calculation for G_DN
            util.set_requires_grad([self.G_DN], True)
            util.set_requires_grad([self.D_DN], False)

            # Reset gradient valus
            # self.optimizer_G_UP.zero_grad()
            self.optimizer_G_DN.zero_grad()

            # Forward path
            if self.train_strategy[1] == 1:
                # import ipdb; ipdb.set_trace()
                self.fake_HR = self.G_UP(self.real_LR)
                self.rec_LR = self.G_DN(self.fake_HR)
                # import ipdb; ipdb.set_trace()
                loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(
                    self.real_LR, self.rec_LR)) * self.conf.lambda_cycle

            if self.train_strategy[3] == 1 or self.train_strategy[0] == 1:
                self.fake_LR = self.G_DN(self.real_HR)

            # Backward path
            if self.train_strategy[3] == 1:
                # import ipdb; ipdb.set_trace()
                self.rec_HR = self.G_UP(self.fake_LR)
                loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(
                    self.real_HR, self.rec_HR)) * self.conf.lambda_cycle

            # import ipdb; ipdb.set_trace()
            # Losses
            if self.train_strategy[0] == 1:
                loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)

            # sobel_A = Sobel()(self.real_LR_bicubic.detach())
            # loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
            # self.loss_interp = self.criterion_interp(self.fake_HR * loss_map_A, self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp

            # self.curr_k = util.calc_curr_k(self.G_DN.parameters())
            # self.loss_regularization = self.regularization(self.curr_k, self.real_HR, self.fake_LR) * self.conf.lambda_regularization

            # self.total_loss = self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization
            # total_loss = loss_cycle_forward + loss_cycle_backward

            total_loss = loss_cycle_forward + loss_cycle_backward + loss_GAN

            # total_loss = loss_cycle_forward + loss_cycle_backward + self.loss_regularization

            total_loss.backward()

            # self.optimizer_G_UP.step()
            self.optimizer_G_DN.step()

        return {
            "train_G_DN/loss_cycle_forward": loss_cycle_forward,
            "train_G_DN/loss_cycle_backward": loss_cycle_backward,
            "train_G_DN/total_loss": total_loss,
            "train_G_DN/loss_GAN": loss_GAN,
            # "train_G_DN/loss_regularization": self.loss_regularization,

        }

    def train_G_UP(self):
        loss_cycle_forward = 0
        loss_cycle_backward = 0
        total_loss = 0
        loss_interp = 0

        if self.train_G_UP_switch:

            # Turn on gradient calculation for G_UP
            util.set_requires_grad([self.G_UP], True)
            # Turn off gradient calculation for G_DN
            # util.set_requires_grad([self.G_DN], True)
            util.set_requires_grad([self.G_DN], False)
            # Turn off gradient calculation for D_DN
            util.set_requires_grad([self.D_DN], False)

            # Rese gradient valus
            self.optimizer_G_UP.zero_grad()

            # Forward path
            if self.train_strategy[2] == 1:
                self.fake_HR = self.G_UP(self.real_LR)
                self.rec_LR = self.G_DN(self.fake_HR)
                loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(
                    self.real_LR, self.rec_LR)) * self.conf.lambda_cycle

            # Backward path
            if self.train_strategy[4] == 1:
                self.fake_LR = self.G_DN(self.real_HR)
                self.rec_HR = self.G_UP(self.fake_LR)
                loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(
                    self.real_HR, self.rec_HR)) * self.conf.lambda_cycle

            # sobel_A = Sobel()(self.real_LR_bicubic.detach())
            # loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
            # loss_interp = self.criterion_interp(self.fake_HR * loss_map_A, self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp

            # Losses

            total_loss = loss_cycle_forward + loss_cycle_backward

            total_loss.backward()

            # Update weights
            self.optimizer_G_UP.step()

        return {
            "train_G_UP/loss_cycle_forward": loss_cycle_forward,
            "train_G_UP/loss_cycle_backward": loss_cycle_backward,
            "train_G_UP/total_loss": total_loss,
            "train_G_UP/loss_interp": loss_interp,
        }

    def train_D_DN(self):
        if self.train_strategy[0] == 1 and self.train_D_DN_switch:
            # Turn on gradient calculation for discriminator
            util.set_requires_grad([self.D_DN], True)

            # Rese gradient valus
            self.optimizer_D_DN.zero_grad()

            # Fake
            pred_fake = self.D_DN(self.fake_LR.detach())
            loss_D_fake = self.criterion_gan(pred_fake, False)
            # Real
            # import ipdb; ipdb.set_trace()
            pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
            loss_D_real = self.criterion_gan(pred_real, True)
            # Combined loss and calculate gradients
            self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
            self.loss_Discriminator.backward()

            # Update weights
            self.optimizer_D_DN.step()

        else:
            self.loss_Discriminator = 0

        return {
            "train_D_DN/loss_Discriminator": self.loss_Discriminator
        }

    def pst(self, tensor, filename):
        '''
        Plot and save a tensor (PST)
        '''
        image = util.tensor2im(tensor)
        plt.imsave(os.path.join(
            self.conf.visual_dir, f"{filename}.png"), image)
        plt.close()

    def eval(self, iteration, save_result=False):
        self.quick_eval()
        torch.cuda.empty_cache()
        # if self.conf.debug:
        #     self.plot()

        if save_result:
            plt.imsave(os.path.join(self.conf.visual_dir,
                       f"upsampled_img_{self.conf.abs_img_name}_{iteration+1}.png"), self.upsampled_img)
            # if not self.conf.test_only:
            #     plt.imsave(os.path.join(self.conf.visual_dir, f"downsampled_img_{self.conf.abs_img_name}_{iteration+1}.png"), self.downsampled_img)
            plt.close('all')

        # if self.gt_img is not None:
        #     print('Upsampler PSNR = ', self.UP_psnrs[-1])
        # if self.gt_kernel is not None:
        #     print("Downsampler PSNR = ", self.DN_psnrs[-1])
        # print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.visual_dir)
        # plt.close('all')

    def quick_eval(self):
        # Evaluate trained upsampler and downsampler on input data
        with torch.no_grad():

            downsampled_img_t = self.G_DN(self.in_img_cropped_t)
            self.G_UP.eval()

            if self.conf.source_model == "swinir":
                window_size = 8
                _, _, h_old, w_old = self.in_img_t.size()
                in_img_t = self.in_img_t.clone()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                in_img_t = torch.cat([in_img_t, torch.flip(in_img_t, [2])], 2)[
                    :, :, :h_old + h_pad, :]
                in_img_t = torch.cat([in_img_t, torch.flip(in_img_t, [3])], 3)[
                    :, :, :, :w_old + w_pad]

                upsampled_img_t = self.G_UP(in_img_t)
                upsampled_img_t = upsampled_img_t[..., :h_old *
                                                  self.conf.scale_factor, :w_old * self.conf.scale_factor]
                self.downsampled_img = util.tensor2im(downsampled_img_t)
                self.upsampled_img = util.tensor2im(upsampled_img_t)

                if self.gt_img is not None:
                    _, _, h_old, w_old = self.in_img_t.size()
                    self.UP_psnrs = [util.cal_y_psnr(self.upsampled_img, self.gt_img[:h_old * self.conf.scale_factor,
                                                                                     :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]
                    # self.UP_ssims = [util.util_calculate_psnr_ssim(self.upsampled_img, self.gt_img[:h_old * self.conf.scale_factor,
                    #                                                                  :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]


            elif self.conf.source_model == "edsr" or self.conf.source_model == "rcan":
                in_img_t = self.in_img_t
                upsampled_img_t = self.G_UP(in_img_t)
                pixel_range = 255/255
                upsampled_img_t = upsampled_img_t.mul(
                    pixel_range).clamp(0, 255).round().div(pixel_range)

                from tta_util import calc_psnr_edsr
                _, _, h_old, w_old = self.in_img_t.size()
                # gt_img = torch.tensor(self.gt_img.transpose((2, 0, 1))).cuda()
                # upsampled_img_t = upsampled_img_t.squeeze()
                # self.UP_psnrs += [calc_psnr_edsr(upsampled_img_t, gt_img)]
                upsampled_img_t = upsampled_img_t.squeeze()
                upsampled_img_t = upsampled_img_t.cpu().numpy().transpose((1, 2, 0))
                # import ipdb; ipdb.set_trace()
                self.UP_psnrs += [util.cal_y_psnr(upsampled_img_t, self.gt_img[:h_old * self.conf.scale_factor,
                                                                               :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]

                # if self.gt_img is not None:
                #     _, _, h_old, w_old = self.in_img_t.size()
                #     self.UP_psnrs += [util.cal_y_psnr(self.upsampled_img, self.gt_img[:h_old * self.conf.scale_factor,
                #                                     :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]

            elif self.conf.source_model == "cdc":
                psize = 64
                overlap = 256
                gpus = 1
                device = "cuda"

                # import ipdb; ipdb.set_trace()
                blocks = util.tensor_divide(self.in_img_t, psize, overlap)
                blocks = torch.cat(blocks, dim=0)
                results = []

                # // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
                iters = blocks.shape[0]
                for idx in range(iters):
                    if idx + 1 == iters:
                        input = blocks[idx * gpus:]
                    else:
                        input = blocks[idx * gpus: (idx + 1) * gpus]
                    hr_var = input.to(device)
                    sr_var, SR_map = self.G_UP(hr_var, return_all=True)

                    if isinstance(sr_var, list) or isinstance(sr_var, tuple):
                        sr_var = sr_var[-1]

                    results.append(sr_var.to('cpu'))
                    # print('Processing Image: %d Part: %d / %d'
                    #     % (batch + 1, idx + 1, iters), end='\r')
                    # sys.stdout.flush()

                results = torch.cat(results, dim=0)
                # import ipdb; ipdb.set_trace()

                gt_img = torch.tensor(self.gt_img.transpose((2, 0, 1))) / 255.
                gt_img = gt_img.unsqueeze(0)

                sr_img = util.tensor_merge(
                    results, gt_img, psize * self.conf.scale_factor, overlap * self.conf.scale_factor)

                self.UP_psnrs += [util.YCbCr_psnr(sr_img, gt_img,
                                                  scale=self.conf.scale_factor, peak=1.)]

                self.upsampled_img = sr_img

            else:
                # import ipdb; ipdb.set_trace()
                in_img_t = self.in_img_t
                upsampled_img_t = self.G_UP(in_img_t)

                self.downsampled_img = util.tensor2im(downsampled_img_t)
                self.upsampled_img = util.tensor2im(upsampled_img_t)

                if self.gt_img is not None:
                    _, _, h_old, w_old = self.in_img_t.size()
                    self.UP_psnrs += [util.cal_y_psnr(self.upsampled_img, self.gt_img[:h_old * self.conf.scale_factor,
                                                                                      :w_old * self.conf.scale_factor, ...], border=self.conf.scale_factor)]

    def plot(self):
        loss_names = ['loss_GANs', 'loss_cycle_forwards',
                      'loss_cycle_backwards', 'loss_interps', 'loss_Discriminators']

        if self.gt_img is not None:
            plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                                       in zip([self.UP_psnrs, self.DN_psnrs],
                                              ['Upsampler PSNR', 'Downsampler PSNR']) if x is not None])
        else:
            plots_data, labels = [0.0], 'None'

        plots_data2, labels2 = zip(*[(np.array(x), l) for (x, l)
                                   in zip([getattr(self, name) for name in loss_names],
                                          loss_names) if x is not None])
        # For the first iteration create the figure
        if not self.iter:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9, 8))
            #self.fig = plt.figure()
            grid = GridSpec(4, 4)
            self.psnr_plot_space = plt.subplot(grid[0:2, 0:2])
            self.loss_plot_space = plt.subplot(grid[0:2, 2:4])

            self.real_LR_space = plt.subplot(grid[2, 0])
            self.fake_HR_space = plt.subplot(grid[2, 1])
            self.rec_LR_space = plt.subplot(grid[2, 2])
            self.real_HR_space = plt.subplot(grid[3, 0])
            self.fake_LR_space = plt.subplot(grid[3, 1])
            self.rec_HR_space = plt.subplot(grid[3, 2])
            self.curr_ker_space = plt.subplot(grid[2, 3])
            self.ideal_ker_space = plt.subplot(grid[3, 3])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.psnr_plot_space.set_ylabel('db')
            self.psnr_plot_space.grid(True)
            self.psnr_plot_space.legend(labels)

            self.loss_plot_space.grid(True)
            self.loss_plot_space.legend(labels2)

            self.curr_ker_space.title.set_text('estimated kernel')
            self.ideal_ker_space.title.set_text('gt kernel')
            self.real_LR_space.title.set_text('$x$')
            self.real_HR_space.title.set_text('$y$')
            self.fake_HR_space.title.set_text('$G_{UP}(x)$')
            self.fake_LR_space.title.set_text('$G_{DN}(y)$')
            self.rec_LR_space.title.set_text('$G_{DN}(G_{UP}(x))$')
            self.rec_HR_space.title.set_text('$G_{UP}(G_{DN}(y))$')

            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.psnr_plot_space.plot(
                *[[0]] * 2 * len(plots_data))
            self.plots2 = self.loss_plot_space.plot(
                *[[0]] * 2 * len(plots_data2))

            # These line are needed in order to see the graphics at real time
            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.01)
            return

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.debug_steps, plot_data)

        for plot, plot_data in zip(self.plots2, plots_data2):
            plot.set_data(self.debug_steps, plot_data)

        self.psnr_plot_space.set_xlim([0, self.iter + 1])
        all_losses = np.array(plots_data)
        self.psnr_plot_space.set_ylim(
            [np.min(all_losses)*0.9, np.max(all_losses)*1.1])

        self.loss_plot_space.set_xlim([0, self.iter + 1])
        all_losses2 = np.array(plots_data2)
        self.loss_plot_space.set_ylim(
            [np.min(all_losses2)*0.9, np.max(all_losses2)*1.1])

        self.psnr_plot_space.legend(labels)
        self.loss_plot_space.legend(labels2)

        # Show current images
        self.curr_ker_space.imshow(util.move2cpu(self.curr_k))
        if self.gt_kernel is not None:
            self.ideal_ker_space.imshow(self.gt_kernel)
        self.real_LR_space.imshow(util.tensor2im(self.real_LR))
        self.real_HR_space.imshow(util.tensor2im(self.real_HR))
        self.fake_HR_space.imshow(util.tensor2im(self.fake_HR))
        self.fake_LR_space.imshow(util.tensor2im(self.fake_LR))
        self.rec_LR_space.imshow(util.tensor2im(self.rec_LR))
        self.rec_HR_space.imshow(util.tensor2im(self.rec_HR))

        self.curr_ker_space.axis('off')
        self.ideal_ker_space.axis('off')
        self.real_LR_space.axis('off')
        self.real_HR_space.axis('off')
        self.fake_HR_space.axis('off')
        self.fake_LR_space.axis('off')
        self.rec_LR_space.axis('off')
        self.rec_HR_space.axis('off')

        # These line are needed in order to see the graphics at real time
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    def save_model(self, iteration):

        torch.save(self.G_UP.state_dict(), os.path.join(
            self.conf.model_save_dir, f"ckpt_GUP_{iteration+1}.pth"))
        torch.save(self.G_DN.state_dict(), os.path.join(
            self.conf.model_save_dir, f"ckpt_GDN_{iteration+1}.pth"))
        torch.save(self.D_DN.state_dict(), os.path.join(
            self.conf.model_save_dir, f"ckpt_DDN_{iteration+1}.pth"))






def test(img_lq, model, args, window_size):
    args = {
        "tile": 48,
        "tile_overlap": 8,
    }
    if args["tile"] is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args["tile"], h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args["tile_overlap"]
        sf = args["scale"]

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

