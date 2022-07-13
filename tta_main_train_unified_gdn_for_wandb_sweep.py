import copy
import glob
import json
import os
import shutil
from datetime import datetime

import numpy as np
# import setGPU
import torch
import torch.nn.functional as F
import tqdm
import wandb

from tta_data import create_dataset, create_dataset_for_image_agnostic_gdn
import tta_util as util
from tta_learner import Learner
from tta_options import options
from tta_sr_loss_ablation import TTASR



def main():
    torch.set_num_threads(2)

    opt = options()
    opt.conf.num_iters = opt.conf.gdn_iters + opt.conf.gup_iters
    opt.conf.switch_iters = 1 if opt.conf.gdn_iters == 0 else opt.conf.gdn_iters
    opt.conf.lr_G_DN_step_size = int(opt.conf.gdn_iters/4)

    #############################################################################################
    #############################################################################################
    print("Start file saving...")
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # ipdb.set_trace()
    data_set_name = opt.conf.gt_dir.split("/")[-2]
    run_name = f"{opt.conf.output_dir}-{opt.conf.source_model}-{data_set_name}-{opt.conf.training_strategy}"
    if opt.conf.pretrained_gdn != "":
        run_name += "-use_pretrained_gdn"
    experimentdir = f"./log/{run_name}/time_{time_stamp}"
    experimentdir += f"lr_GUP_{opt.conf.lr_G_UP}-lr_GDN_{opt.conf.lr_G_DN}input_size_{opt.conf.input_crop_size}-scale_factor_{opt.conf.scale_factor}"
    opt.conf.experimentdir = experimentdir
    code_saving_dir = os.path.join(experimentdir, "code")
    os.makedirs(code_saving_dir)

    shutil.copytree(f"./tta_model", os.path.join(code_saving_dir, 'tta_model'))
    shutil.copytree(f"./bash_files",
                    os.path.join(code_saving_dir, 'bash_files'))

    # search main dir .py file to save
    pathname = "*.py"
    files = glob.glob(pathname, recursive=True)
    for file in files:
        dest_fpath = os.path.join(code_saving_dir, os.path.basename(file))
        shutil.copy(file, dest_fpath)

    opt.conf.visual_dir = os.path.join(experimentdir, "visual")
    os.makedirs(opt.conf.visual_dir)
    opt.conf.model_save_dir = os.path.join(experimentdir, "ckpt")
    os.makedirs(opt.conf.model_save_dir)

    # save running argument
    with open(os.path.join(experimentdir, 'commandline_args.txt'), 'w') as f:
        json.dump(opt.conf.__dict__, f, indent=2)
    #############################################################################################
    #############################################################################################

    all_psnr = []
    all_ssim = []
    if True:
        # wandb logger
        wandb.init(
            project=f"TTA_SR-reproduce",
            entity="kaistssl",
            config=opt.conf,
            dir=opt.conf.experimentdir,
            save_code=True,
        )
        opt.conf = wandb.config
        
        unified_gdn = True
        if unified_gdn:
            opt.conf.update({"lr_G_DN_step_size": int(opt.conf.pretrained_gdn_num_iters/4)}, allow_val_change=True)

            # if opt.conf.pretrained_gdn_with_imgenet:
            #     opt.conf.input_dir = '/workspace/ssd1_2tb/nax_projects/super_resolution/dataset/imagenet_selected'

            model = TTASR(opt.conf)
            # use all image to train, every batch contains all image
            from tta_data import create_dataset_for_image_agnostic_gdn
            dataloader = create_dataset_for_image_agnostic_gdn(opt.conf)
            learner = Learner(model)


            # freeze GUP
            model.train_G_DN_switch = True
            model.train_G_UP_switch = False
            model.train_D_DN_switch = True
            model.reshap_train_data = True

            # for unified_gdn, we set gup freeze and gdn train
            util.set_requires_grad([model.G_UP], False)
            # Turn on gradient calculation for G_DN
            util.set_requires_grad([model.G_DN], True)

            # train GDN
            print("Train unified GDN starts ...")
            for iteration, data in enumerate(tqdm.tqdm(dataloader)):

                loss = model.train(data)

                if (iteration+1) % opt.conf.eval_iters == 0:
                    loss_log = {}
                    for key, val in loss.items():
                        key = f"train_GDN/{key}"
                        loss_log[key] = val

                    loss_log["train_GDN/iteration"] = iteration
                    wandb.log(loss_log)

                learner.update(iteration, model)

            # save the pretrained GDN
            torch.save(model.G_DN.state_dict(), os.path.join(
                opt.conf.model_save_dir, "pretrained_GDN.ckpt"))

            pretrained_GDN_state_dict = model.G_DN.state_dict()



        # Run DualSR on all images in the input directory
        if opt.conf.gdn_iters != 0:
            opt.conf.update({"lr_G_DN_step_size": int(opt.conf.gdn_iters/4)}, allow_val_change=True)
        else:
            opt.conf.update({"lr_G_DN_step_size": 1}, allow_val_change=True)

        # for img_name in os.listdir(opt.conf.input_dir):
        img_list = []
        all_psnr = []
        all_ssim = []

        for img_idx, img_name in enumerate(os.listdir(opt.conf.input_dir)):


            conf = opt.get_config(img_name, wandb_config=True)
            conf.update({"img_idx": img_idx}, allow_val_change=True)



            model = TTASR(conf)
            if unified_gdn:
                model.G_DN.load_state_dict(pretrained_GDN_state_dict)
                
            model.read_image(model.conf)
            # validation check
            model.eval(0)
            
            dataloader = create_dataset(conf)
            learner = Learner(model)

            print('*' * 60 + '\nTraining started ...')
            
            best_res = {
                "iteration": 0,
                "PSNR": 0,
            }
            psnr_record = []
            ssim_record = []


            model.train_G_DN_switch = True
            model.train_G_UP_switch = False
            model.train_D_DN_switch = True

            for iteration, data in enumerate(tqdm.tqdm(dataloader)):

                if iteration == model.conf.gdn_iters:
                    model.train_G_UP_switch = not model.train_G_UP_switch
                    model.train_G_DN_switch = not model.train_G_DN_switch
                    model.train_D_DN_switch = False

                loss = model.train(data)
                learner.update(iteration, model)
                
                # reset ddn
                # if (iteration+1) % 1000 == 0:
                #     model.reset_ddn()

                # if ((iteration+1) % conf.model_save_iter == 0) or ((iteration+1) % model.conf.switch_iters == 0):
                #     model.save_model(iteration+1)

                if (iteration+1) % conf.eval_iters == 0 and model.train_G_UP_switch:
                    model.eval(iteration)
                    with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                        f.write(
                            f"IMG_IDX: {conf.img_idx}. iteration: {iteration}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}. SSIM: {model.UP_ssims[-1]} \n")

                    loss["eval/psnr"] = model.UP_psnrs[-1]
                    loss["eval/ssim"] = model.UP_ssims[-1]

                    if model.UP_psnrs[-1] > best_res["PSNR"]:
                        best_res["PSNR"] = model.UP_psnrs[-1]
                        best_res["iteration"] = iteration
                    pass

                    psnr_record.append([iteration, model.UP_psnrs[-1]])
                    ssim_record.append([iteration, model.UP_ssims[-1]])


                if (iteration+1) % conf.eval_iters == 0:
                    loss_log = {}
                    for key, val in loss.items():
                        key = f"{conf.abs_img_name}/{key}"
                        loss_log[key] = val

                    loss_log["iteration"] = iteration
                    wandb.log(loss_log)

            torch.save(psnr_record, os.path.join(
                conf.model_save_dir, f"{conf.abs_img_name}_psnr.pt"))
            torch.save(ssim_record, os.path.join(
                conf.model_save_dir, f"{conf.abs_img_name}_ssim.pt"))


            print("Best PSNR: {}, at iteration: {}".format(
                best_res["PSNR"], best_res["iteration"]))
            wandb.run.summary[f"best_psnr_{conf.abs_img_name}"] = best_res["PSNR"]

            model.eval(0, save_result=True, )
            psnr = model.UP_psnrs[-1]
            ssim = model.UP_ssims[-1]


            
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            img_list.append(img_name)

            # model.save_model(iteration+1)
            
            with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
                f.write(f"IMG: {img_name}, psnr: {psnr}, ssim: {ssim} .\n")
            


    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)

    with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
        f.write(f"Input directory: {opt.conf.input_dir}.\n")

        # for img, psnr in zip(img_list, all_psnr):
        #     f.write(f"IMG: {img}, psnr: {psnr} .\n")

        f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
        f.write(f"Average SSIM: {np.mean(all_ssim)}.\n")
    print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")
    print(f"Average SSIM for {opt.conf.input_dir}: {np.mean(all_ssim)}")

    wandb.run.summary[f"average_psnr"] = np.mean(all_psnr)
    wandb.run.summary[f"average_ssim"] = np.mean(all_ssim)


if __name__ == '__main__':
    main()
