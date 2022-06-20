import glob
import json
import os
import shutil
from datetime import datetime

import ipdb
import numpy as np
import setGPU
import torch
import tqdm
import wandb

from tta_data import create_dataset
# from tta_sr_one_cycle import TTASR
from tta_learner import Learner
from tta_options import options
# from DualSR import DualSR
from tta_sr import TTASR


def train_and_eval(conf):
    model = TTASR(conf)
    dataloader = create_dataset(conf)
    learner = Learner(model)

    # generate dataset first
    data_dir = f"generating_data/data_{conf.abs_img_name}_{conf.input_crop_size}_{conf.num_iters}.pth"
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    data_collection = []
    if not os.path.exists(data_dir):
        for iteration, data in enumerate(tqdm.tqdm(dataloader)):
            data_collection.append(data)

        torch.save(data_collection, data_dir)
    else:
        data_collection = torch.load(data_dir)

    print('*' * 60 + '\nTraining started ...')
    # for iteration, data in enumerate(tqdm.tqdm(dataloader)):
    best_res = {
        "iteration": 0,
        "PSNR": 0,
    }
    psnr_record = []
    
    for iteration, data in enumerate(tqdm.tqdm(data_collection)):
        if iteration == 0:
            model.train_G_DN_switch = True
            model.train_G_UP_switch = False
        loss = model.train(data)
        learner.update(iteration, model)

        loss["iteration"] = iteration

        if (iteration+1) % conf.model_save_iter == 0:
            model.save_model(iteration+1)

        if (iteration+1) % conf.eval_iters == 0 and model.train_G_UP_switch:
            model.eval(iteration)
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(
                    f"IMG_IDX: {conf.img_idx}. iteration: {iteration}. {conf.abs_img_name}. PSNR: {model_img_specific.UP_psnrs[-1]} \n")

            loss["eval/psnr"] = model.UP_psnrs[-1]

            if model.UP_psnrs[-1] > best_res["PSNR"]:
                best_res["PSNR"] = model.UP_psnrs[-1]
                best_res["iteration"] = iteration
            pass

            psnr_record.append([iteration, model.UP_psnrs[-1]])

        if (iteration+1) % conf.eval_iters == 0:
            loss_log = {}
            for key, val in loss.items():
                key = f"{conf.abs_img_name}/{key}"
                loss_log[key] = val

            loss_log["iteration"] = iteration
            wandb.log(loss_log)

    torch.save(psnr_record, os.path.join(
        conf.model_save_dir, f"{conf.abs_img_name}_psnr.pt"))
    print("Best PSNR: {}, at iteration: {}".format(
        best_res["PSNR"], best_res["iteration"]))
    wandb.run.summary[f"best_psnr_{conf.abs_img_name}"] = best_res["PSNR"]

    model.eval(0)
    return model.UP_psnrs[-1]


def main():
    torch.set_num_threads(5)

    opt = options()

    #############################################################################################
    #############################################################################################
    print("Start file saving...")
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # ipdb.set_trace()
    experimentdir = f"./log/{opt.conf.output_dir}/time_{time_stamp}"
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
    # Testing
    if opt.conf.test_only:
        model = TTASR(opt.conf)
        
        all_psnr = []
        img_list = []
        for img_idx, img_name in enumerate(os.listdir(opt.conf.input_dir)):
        # for img_name in os.listdir(opt.conf.input_dir):
            conf = opt.get_config(img_name)
            conf.img_idx = img_idx
            
            model.read_image(conf)
            model.eval(0)
            with open(os.path.join(conf.experimentdir, "psnr.txt"), "a") as f:
                f.write(
                    f"IMG: {img_idx}. Iteration: {0}. {conf.abs_img_name}. PSNR: {model.UP_psnrs[-1]}\n")
            all_psnr.append(model.UP_psnrs[-1])
            all_psnr.append(model.UP_psnrs[-1])
            img_list.append(img_name)
            
        all_psnr = np.array(all_psnr)
        with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
            f.write(f"Input directory: {opt.conf.input_dir}.\n")

            for img, psnr in zip(img_list, all_psnr):
                f.write(f"IMG: {img}, psnr: {psnr} .\n")

            f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
        print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")
        
        return

    # Training
    else:
        # wandb logger
        wandb.init(
            project="TTA_SR",
            entity="kaistssl",
            name=opt.conf.output_dir,
            config=opt.conf,
            dir=opt.conf.experimentdir,
            save_code=True,
        )

        # Run DualSR on all images in the input directory
        # for img_name in os.listdir(opt.conf.input_dir):
        img_list = []
        all_psnr = []
        for img_idx, img_name in enumerate(os.listdir(opt.conf.input_dir)):
            
            conf = opt.get_config(img_name)
            conf.img_idx = img_idx

            psnr = train_and_eval(conf)
            all_psnr.append(psnr)
            img_list.append(img_name)

    all_psnr = np.array(all_psnr)
    with open(os.path.join(conf.experimentdir, "final_psnr.txt"), "a") as f:
        f.write(f"Input directory: {opt.conf.input_dir}.\n")

        for img, psnr in zip(img_list, all_psnr):
            f.write(f"IMG: {img}, psnr: {psnr} .\n")

        f.write(f"Average PSNR: {np.mean(all_psnr)}.\n")
    print(f"Average PSNR for {opt.conf.input_dir}: {np.mean(all_psnr)}")


if __name__ == '__main__':
    main()
