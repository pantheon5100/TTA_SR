from argparse import Namespace

import torch

from tta_model.network_swinir import define_model
from tta_model.rcan import RCAN
from tta_model.edsr import EDSR
from tta_model.cdc import HourGlassNetMultiScaleInt


def get_model(conf):
    if conf.source_model == "swinir":
        G_UP_model_conf = {
            "task": conf.swinir_task,
            "scale": conf.scale_factor,
            "model_type": f"{conf.swinir_task}_{conf.scale_factor}",
            # "training_patch_size": conf.input_crop_size,
            "training_patch_size": 48,
            "large_model": False
            }
        G_UP = define_model(**G_UP_model_conf).cuda()
        
        return G_UP

    elif conf.source_model == "rcan":
        rcan_config = Namespace()
        rcan_config.n_resgroups = 10
        rcan_config.n_resblocks = 20
        rcan_config.n_feats = 64
        rcan_config.scale = [2]
        rcan_config.data_train = "DIV2K"
        rcan_config.rgb_range = 255
        rcan_config.n_colors = 3
        rcan_config.res_scale = 1
        rcan_config.reduction = 16

        G_UP = RCAN(rcan_config)

        state_dict = torch.load("tta_pretrained/models_ECCV2018RCAN/RCAN_BIX2.pt")
        G_UP.load_state_dict(state_dict=state_dict)
        G_UP.cuda()

        return G_UP
    
    elif conf.source_model == "edsr":
        edsr_config = Namespace()
        edsr_config.n_resblocks = 32
        edsr_config.n_feats = 256
        edsr_config.scale = [2]
        edsr_config.rgb_range = 255
        edsr_config.n_colors = 3
        edsr_config.res_scale = 0.1

        G_UP = EDSR(edsr_config)
        # import ipdb; ipdb.set_trace();
        state_dict = torch.load("tta_pretrained/EDSR_x2.pt") 
        G_UP.load_state_dict(state_dict=state_dict)
        G_UP.cuda()

        return G_UP
    
    elif conf.source_model == "rrdb":
        edsr_config = Namespace()
        edsr_config.n_resblocks = 32
        edsr_config.n_feats = 256
        edsr_config.scale = [2]
        edsr_config.rgb_range = 255
        edsr_config.n_colors = 3
        edsr_config.res_scale = 0.1

        G_UP = EDSR(edsr_config)
        # import ipdb; ipdb.set_trace();
        state_dict = torch.load("tta_pretrained/EDSR_x2.pt") 
        G_UP.load_state_dict(state_dict=state_dict)
        G_UP.cuda()

        return G_UP



    elif conf.source_model == "cdc":
        # cdc_config = Namespace()
        # cdc_config.in_ch = 3
        # cdc_config.out_ch = 3
        # cdc_config.n_HG = 6
        # cdc_config.model = 'HGSR-MHR'
                               
        G_UP = HourGlassNetMultiScaleInt(in_nc=3, out_nc=3, upscale=2,
                                   nf=64, res_type='res', n_mid=2,
                                   n_HG=6, inter_supervis=True)
        # G_UP = HourGlassNet(in_nc=3, out_nc=3, upscale=2) 
        # import ipdb; ipdb.set_trace();
        state_dict = torch.load("./tta_pretrained/CDC_X2_Model/HGSR-MHR_X2_CDC.pth") 
        # G_UP.load_state_dict(state_dict=state_dict, strict=False)
        G_UP.load_state_dict(state_dict=state_dict["state_dict"])
        
        G_UP.cuda()
        return G_UP



        

