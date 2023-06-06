import argparse
import os

import numpy as np
import torch as th
from utils.transfer_db import MongoDataset
from scripts.image_sample import bin_to_int_sample, save_samples
from scripts.handle_json import function_test
from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)


def create_argparser():
    defaults = dict(
        dataset='rplan',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/home/akmal/APIIT/FYP Code/house_diffusion/ckpts/exp/model250000.pt",
        draw_graph=False,
        save_svg=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def create_layout(filename):
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                13: '#785A67', 12: '#D3A2C7'}
    num_room_types = 14
    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    data_sample, model_kwargs = function_test(filename)

    data_sample = th.from_numpy(np.array([data_sample]))
    for key in model_kwargs:
        model_kwargs[key] = th.from_numpy(np.array([model_kwargs[key]])).cuda()

    sample = sample_fn(
        model,
        data_sample.shape,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        analog_bit=args.analog_bit,
    )

    sample_gt = data_sample.cuda().unsqueeze(0)
    sample = sample.permute([0, 1, 3, 2])
    sample_gt = sample_gt.permute([0, 1, 3, 2])
    if args.analog_bit:
        sample_gt = bin_to_int_sample(sample_gt)
        sample = bin_to_int_sample(sample)

    gt = save_samples(sample_gt, 'test', model_kwargs, 14, num_room_types, ID_COLOR=ID_COLOR,
                      draw_graph=args.draw_graph, save_svg=args.save_svg)
    pred = save_samples(sample, 'test', model_kwargs, 14, num_room_types, ID_COLOR=ID_COLOR,
                        is_syn=True, draw_graph=args.draw_graph, save_svg=args.save_svg)


ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 2}

mongo_dataset = MongoDataset.objects(**ROOM_CLASS)
create_layout('/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan/16004.json')
