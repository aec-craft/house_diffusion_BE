import os

import numpy as np
import torch as th
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
from random import randint

def create_layout(graphs, corners, room_type, metrics=False):
    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion()
    model.load_state_dict(
        dist_util.load_state_dict("scripts/model.pt", map_location="cpu")
    )
    model.to("cpu") #model.to(dist_util.dev())
    model.eval()
    ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                13: '#785A67', 12: '#D3A2C7'}
    num_room_types = 14
    sample_fn = (diffusion.p_sample_loop if not False else diffusion.ddim_sample_loop)
    model_kwargs = function_test(graphs, corners, room_type)
    for key in model_kwargs:
        model_kwargs[key] = th.from_numpy(np.array([model_kwargs[key]]))#.cuda()

    data_uri = []
    for count in range(1):
        sample = sample_fn(
            model,
            th.Size([1, 2, 100]),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            analog_bit=False,
        )

        sample = sample.permute([0, 1, 3, 2])

        pred = save_samples(sample, 'test', model_kwargs, count, num_room_types, ID_COLOR=ID_COLOR,
                            is_syn=True, draw_graph=False, save_svg=True, metrics=metrics)

        data_uri.append(pred)

    return data_uri
