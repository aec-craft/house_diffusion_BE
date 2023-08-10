from collections import defaultdict
from scripts.creation_utils import build_graph
import numpy as np
import cv2 as cv
import random
# from house_diffusion.rplanhg_datasets import reader


def function_test(org_graphs, corners, room_type):

    get_one_hot = lambda x, z: np.eye(z)[x]
    max_num_points = 100

    house = []
    corner_bounds = []
    num_points = 0

    for i, room in enumerate(room_type):
        # Adding conditions
        num_room_corners = corners[i]
        rtype = np.repeat(np.array([get_one_hot(room, 25)]), num_room_corners, 0)
        room_index = np.repeat(np.array([get_one_hot(len(house) + 1, 32)]), num_room_corners, 0)
        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
        # Src_key_padding_mask
        padding_mask = np.repeat(1, num_room_corners)
        padding_mask = np.expand_dims(padding_mask, 1)
        # Generating corner bounds for attention masks
        connections = np.array([[i, (i + 1) % num_room_corners] for i in range(num_room_corners)])
        connections += num_points
        corner_bounds.append([num_points, num_points + num_room_corners])
        num_points += num_room_corners
        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index,
                               padding_mask, connections), 1)
        house.append(room)

    house_layouts = np.concatenate(house, 0)
    padding = np.zeros((max_num_points - len(house_layouts), 94))
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
    house_layouts = np.concatenate((house_layouts, padding), 0)

    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))
    for i, room in enumerate(room_type):
        if room == 1:
            living_room_index = i
            break
    for i in range(len(corner_bounds)):
        is_connected = False
        for j in range(len(corner_bounds)):
            if i == j:
                self_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], org_graphs).all(1)) or any(np.equal([j, 1, i], org_graphs).all(1)):
                door_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0
                is_connected = True
        if not is_connected:
            door_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

    syn_houses = house_layouts
    syn_door_masks = door_mask
    syn_self_masks = self_mask
    syn_gen_masks = gen_mask

    syn_graph = np.concatenate((org_graphs, np.zeros([200 - len(org_graphs), 3])), 0)

    cond = {
        'syn_door_mask': syn_door_masks,
        'syn_self_mask': syn_self_masks,
        'syn_gen_mask': syn_gen_masks,
        'syn_room_types': syn_houses[:, 2:2 + 25],
        'syn_corner_indices': syn_houses[:, 2 + 25:2 + 57],
        'syn_room_indices': syn_houses[:, 2 + 57:2 + 89],
        'syn_src_key_padding_mask': 1 - syn_houses[:, 2 + 89],
        'syn_connections': syn_houses[:, 2 + 90:2 + 92],
        'syn_graph': syn_graph,
    }

    return cond


def reader(info):
    rms_bbs = np.asarray(info['boxes'])
    fp_eds = info['edges']
    rms_type = info['room_type']
    eds_to_rms = info['ed_rm']
    s_r = 0
    for rmk in range(len(rms_type)):
        if (rms_type[rmk] != 17):
            s_r = s_r + 1
    rms_bbs = np.array(rms_bbs) / 256.0
    fp_eds = np.array(fp_eds) / 256.0
    fp_eds = fp_eds[:, :4]
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    tl -= shift
    br -= shift
    return rms_type, fp_eds, rms_bbs, eds_to_rms
