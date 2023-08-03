from collections import defaultdict
from scripts.creation_utils import build_graph
import numpy as np
import cv2 as cv
import random
# from house_diffusion.rplanhg_datasets import reader


def function_test(info, corner_dict=None):
    rms_type, fp_eds, rms_bbs, eds_to_rms = reader(info)
    fp_size = len([x for x in rms_type if x != 15 and x != 17])
    graph = [rms_type, rms_bbs, fp_eds, eds_to_rms]
    rms_type = graph[0]
    rms_bbs = graph[1]
    fp_eds = graph[2]
    eds_to_rms = graph[3]
    rms_bbs = np.array(rms_bbs)
    fp_eds = np.array(fp_eds)

    # extract boundary box and centralize
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    tl -= shift
    br -= shift

    graph_nodes, graph_edges, rooms_mks = build_graph(rms_type, fp_eds, eds_to_rms)
    house = []
    for room_mask, room_type in zip(rooms_mks, graph_nodes):
        room_mask = room_mask.astype(np.uint8)
        room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
        contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        house.append([contours[:, 0, :], room_type])
    org_graphs = graph_edges
    org_houses = house

    cnumber_dist = defaultdict(list)

    get_one_hot = lambda x, z: np.eye(z)[x]
    max_num_points = 100

    house = []
    corner_bounds = []
    num_points = 0

    for i, room in enumerate(org_houses):
        if room[1] > 10:
            room[1] = {15: 11, 17: 12, 16: 13}[room[1]]
        room[0] = np.reshape(room[0], [len(room[0]),
                                       2]) / 256. - 0.5  # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
        room[0] = room[0] * 2  # map to [-1, 1]
        cnumber_dist[room[1]].append(len(room[0]))
        # Adding conditions
        num_room_corners = len(room[0])
        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
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
        room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
        house.append(room)

    house_layouts = np.concatenate(house, 0)
    if len(house_layouts) > max_num_points:
        print('error')
    padding = np.zeros((max_num_points - len(house_layouts), 94))
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
    house_layouts = np.concatenate((house_layouts, padding), 0)

    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))
    for i in range(len(corner_bounds)):
        for j in range(len(corner_bounds)):
            if i == j:
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],
                corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], org_graphs).all(1)) or any(np.equal([j, 1, i], org_graphs).all(1)):
                door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                corner_bounds[j][0]:corner_bounds[j][1]] = 0
    houses = house_layouts
    door_masks = door_mask
    self_masks = self_mask
    gen_masks = gen_mask
    graphs = graph

    house = []
    corner_bounds = []
    num_points = 0
    print(f"Corner Number{cnumber_dist}")
    if corner_dict is not None:
        cnumber_dist.update(corner_dict)

    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]]) - 1)]
                                              for room in org_houses]
    while np.sum(num_room_corners_total) >= max_num_points:
        num_room_corners_total = [
            cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]]) - 1)] for room in org_houses]

    print(f"Num Room {num_room_corners_total}")
    for i, room in enumerate(org_houses):
        # Adding conditions
        num_room_corners = num_room_corners_total[i]
        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
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
    if np.sum([len(room[0]) for room in org_houses]) > max_num_points:
        print('error')
    padding = np.zeros((max_num_points - len(house_layouts), 94))
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
    house_layouts = np.concatenate((house_layouts, padding), 0)

    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))
    for i, room in enumerate(org_houses):
        if room[1] == 1:
            living_room_index = i
            break
    for i in range(len(corner_bounds)):
        is_connected = False
        for j in range(len(corner_bounds)):
            if i == j:
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],
                corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], org_graphs).all(1)) or any(np.equal([j, 1, i], org_graphs).all(1)):
                door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                corner_bounds[j][0]:corner_bounds[j][1]] = 0
                is_connected = True
        if not is_connected:
            door_mask[corner_bounds[i][0]:corner_bounds[i][1],
            corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

    syn_houses = house_layouts
    syn_door_masks = door_mask
    syn_self_masks = self_mask
    syn_gen_masks = gen_mask

    arr = houses[:, :2]

    graph = np.concatenate((org_graphs, np.zeros([200 - len(org_graphs), 3])), 0)
    cond = {
        'door_mask': door_masks,
        'self_mask': self_masks,
        'gen_mask': gen_masks,
        'room_types': houses[:, 2:2 + 25],
        'corner_indices': houses[:, 2 + 25:2 + 57],
        'room_indices': houses[:, 2 + 57:2 + 89],
        'src_key_padding_mask': 1 - houses[:, 2 + 89],
        'connections': houses[:, 2 + 90:2 + 92],
        'graph': graph,
    }

    syn_graph = np.concatenate((org_graphs, np.zeros([200 - len(org_graphs), 3])), 0)

    cond.update({
        'syn_door_mask': syn_door_masks,
        'syn_self_mask': syn_self_masks,
        'syn_gen_mask': syn_gen_masks,
        'syn_room_types': syn_houses[:, 2:2 + 25],
        'syn_corner_indices': syn_houses[:, 2 + 25:2 + 57],
        'syn_room_indices': syn_houses[:, 2 + 57:2 + 89],
        'syn_src_key_padding_mask': 1 - syn_houses[:, 2 + 89],
        'syn_connections': syn_houses[:, 2 + 90:2 + 92],
        'syn_graph': syn_graph,
    })

    arr = np.transpose(arr, [1, 0])
    return arr.astype(float), cond


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
