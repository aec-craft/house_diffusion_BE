import numpy as np
import torch as th

import io
from PIL import Image, ImageDraw
import imageio


def make_sequence(edges):
    polys = []
    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False
    while len(e_visited) < len(edges):
        if find_next == False:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = not find_next
        else:
            # look for next edge
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if (v_curr == tuple(e[:2])):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif (v_curr == tuple(e[2:])):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break

        # extract next sequence
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(edges[0][:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    polys.append(seq_tracker)

    return polys


def build_graph(rms_type, fp_eds, eds_to_rms, out_size=64):
    # create edges
    triples = []
    nodes = rms_type
    # encode connections
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                if is_adjacent:
                    triples.append([k, 1, l])
                else:
                    triples.append([k, -1, l])

    # get rooms masks
    eds_to_rms_tmp = []
    for l in range(len(eds_to_rms)):
        eds_to_rms_tmp.append([eds_to_rms[l][0]])
    rms_masks = []
    im_size = 256
    fp_mk = np.zeros((out_size, out_size))
    for k in range(len(nodes)):
        # add rooms and doors
        eds = []
        for l, e_map in enumerate(eds_to_rms_tmp):
            if (k in e_map):
                eds.append(l)
        # draw rooms
        rm_im = Image.new('L', (im_size, im_size))
        dr = ImageDraw.Draw(rm_im)
        for eds_poly in [eds]:
            poly = make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
            poly = [(im_size * x, im_size * y) for x, y in poly]
            if len(poly) >= 2:
                dr.polygon(poly, fill='white')
            else:
                print("Empty room")
                exit(0)
        rm_im = rm_im.resize((out_size, out_size))
        rm_arr = np.array(rm_im)
        inds = np.where(rm_arr > 0)
        rm_arr[inds] = 1.0
        rms_masks.append(rm_arr)
        if rms_type[k] != 15 and rms_type[k] != 17:
            fp_mk[inds] = k + 1
    # trick to remove overlap
    for k in range(len(nodes)):
        if rms_type[k] != 15 and rms_type[k] != 17:
            rm_arr = np.zeros((out_size, out_size))
            inds = np.where(fp_mk == k + 1)
            rm_arr[inds] = 1.0
            rms_masks[k] = rm_arr
    # convert to array
    nodes = np.array(nodes)
    triples = np.array(triples)
    rms_masks = np.array(rms_masks)
    return nodes, triples, rms_masks
