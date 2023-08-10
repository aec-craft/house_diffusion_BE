import json
import os
import csv
import codecs
import time


import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from scripts.test import create_layout


class Nodes(BaseModel):
    id: str
    room_type: str
    corners: Optional[str] | None


class Edges(BaseModel):
    id: str
    source: str
    target: str


class HouseGraph(BaseModel):
    nodes: List[Nodes]
    edges: List[Edges]


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOM_CLASS = {'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4, 'Balcony': 5, 'Entrance': 6, 'Dining Room': 7,
              'Study Room': 8, 'Storage': 10, 'Front Door': 11, 'Unknown': 13, 'Interior Door': 12}


@app.post("/generate")
async def generate(house_graph: HouseGraph):
    room_list = []
    room_corners = []
    living_room = 0
    entrance = False
    for i, room in enumerate(house_graph.nodes):
        room_list.append(ROOM_CLASS[room.room_type])
        if room.corners != "0":
            room_corners.append(int(room.corners))
        else:
            if room.room_type == "Living Room":
                living_room = i
                if len(house_graph.nodes) > 3:
                    room_corners.append(13)
                else:
                    room_corners.append(4)
            else:
                room_corners.append(4)

        if room.room_type == "Front Door":
            entrance = True

    edges = []
    for edge in house_graph.edges:
        edges.append([int(edge.source), 1, int(edge.target)])
        index = len(room_list)
        room_list.append(12)
        room_corners.append(4)
        edges.append([int(edge.source), 1, index])
        edges.append([int(edge.target), 1, index])

    if not entrance:
        room_list.append(11)
        room_corners.append(4)
        edges.append([len(room_list) - 1, 1, living_room])

    for x in range(len(room_list)):
        for y in range(len(room_list)):
            if y > x:
                if any(np.equal([x, 1, y], edges).all(1)) or any(np.equal([y, 1, x], edges).all(1)):
                    continue
                else:
                    edges.append([x, -1, y])

    if np.sum(room_corners) > 99:
        return {"Error": "Number of Corners exceeded"}
    data_uri = create_layout(edges, room_corners, room_list)
    return {"dataUri": data_uri}

