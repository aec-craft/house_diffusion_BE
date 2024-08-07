import json
import os
import csv
import codecs
import time
import uvicorn

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from scripts.test import create_layout
import os.path

import gdown


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
    metrics: bool


app = FastAPI()

origins = [
    "http://localhost:3000",  # Your frontend's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOM_CLASS = {
    'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4, 'Balcony': 5,
    'Entrance': 6, 'Dining Room': 7, 'Study Room': 8, 'Storage': 10, 'Front Door': 11,
    'Unknown': 13, 'Interior Door': 12
}


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

    if np.sum(room_corners) > 99:
        return {"Error": "Number of Corners exceeded"}
    data_uri = create_layout(edges, room_corners, room_list, house_graph.metrics)
    return {"dataUri": data_uri}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
