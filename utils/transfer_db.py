from utils.mongo import create_db
from mongoengine import Document, StringField, DateTimeField, BooleanField, ListField, IntField, FloatField
import json
from tqdm import tqdm

create_db()
base_dir = '/home/akmal/APIIT/FYP Code/Housegan-data-reader/sample_output'


class MongoDataset(Document):
    meta = {'collection': 'dataset'}
    living_room = IntField(required=False)
    kitchen = IntField(required=False)
    bedroom = IntField(required=False)
    bathroom = IntField(required=False)
    balcony = IntField(required=False)
    dining_room = IntField(required=False)
    study_room = IntField(required=False)
    storage = IntField(required=False)
    room_type = ListField(IntField(), default=[])
    boxes = ListField(ListField(FloatField()), default=[])
    edges = ListField(ListField(FloatField()), default=[])
    ed_rm = ListField(ListField(FloatField()), default=[])


with open('/home/akmal/APIIT/FYP Code/house_diffusion/list.txt') as f:
    lines = f.readlines()

for line in tqdm(lines):
    file_name = f'{base_dir}/{line[:-1]}'
    with open(file_name) as f:
        info = json.load(f)
        ROOM_CLASS = {"living_room": info['room_type'].count(1), "kitchen": info['room_type'].count(2), "bedroom": info['room_type'].count(3), "bathroom": info['room_type'].count(4), "balcony": info['room_type'].count(5),
                      "dining_room": info['room_type'].count(7), "study_room": info['room_type'].count(8), "storage": info['room_type'].count(10)}
        info.update(ROOM_CLASS)
        mongo_dataset = MongoDataset(**info)
        mongo_dataset.save()
