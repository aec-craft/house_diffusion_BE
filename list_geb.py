import os
import json

total_files = os.listdir('/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan')

with open('list.txt', 'w') as f:
    for line in total_files:
        f.write(f"{line}")
        f.write('\n')

# json_files = os.listdir('/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan')
# for x in json_files:
#     with open(f'/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan/{x}') as f:
#         try:
#             info = json.load(f)
#         except json.decoder.JSONDecodeError:
#             print(f'deleted /home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan/{x}')
#             os.remove(f'/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan/{x}')
