# HouseDiffusion API
**[HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)**
<img src='figs/teaser.png' width=100%>
## Installation
**1. Clone repo and install the requirements:**

Implementation is based on the public implementation of [HouseDiffusion](https://github.com/aminshabani/house_diffusion.git). 

```
git clone https://github.com/sakmalh/house_diffusion
cd house_diffusion
pip install -r requirements.txt
```
Add the model from the sharepoint or google drive into the scripts folder and name it as model.pt.

**2. Initiate the Backend**

```
uvicorn app:app --host 0.0.0.0 --port 8000

```

## Endpoint

**1. Request and Response Types**


- To test the endpoint you could simply access http://0.0.0.0:8080/generate

```
{
  "nodes": [
    {
      "id": "string", # Nodes unique id   
      "room_type": "string", # Room Type in ['Living Room', 'Kitchen', 'Bedroom', 'Bathroom', 'Balcony', 'Entrance', 'Dining Room', 'Study Room', 'Storage', 'Front Door', 'Unknown', 'Interior Door']
      "corners": "string" # Number of Corners
    }
  ],
  "edges": [
    {
      "id": "string", # Edge Unique id
      "source": "string", # First Nodes id
      "target": "string"  # Second Nodes id
    }
  ],
  "metrics": true # If True provides the length and width of rooms in pixels
}
```

## Deployment

- DockerFile is provided for hosting. 

```
docker build -t {docker-repo}/backend .
```
## Citation

```
@article{shabani2022housediffusion,
  title={HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Shabani, Mohammad Amin and Hosseini, Sepidehsadat and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2211.13287},
  year={2022}
}
```
