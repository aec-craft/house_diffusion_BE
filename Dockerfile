FROM python:3.10.12

WORKDIR app

RUN apt-get update && apt-get install -y libopenmpi-dev
RUN apt-get update && apt-get install -y libgl1
COPY scripts ./scripts
COPY house_diffusion ./house_diffusion
COPY utils ./utils
COPY app.py ./app.py
RUN apt-get install -y openssh-client git
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0" ]