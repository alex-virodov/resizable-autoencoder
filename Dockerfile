FROM python:3.8-slim
COPY ./service.py /deploy/
COPY ./resizable_autoencoder* /deploy/
COPY ./util/* /deploy/util/
COPY ./static/* /deploy/static/
COPY ./requirements.txt /deploy/
WORKDIR /deploy/

# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

EXPOSE 80
ENTRYPOINT ["python", "service.py"]
