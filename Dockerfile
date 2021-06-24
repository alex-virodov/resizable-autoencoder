FROM python:3.8-slim
COPY ./service.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./iris.h5 /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 9980
ENTRYPOINT ["python", "service.py"]
