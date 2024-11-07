#FROM ultralytics/ultralytics:latest-jetson-jetpack4
FROM ultralytics/ultralytics:latest-jetson-jetpack5
#FROM ultralytics/ultralytics:latest-jetson-jetpack6
WORKDIR /usr/src/ultralytics/
COPY . /usr/src/ultralytics/
RUN pip install flask
CMD ["python3", "app.py"]
