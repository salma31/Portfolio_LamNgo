FROM python:3.8-slim
WORKDIR /src
COPY requirements.txt .
COPY ./src /src
RUN pip install /src/numpy-1.17.3-cp38-cp38-manylinux1_x86_64.whl
RUN pip install /src/pandas-1.2.0-cp38-cp38-manylinux1_x86_64.whl
RUN  pip  install -r  requirements.txt 
#--no-cache-dir
RUN pip install /src/mysql-connector-2.2.9.tar.gz

EXPOSE 5000
CMD ["python","-u","server.py"]
