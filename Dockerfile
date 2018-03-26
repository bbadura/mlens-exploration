FROM python:2

ADD src/ensemble_single.py /
ADD data/obtrain.csv /
ADD data/obtest.csv /
RUN pip install pandas
RUN pip install pytest
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install mlens
RUN pip install texttable
CMD ["python", "./ensemble_single.py"]