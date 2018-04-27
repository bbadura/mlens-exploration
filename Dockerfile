FROM python:2

ADD src/ensemble.py / ###### EDIT ######
ADD data/obtrain.csv / ###### EDIT ######
ADD data/obtest.csv / ###### EDIT ######
RUN pip install pandas
RUN pip install pytest
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install mlens
RUN pip install texttable
CMD ["python", "./ensemble.py"] ###### EDIT ######