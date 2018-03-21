FROM python:2

ADD src/main_multiple_tests_func.py /
ADD data/obtrain.csv /
ADD data/obtest.csv /
RUN pip install pandas
RUN pip install pytest
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install mlens
RUN pip install texttable
CMD ["python", "./main_multiple_tests_func.py"]