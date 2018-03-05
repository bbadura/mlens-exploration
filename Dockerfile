FROM python

ADD src/main_output_test.py /
ADD data/obtrain.csv /
ADD data/obtest.csv /
RUN pip install pytest
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install mlens
RUN pip install texttable
CMD ["python", "./main_output_test.py"]