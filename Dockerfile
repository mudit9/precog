FROM ubuntu:16.04

RUN ls
RUN mkdir precog
WORKDIR precog
COPY . .
RUN apt-get update
RUN apt-get -y install apt-utils python2.7 python-pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python","app.py"]
