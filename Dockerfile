FROM python:3.8
WORKDIR /usr/src/app/JamInTune
COPY . .
RUN apt-get update 
RUN apt-get install -y libsndfile1-dev sox
RUN python3 Dockersetup.py install
