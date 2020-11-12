FROM python:3.8
WORKDIR /usr/src/app/JamInTune
COPY . .
RUN python3 Dockersetup.py install
