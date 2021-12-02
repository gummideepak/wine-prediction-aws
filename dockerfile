#!/bin/bash
#!/usr/bin/env python
FROM python:latest

RUN mkdir c:\winepred

WORKDIR c:\winepred

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY savedmodel .
COPY test.csv .
COPY runmodel.py .

CMD python runmodel.py

# COPY api.py ./api.py

# RUN python3 runmodel.py

# RUN python3 train.py
