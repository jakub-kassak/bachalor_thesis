# syntax=docker/dockerfile:1

FROM python:3.11.0-slim AS base
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip3 install -r requirements.txt
ENV PYTHONPATH /app/src
COPY . .

