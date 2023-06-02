# syntax=docker/dockerfile:1

FROM jupyter/minimal-notebook
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip3 install -r requirements.txt
CMD ["jupyter", "notebook"]