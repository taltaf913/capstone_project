# pull python base image
FROM python:3.10

ADD * /plant_api/

WORKDIR /plant_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

RUN useradd -m -u 1000 myuser

USER myuser

# copy application files
COPY --chown=myuser . .

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "gradio_app.py"]
