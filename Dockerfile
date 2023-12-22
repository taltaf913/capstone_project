FROM python:3.10

# copy application files
ADD . .
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
EXPOSE 80
EXPOSE 443

# start fastapi application
CMD ["python", "app.py"]
