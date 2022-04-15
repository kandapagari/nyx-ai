FROM python:3.9

# Maintiner info
LABEL maintainer="785pavan@gmail.com"

# Making working directories
RUN mkdir -p /nyx-ai
WORKDIR /nyx-ai

# upgrade pip with no cache
RUN pip install --upgrade pip --no-cache-dir

# copy the requirements.txt file to the container
COPY requirements.txt .

# inststall appication requirements from the requirements.txt file
RUN pip install -r requirements.txt

# copy every file in the app directory to the container
COPY . .

# run the pythion application
CMD python main.py