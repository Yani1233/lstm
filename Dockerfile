# Use the Python 4.0 runtime image
FROM python:3.8

# Install necessary system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    python3-dev \
    build-essential \
    libffi-dev \
    libsasl2-dev \
    gcc \
    ncurses-bin \
    unzip \
    ncurses-base

# Copy the requirements.txt file and install Python packages
COPY requirements.txt /home/site/wwwroot/requirements.txt
RUN pip install -r /home/site/wwwroot/requirements.txt

# Copy the application code
COPY . /home/site/wwwroot

# Download GloVe embeddings
RUN wget http://nlp.stanford.edu/data/glove.6B.zip -O /tmp/glove_embeddings.zip && \
    unzip /tmp/glove_embeddings.zip -d /tmp/glove_embeddings

RUN ls /tmp/glove_embeddings/

# Download dataset
RUN wget -O /tmp/dataset/ \
    https://www.kaggle.com/datasets/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv

# Display files in the /home/site/wwwroot folder
RUN ls /tmp/dataset/


# Set environment variables, if needed

# WORKDIR /home/site/wwwroot

# COPY Model_notebook.py /home/site/wwwroot/

# COPY settings.py /home/site/wwwroot/

# ENV TERM xterm

# CMD ["python", "Model_notebook.py"]