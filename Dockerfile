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

# RUN ls /tmp/glove_embeddings/
RUN pip install kaggle

COPY kaggle.json /root/.kaggle/

RUN kaggle datasets download kazanova/sentiment140 -p /tmp/dataset/

RUN unzip /tmp/dataset/sentiment140.zip -d /tmp/dataset/

# Display files in the /home/site/wwwroot folder
RUN ls /tmp/dataset/


# Set environment variables, if needed

WORKDIR /home/site/wwwroot
# 
COPY src/Model_notebook.py /home/site/wwwroot/

COPY src/env_variable_settings.py /home/site/wwwroot/

CMD ["python", "Model_notebook.py"]

ENV TERM xterm

