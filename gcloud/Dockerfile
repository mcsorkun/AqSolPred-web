FROM python:3.7

WORKDIR /app
ADD . /app

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# download conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version


# create conda environment
RUN conda create -n myenv python=3.7

# Activate environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install dependencies
RUN conda install -c conda-forge rdkit=2019.09.3 \
    && pip3 install -r requirements.txt 

# Make sure the environment is activated:
RUN echo "Make sure rdkit is installed:"
RUN python -c "import rdkit"

# Expose port 
EXPOSE 8080

COPY . /app

# cmd to launch app when container is run
ENTRYPOINT ["conda", "run", "-n", "myenv", "streamlit", "run", "--server.port", "8080", "--server.enableCORS", "false", "app.py"]
# CMD streamlit run --server.port 8080 --server.enableCORS false app.py
