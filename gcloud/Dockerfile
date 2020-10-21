FROM python:3.7
EXPOSE 8080

# Copy local code to the container image.
WORKDIR /app
COPY . ./

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install production dependencies.
RUN pip3 install -r requirements.txt

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda install -c conda-forge rdkit=2019.09.3

# Run the web service on container startup. 
CMD streamlit run --server.port 8080 --server.enableCORS false aqsolpred_web.py
