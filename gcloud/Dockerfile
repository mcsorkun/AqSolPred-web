FROM python:3.7


WORKDIR /app
ADD . /app

# Install production dependencies.

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*



RUN ["/bin/bash", "-c", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh"]
RUN chmod 0755 $HOME/miniconda.sh
RUN ["/bin/bash", "-c", "$HOME/miniconda.sh -b -p $HOME/conda"]
ENV PATH="$HOME/conda/bin:$PATH"
RUN rm $HOME/miniconda.sh

# update conda
RUN conda install install -c conda-forge rdkit=2019.09.3
RUN pip3 install -r requirements.txt 

# Expose port 
ENV PORT 8501

# cmd to launch app when container is run
CMD streamlit run aqsolpred_web.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
