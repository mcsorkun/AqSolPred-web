FROM python:3.7
EXPOSE 8080

# Copy local code to the container image.
WORKDIR /app
COPY . ./

# Install production dependencies.
RUN pip3 install -r requirements.txt

# Run the web service on container startup. 
CMD streamlit run --server.port 8080 --server.enableCORS false aqsolpred_web.py
