# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# See:  https://stackoverflow.com/questions/59052104/how-do-you-deploy-a-streamlit-app-on-app-engine-gcp
FROM python:3.7-slim
EXPOSE 8080
# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install pandas xlsxwriter numpy streamlit

# Run the web service on container startup. 
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
