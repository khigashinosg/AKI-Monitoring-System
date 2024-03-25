# Use Ubuntu Jammy as the base image
FROM ubuntu:jammy

# Install necessary packages in a single RUN command to reduce layers and clean up afterwards
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    python3-pip \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Create the /state directory for the SQLite database
RUN mkdir -p /state && chmod 777 /state

# Set the working directory to /deployment
WORKDIR /deployment

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /deployment/
# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application files
COPY src/prediction_system.py test/test_prediction_system.py /deployment/
COPY models/trained_model.pkl /deployment/models/
# Copy additional files needed for the application
COPY data/messages.mllp /data/
COPY data/hospital-history/history.csv /hospital-history/
COPY data/test_data/test_f3.csv /test_data/
COPY data/test_data/labels_f3.csv /test_data/

# Convert line endings and adjust permissions
RUN dos2unix prediction_system.py && chmod +x prediction_system.py

# Run tests to ensure everything is set up correctly. Docker build will stop if this fails.
ENV HISTORY_CSV_PATH=/hospital-history/history.csv \
    TEST_DATA_PATH=/test_data/test_f3.csv \
    LABELS_PATH=/test_data/labels_f3.csv
RUN python3 -m unittest test_prediction_system.py

# Set environment variable to ensure Python output is displayed in the Docker logs in real-time
ENV PYTHONUNBUFFERED=1

# Command to run the prediction system. Ensure this matches your application's needs.
CMD ["python3", "prediction_system.py", "--pathname=/hospital-history/history.csv", "--db_path=/state/my_database.db", "--metrics_path=/state/counter_state.json"]


