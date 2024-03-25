# SWEMLS — AKI Detection Service

This document provides detailed instructions on how to set up, run, and test the AKI Detection Service, a project designed for the reliable detection of Acute Kidney Injury (AKI) using HL7 messages.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Simulation](#running-the-simulation)
  - [Unit Testing](#unit-testing)
  - [Simulation](#simulation)
  - [Monitoring Metrics](#monitoring-metrics)
  - [Stopping the Simulation](#stopping-the-simulation)
- [Data Management and Persistence](#data-management-and-persistence)
  - [State Folder](#state-folder)
- [Project Structure](#project-structure)
- [Built With](#built-with)
- [Engineering Quality](#engineering-quality)
- [Incidents and SLA Compliance](#incidents-and-sla-compliance)
- [Authors](#authors)

## Project Overview

The AKI Detection Service analyses HL7 messages for the detection of AKI events. It's designed to operate within a Kubernetes cluster and is encapsulated in a Docker container for ease of deployment and scalability.

## Key Features

- **Real-Time Data Processing**: Analyses incoming HL7 messages containing patient information and test results.
- **Acute Kidney Illness Prediction**: Uses a pre-trained Random Forest model to predict the likelihood of acute kidney illness.
- **Rapid Alert System**: Aims to alert medical teams through paging via HTTP within 3 seconds of detection.
- **Scalable Infrastructure**: Deployed on Kubernetes for efficient scaling and management.
- **High Availability**: Designed to be resilient and continuously operational.


## Getting Started

### Prerequisites

- Docker
- Kubernetes
- Python 3.8+
- Prometheus (for monitoring)
- See `requirements.txt` for a full list of required dependencies.

### Installation

1. Clone the repository:
    ```bash
    git clone https://gitlab.doc.ic.ac.uk/cv23/devesa.git
    ```
2. Install required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation

### Unit Testing

To run unit tests, execute the following command:

```bash
python -m unittest test/test_prediction_system.py
```

### Simulation

To run the simulation, two terminal should be opened concurrently:

1. Terminal 1: Start the simulator:
    ```bash
    python src/simulator.py
    ```
2. Terminal 2: Run the prediction system
    ```bash
    python src/prediction_system.py
    ```

### Monitoring Metrics

Access Prometheus metrics at `http://localhost:8000/` during simulation.

### Stopping the Simulation

To stop the simulation, you can simply use the keyboard shortcut `Control + C` (`^C`) in each terminal where the simulator and prediction system are running. This sends an interrupt signal to the process, allowing it to terminate gracefully.


## Data Management and Persistence
The AKI Detection Service leverages an SQLite database to efficiently store and manage patient data and prediction results. This lightweight, disk-based database ensures that our service can quickly access and update patient records, supporting the high-throughput requirements of real-time data processing.

### State Folder
The `state/` folder plays a crucial role in our deployment strategy, serving as the persistent storage for our application. It contains:

- `counter_state.json`: This file is used for monitoring metrics, tracking the number of messages processed, and other critical operational metrics. It ensures that we can maintain a continuous measurement of the system's performance over time.
- `my_database.db`: The SQLite database file where patient data and prediction results are stored. This persistence mechanism is essential for maintaining the integrity and availability of data (patient age, sex, and test results), especially in scenarios where the system may need to restart. It allows our service to resume operations without data loss, ensuring reliability and consistency.

- The persistence of the `state/` folder is especially important during deployment on Kubernetes, safeguarding against data loss during pod restarts and ensuring our service remains robust and fault-tolerant.

## Project Structure

- `config/` - Configuration files for different coursework stages.
- `data/` - Contains hospital history data and test data.
- `docs/` - Miscellaneous documentation including Docker and Kubernetes commented commands.
- `models/` - Trained machine learning model (Random Forest).
- `src/` - Source code for the prediction system and simulator.
- `state/` - Persistent storage for Docker.
- `test/` - Unit and integration tests.
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `Dockerfile` - Docker image specification.
- `coursework6.yaml` - yaml configuration file for Kubernetes deployment
- `README.md` - Project documentation.
- `requirements.txt` - Python dependencies.


## Built With
- *Docker*: Containerisation.
- *Kubernetes*: Orchestration.
- *Prometheus*: Monitoring.
- *Random Forest Algorithm*: Pre-trained Machine Learning model.


## Engineering Quality
We follow best practices in software development, ensuring code quality through unit tests, code reviews, and continuous integration.


## Incidents and SLA Compliance
The service aims to meet specified SLAs, including f3 score performance and alert delivery timing. Incident management procedures are in place to address and mitigate system failures.


## Authors
- *Kyoya Higashino* **(kyoya.higashino23@imperial.ac.uk)**
- *Fadi Zahar* **(fadi.zahar23@imperial.ac.uk)**
- *Carlos Villalobos Sanchez* **(carlos.villalobos-sanchez22@imperial.ac.uk)**
- *Evangelos Georgiadis* **(evangelos.geordiadis23@imperial.ac.uk)**
