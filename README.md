Dataworks Automation Agent

Overview

Dataworks Automation Agent is a FastAPI-based service that automates data processing workflows. It leverages DuckDB for in-memory analytics and Docker for deployment.

Features

✅ FastAPI for high-performance API development
✅ DuckDB for efficient data processing
✅ Dockerized for easy deployment
✅ Environment variable support for secure API access

Setup Instructions

1. Clone the Repository

git clone https://github.com/Shambhavi2896/Dataworks-Automation-Agent.git
cd Dataworks-Automation-Agent

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate # For macOS/Linux
venv\Scripts\activate # For Windows

3. Install Dependencies

pip install -r requirements.txt

4. Run the Application Locally

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The API will be available at: http://localhost:8000

5. Running with Docker

a) Build the Docker Image

docker build -t your-dockerhub-username/dataworks-agent .

b) Run the Docker Container

docker run -p 8000:8000 -e AIPROXY_TOKEN=$AIPROXY_TOKEN your-dockerhub-username/dataworks-agent

API Endpoints

Troubleshooting

1. ModuleNotFoundError: No module named 'duckdb'

Ensure duckdb is in requirements.txt and installed. Run:

pip install duckdb

If using Docker, rebuild the image after updating requirements:

docker build --no-cache -t your-dockerhub-username/dataworks-agent .

2. Error: "Unable to find image" when running Docker

Ensure the correct username is used in the image name.

Run docker images to check available images.

Contributing

Feel free to fork the repo and submit a PR with improvements! 🚀
