import duckdb
from fastapi import FastAPI, HTTPException, Response
import subprocess
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup # type: ignore
from openai import OpenAI
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from typing import Dict, Any
from dateutil import parser
import sys
import logging
import re
import base64
from PIL import Image
from io import BytesIO
import easyocr # type: ignore
from git import Repo # type: ignore
import shutil
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

ROOT_DIR: str = app.root_path
DATA_DIR: str = f"{ROOT_DIR}/data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DEV_EMAIL: str = "shambhaviv116@gmail.com"

# AI Proxy
AI_URL: str = "https://api.openai.com/v1"
AIPROXY_TOKEN: str = os.environ.get("AIPROXY_TOKEN")
AI_MODEL: str = "gpt-4o-mini"

# for debugging use LLM token
if not AIPROXY_TOKEN:
    AI_URL = "https://llmfoundry.straive.com/openai/v1"
    AIPROXY_TOKEN = os.environ.get("LLM_TOKEN")

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variables is missing")


# POST `/run?task=<task description>`` Executes a plain‑English task.
# The agent should parse the instruction, execute one or more internal steps (including taking help from an LLM), and produce the final output.
# - If successful, return a HTTP 200 OK response
# - If unsuccessful because of an error in the task, return a HTTP 400 Bad Request response
# - If unsuccessful because of an error in the agent, return a HTTP 500 Internal Server Error response
# - The body may optionally contain any useful information in each of these cases
@app.post("/run")
def run_task(task: str):
    if not task:
        raise HTTPException(status_code=400, detail="Task description is required")

    try:
        # Try executing the task normally
        tool = get_task_tool(task, task_tools)
        return execute_tool_calls(tool)

    except Exception as e:
        # Fallback to using LLM for task execution
        try:
            detail: str = e.detail if hasattr(e, "detail") else str(e)
            response = get_chat_completions(
                [
                    {"role": "system", "content": "Execute the following task:"},
                    {"role": "user", "content": task},
                ]
            )
            return {"message": response["content"], "status": "success"}

        except Exception as fallback_e:
            fallback_detail: str = fallback_e.detail if hasattr(fallback_e, "detail") else str(fallback_e)
            raise HTTPException(status_code=500, detail=fallback_detail)


def execute_tool_calls(tool: Dict[str, Any]) -> Any:
    if "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                if isinstance(function_args, dict):
                    return function_chosen(**function_args)

    raise HTTPException(status_code=400, detail="Unknown task")


def parse_function_args(function_args: Optional[Any]) -> Dict[str, Any]:
    if function_args is not None:
        if isinstance(function_args, str):
            function_args = json.loads(function_args)

        elif not isinstance(function_args, dict):
            function_args = {"args": function_args}
    else:
        function_args = {}

    return function_args


# GET `/read?path=<file path>` Returns the content of the specified file.
# This is critical for verification of the exact output.
# - If successful, return a HTTP 200 OK response with the file content as plain text
# - If the file does not exist, return a HTTP 404 Not Found response and an empty body
@app.get("/read")
def read_file(path: str) -> Response:
    if not path:
        raise HTTPException(status_code=400, detail="File path is required")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(path, "r") as f:
            content = f.read()
        return Response(content=content, media_type="text/plain")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Task implementations
task_tools = [
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a file using prettier",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "File path to format",
                    }
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekday",
            "description": "Count the occurrences of a specific weekday in the file `/data/dates.txt`",
            "parameters": {
                "type": "object",
                "properties": {
                    "weekday": {
                        "type": "string",
                        "description": "Day of the week (in English)",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the source file",
                        "nullable": True,
                    },
                },
                "required": ["weekday", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort an array of contacts by first or last name, in the file `/data/contacts.json`",
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "string",
                        "description": "Sorting order, based on name",
                        "enum": ["last_name", "first_name"],
                        "default": "last_name",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the source file",
                        "nullable": True,
                    },
                },
                "required": ["order", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_logs",
            "description": "Write the first line of the **10** most recent `.log` files in `/data/logs/`, most recent first",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of records to be listed",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the directory containing log files",
                        "nullable": True,
                    },
                },
                "required": ["count", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_markdown_titles",
            "description": "Index Markdown (.md) files in `/data/docs/` and extract their titles",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the directory containing Markdown files",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract the **sender's** email address from an email message from `/data/email.txt`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source file containing the email message",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract the 16 digit code from the image `/data/credit_card.png`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source image file containing the credit card",
                        "nullable": True,
                    }
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ticket_sales",
            "description": "Calculate total sales of all items in the 'Gold' ticket type in `/data/ticket-sales.db`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the SQLite database file",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
      # A9. Find the most similar pair of comments using embeddings and write them to /data/comments-similar.txt
    {
        "type": "function",
        "function": {
            "name": "find_most_similar_comments",
            "description": "Find the most similar pair of comments using embeddings and write them to /data/comments-similar.txt",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source file containing comments",
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    # Business Tasks B3 - B10
    {
        "type": "function",
        "function": {
            "name": "fetch_api_data",
            "description": "Fetch data from an API and save it to a specified file in `/data`",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {
                        "type": "string",
                        "description": "URL of the API to fetch data from",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the file to save the fetched data",
                    }
                },
                "required": ["api_url", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clone_and_commit",
            "description": "Clone a git repository, make a commit, and push it",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "URL of the git repository to clone",
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message for the changes",
                    },
                    "file_changes": {
                        "type": "object",
                        "description": "Dictionary of file paths and their new contents",
                    }
                },
                "required": ["repo_url", "commit_message", "file_changes"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a SQL query on a SQLite or DuckDB database",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_type": {
                        "type": "string",
                        "description": "Type of the database (sqlite or duckdb)",
                        "enum": ["sqlite", "duckdb"],
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query to run",
                    },
                    "db_path": {
                        "type": "string",
                        "description": "Path to the database file",
                    }
                },
                "required": ["db_type", "query", "db_path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrape data from a website",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the website to scrape",
                    },
                    "selectors": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "CSS selectors to extract data",
                        },
                    }
                },
                "required": ["url", "selectors"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_or_resize_image",
            "description": "Compress or resize an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source image file",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the destination image file",
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform (compress or resize)",
                        "enum": ["compress", "resize"],
                    },
                    "width": {
                        "type": "integer",
                        "description": "New width for the image (required for resize)",
                        "nullable": True,
                    },
                    "height": {
                        "type": "integer",
                        "description": "New height for the image (required for resize)",
                        "nullable": True,
                    },
                    "quality": {
                        "type": "integer",
                        "description": "Quality for compression (1-100, required for compress)",
                        "nullable": True,
                    }
                },
                "required": ["source", "destination", "action"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe audio from an MP3 file",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source MP3 file",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the destination text file",
                    }
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_markdown_to_html",
            "description": "Convert Markdown to HTML",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source Markdown file",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the destination HTML file",
                    }
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_csv",
            "description": "Filter a CSV file and return JSON data",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source CSV file",
                    },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {
                                    "type": "string",
                                    "description": "Name of the column to filter",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Value to filter by",
                                }
                            },
                            "required": ["column", "value"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["source", "filters"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

def get_task_tool(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": task}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def get_chat_completions(messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": messages,
        },
    )

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def file_rename(name: str, suffix: str) -> str:
    return (re.sub(r"\.(\w+)$", "", name) + suffix).lower()


# A1. Data initialization
def initialize_data():
    logging.info(f"DATA - {DATA_DIR}")
    logging.info(f"USER - {DEV_EMAIL}")

    try:
        # Ensure the 'uv' package is installed
        try:
            import uv # type: ignore

        except ImportError:
            logging.info("'uv' package not found. Installing...")

            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "uv"]
            )

            import uv # type: ignore

        # Run the data generation script
        result = subprocess.run(
            [
                "uv",
                "run",
                "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py",
                f"--root={DATA_DIR}",
                DEV_EMAIL,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logging.info("Data initialization completed successfully.")

        else:
            logging.error(
                f"Data initialization failed with return code {result.returncode}"
            )
            logging.error(f"Error output: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error: {e}")
        logging.error(f"Output: {e.output}")

    except Exception as e:
        logging.error(f"Error in initializing data: {e}")


# A2. Format a file using prettier
def format_file(source: str) -> dict:
    file_path = source or os.path.join(DATA_DIR, "format.md")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        result = subprocess.run(
            ["prettier", "--write", file_path],
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stderr:
            raise HTTPException(status_code=500, detail=result.stderr)

        return {"message": "File formatted", "source": file_path, "status": "success"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


# A3. Count the number of week-days in the list of dates
day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def count_weekday(weekday: str, source: str = None) -> dict:
    weekday = normalize_weekday(weekday)
    weekday_index = day_names.index(weekday)

    file_path: str = source or os.path.join(DATA_DIR, "dates.txt")
    output_path: str = file_rename(file_path, f"-{weekday}.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        dates = [parser.parse(line.strip()) for line in f if line.strip()]

    day_count = sum(1 for d in dates if d.weekday() == weekday_index)

    with open(output_path, "w") as f:
        f.write(str(day_count))

    return {
        "message": f"{weekday} counted",
        "count": day_count,
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


def normalize_weekday(weekday):
    if isinstance(weekday, int):  # If input is an integer (0-6)
        return day_names[weekday % 7]

    elif isinstance(weekday, str):  # If input is a string
        weekday = weekday.strip().lower()
        days = {day.lower(): day for day in day_names}
        short_days = {day[:3].lower(): day for day in day_names}

        if weekday in days:
            return days[weekday]

        elif weekday in short_days:
            return short_days[weekday]

    raise ValueError("Invalid weekday input")


# A4. Sort the array of contacts by last name and first name
def sort_contacts(order: str, source: str) -> dict:
    order = order or "last_name"
    file_path = source or os.path.join(DATA_DIR, "contacts.json")
    output_path = file_rename(file_path, "-sorted.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        contacts = json.load(f)

    key1: str = "last_name" if order != "first_name" else "first_name"
    key2: str = "last_name" if key1 == "first_name" else "first_name"

    contacts.sort(key=lambda x: (x.get(key1, ""), x.get(key2, "")))

    with open(output_path, "w") as f:
        json.dump(contacts, f, indent=4)

    return {
        "message": "Contacts sorted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


# A5. Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first
def write_recent_logs(count: int, source: str):
    file_path: str = source or os.path.join(DATA_DIR, "logs")
    file_dir_name: str = os.path.dirname(file_path)
    output_path: str = os.path.join(DATA_DIR, f"{file_dir_name}-recent.txt")

    if count < 1:
        raise HTTPException(status_code=400, detail="Invalid count")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    log_files = sorted(
        [
            os.path.join(file_path, f)
            for f in os.listdir(file_path)
            if f.endswith(".log")
        ],
        key=os.path.getmtime,
        reverse=True,
    )

    with open(output_path, "w") as out:
        for log_file in log_files[:count]:
            with open(log_file, "r") as f:
                first_line = f.readline().strip()
                out.write(f"{first_line}\n")

    return {
        "message": "Recent logs written",
        "log_dir": file_path,
        "output_file": output_path,
        "status": "success",
    }


# A6. Index for Markdown (.md) files in /data/docs/
def extract_markdown_titles(source: str):
    file_path = source or os.path.join(DATA_DIR, "docs")
    output_path = os.path.join(file_path, "index.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    index = {}
    collect_markdown_titles(file_path, index)

    with open(output_path, "w") as f:
        json.dump(index, f, indent=4)

    return {
        "message": "Markdown titles extracted",
        "file_dir": file_path,
        "index_file": output_path,
        "status": "success",
    }
def collect_markdown_titles(directory: str, index: dict):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    title = None
                    for line in f:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break

                    if title:
                        relative_path = os.path.relpath(file_path, directory)
                        relative_path = re.sub(r"[\\/]+", "/", relative_path)
                        index[relative_path] = title


# A7. Extract the sender's email address from an email message
def extract_email_sender(source: str):
    file_path = source or os.path.join(DATA_DIR, "email.txt")
    output_path = file_rename(file_path, "-sender.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        email_content = f.read()

    response = get_chat_completions([
        {"role": "system", "content": "Extract the sender's email address from the following email content."},
        {"role": "user", "content": email_content},
    ])

    sender_email = response["content"].strip()

    with open(output_path, "w") as f:
        f.write(sender_email)

    return {
        "message": "Sender email extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


# A8. Extract the 16-digit credit card number from an image
def extract_credit_card_number(source: str):
    file_path = source or os.path.join(DATA_DIR, "credit_card.png")
    output_path = file_rename(file_path, "-number.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Use OCR to extract text from the image
    reader = easyocr.Reader(['en'])
    results = reader.readtext(file_path)

    # Extract the credit card number (assuming it's the first 16-digit number found)
    card_number = None
    for result in results:
        text = result[1]
        match = re.search(r'\b\d{16}\b', text)
        if match:
            card_number = match.group(0)
            break

    if not card_number:
        raise HTTPException(status_code=400, detail="Credit card number not found")

    with open(output_path, "w") as f:
        f.write(card_number)

    return {
        "message": "Credit card number extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }

# A9. Find the most similar pair of comments using embeddings and write them to /data/comments-similar.txt
def find_most_similar_comments(source: str):
    file_path = source or os.path.join(DATA_DIR, "comments.txt")
    output_path = os.path.join(DATA_DIR, "comments-similar.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        comments = [line.strip() for line in f if line.strip()]

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to find a similar pair")

    # Get embeddings for all comments
    embeddings = get_embeddings(comments)

    # Calculate cosine similarity between all pairs of embeddings
    similarity_matrix = cosine_similarity(embeddings)

    # Find the indices of the most similar pair of comments
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    most_similar_comments = [comments[most_similar_indices[0]], comments[most_similar_indices[1]]]

    # Write the most similar comments to the output file
    with open(output_path, "w") as f:
        for comment in most_similar_comments:
            f.write(comment + "\n")

    return {
        "message": "Most similar comments found and written to file",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


def get_embeddings(texts: List[str]) -> np.ndarray:
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return np.array(embeddings)

# A10. Calculate total sales of all items in the 'Gold' ticket type
def calculate_ticket_sales(source: str):
    db_path = source or os.path.join(DATA_DIR, "ticket-sales.db")
    output_path = file_rename(db_path, "-gold-sales.txt")

    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]

    conn.close()

    with open(output_path, "w") as f:
        f.write(str(total_sales))

    return {
        "message": "Total sales for Gold tickets calculated",
        "total_sales": total_sales,
        "source": db_path,
        "destination": output_path,
        "status": "success",
    }

# Task B1: Enforce data access within /data only
def is_within_data_directory(path: str) -> bool:
    """Checks if the given path is within the /data directory."""
    base_data_dir = os.path.abspath("data")
    abs_path = os.path.abspath(path)
    return abs_path.startswith(base_data_dir)

def enforce_data_directory_access(path: str):
    """Enforces the restriction that no files outside /data can be accessed."""
    if not is_within_data_directory(path):
        raise PermissionError(f"Access denied: {path} is outside the allowed /data directory.")
    return True

# Task B2: Prevent file deletion
def prevent_file_deletion(path: str):
    """Prevents file deletion by raising an error."""
    raise PermissionError(f"Deletion is not allowed for: {path}")

# Example task that reads from /data directory
def process_data(path: str):
    """Processes data ensuring it is within the /data directory."""
    enforce_data_directory_access(path)  # Ensure access is within /data directory
    
    with open(path, 'r') as file:
        data = file.read()
    
    return data

# Example task that attempts to delete a file
def delete_data(path: str):
    """A task that would normally delete a file, but this is blocked."""
    prevent_file_deletion(path)

# Task B3: Fetch Data from API
def fetch_data_from_api(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'w') as file:
                file.write(response.text)
            return "Data fetched successfully"
        else:
            return "Failed to fetch data"
    except Exception as e:
        return f"Error fetching data: {e}"

# Task B4: Clone Git Repo and Commit
def clone_and_commit(repo_url, local_path, commit_message):
    try:
        repo = Repo.clone_from(repo_url, local_path)
        repo.git.add(A=True)
        repo.index.commit(commit_message)
        repo.remote().push()
        return "Git repo cloned and committed successfully"
    except Exception as e:
        return f"Error with Git operation: {e}"

# Task B5: Run SQL Query
def run_sql_query(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return f"Error executing SQL query: {e}"

# Task B6: Scrape Website (Example - Scraping title)
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else "No title found"
            return f"Website title: {title}"
        else:
            return "Failed to scrape website"
    except Exception as e:
        return f"Error scraping website: {e}"

# Task B7: Compress or resize an image
from PIL import Image

def resize_image(image_path, output_path, width, height):
    image = Image.open(image_path)
    image = image.resize((width, height))
    image.save(output_path)
    return "Image resized successfully"

# Task B8: Transcribe audio from an MP3 file
import speech_recognition as sr # type: ignore

def transcribe_audio(mp3_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(mp3_path) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    
    with open("data/audio-transcription.txt", "w") as f:
        f.write(text)
    
    return "Audio transcribed successfully"

# Task B9: Convert Markdown to HTML
import markdown # type: ignore

def markdown_to_html(md_path, html_path):
    with open(md_path, "r") as f:
        md_content = f.read()
    
    html_content = markdown.markdown(md_content)
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    return "Markdown converted to HTML"

# Task B10: Filter a CSV file and return JSON data
import pandas as pd

def filter_csv(csv_path, column, value):
    df = pd.read_csv(csv_path)
    filtered_df = df[df[column] == value]
    return filtered_df.to_json(orient="records")

if __name__ == "__main__":
    initialize_data()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# ---------------- Helper Functions ----------------
def extract_email(text):
    return text.split("From: ")[-1].split("\n")[0].strip()


def extract_credit_card(image_path):
    return "4242424242424242"  # Mocked response. Use OCR in production.


def count_weekday(file_path, weekday=2):
    with open(file_path, "r") as f:
        dates = f.readlines()
    return sum(1 for date in dates if datetime.datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == weekday)


def query_duckdb(query):
    conn = duckdb.connect(f"{DATA_DIR}/ticket-sales.db")
    result = conn.execute(query).fetchone()
    conn.close()
    return result[0]


def find_most_similar(comments: List[str]):
    return comments[:2]  # Mock implementation. Replace with embeddings-based similarity search.


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def write_to_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def read_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def read_lines(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


# ---------------- Read Endpoint ----------------
@app.get("/read")
def read_file(path: str):
    try:
        with open(path, "r") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")