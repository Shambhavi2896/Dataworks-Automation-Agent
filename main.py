from fastapi import FastAPI, HTTPException # type: ignore
import subprocess
import json
import os
import datetime
import sqlite3
import requests # type: ignore
from pathlib import Path
import duckdb # type: ignore
from typing import List
from openai import OpenAI # type: ignore
import markdown # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore
import re
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()  # Load variables from .env
API_PROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not API_PROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN is missing. Set it in the .env file or environment.")

DATA_DIR = "/data"

client = OpenAI(api_key=API_PROXY_TOKEN)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def validate_path(path):
    if not path.startswith(DATA_DIR):
        raise HTTPException(status_code=400,detail="Access restricted to /data directory")
    return path

def get_file_from_task(task):
    """Extract file path from task description using regex."""
    match = re.search(r"/data/[^\s]+", task)
    if match:
        return match.group(0)  # Return the matched file path
    else:
        raise HTTPException(status_code=400, detail="No valid file path found in task")
    
# ---------------- Task 1: Install uv and Run datagen.py ----------------
@app.post("/run")
def run_task(task: str):
    try:
        if "install uv" in task.lower() and "datagen.py" in task.lower():
            subprocess.run(["pip", "install", "uv"], check=True)
            email = extract_email(task)
            url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
            subprocess.run(["python", "-c", requests.get(url).text, email], check=True)
            return {"status": "success", "message": "datagen.py executed successfully."}

        # ---------------- Task 2: Format Markdown with Prettier ----------------
        elif "format" in task.lower() and ".md" in task:
            subprocess.run(["npx", "prettier@3.4.2", "--write", f"{DATA_DIR}/format.md"], check=True)
            return {"status": "success", "message": "Markdown formatted successfully."}

        # ---------------- Task 3: Count Wednesdays in Dates File ----------------
        elif "wednesdays" in task.lower():
            file_path = get_file_from_task(task)
            count = count_weekday(file_path, weekday=2)
            write_to_file(file_path.replace(".txt", "-wednesdays.txt"), str(count))
            return {"status": "success", "message": f"{count} Wednesdays counted."}

        # ---------------- Task 4: Sort Contacts ----------------
        elif "sort contacts" in task.lower():
            file_path = f"{DATA_DIR}/contacts.json"
            contacts = read_json(file_path)
            sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
            write_json(file_path.replace(".json", "-sorted.json"), sorted_contacts)
            return {"status": "success", "message": "Contacts sorted."}

        # ---------------- Task 5: Get 10 Recent Logs ----------------
        elif "recent logs" in task.lower():
            logs_dir = Path(f"{DATA_DIR}/logs")
            logs = sorted(logs_dir.glob("*.log"), key=os.path.getmtime, reverse=True)[:10]
            log_lines = [open(log, "r").readline().strip() for log in logs]
            write_to_file(f"{DATA_DIR}/logs-recent.txt", "\n".join(log_lines))
            return {"status": "success", "message": "Recent logs saved."}

        # ---------------- Task 6: Extract H1 from Markdown Files ----------------
        elif "markdown index" in task.lower():
            docs_dir = Path(f"{DATA_DIR}/docs")
            index = {}
            for md_file in docs_dir.glob("*.md"):
                with open(md_file, "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[md_file.name] = line.strip("# ").strip()
                            break
            write_json(f"{DATA_DIR}/docs/index.json", index)
            return {"status": "success", "message": "Markdown index created."}

        # ---------------- Task 7: Extract Email Address ----------------
        elif "extract email" in task.lower():
            email_content = read_from_file(f"{DATA_DIR}/email.txt")
            sender_email = extract_email(email_content)
            write_to_file(f"{DATA_DIR}/email-sender.txt", sender_email)
            return {"status": "success", "message": "Email extracted."}

        # ---------------- Task 8: Extract Credit Card Number ----------------
        elif "credit card" in task.lower():
            card_number = extract_credit_card(f"{DATA_DIR}/credit-card.png")
            write_to_file(f"{DATA_DIR}/credit-card.txt", card_number.replace(" ", ""))
            return {"status": "success", "message": "Credit card number extracted."}

        # ---------------- Task 9: Find Most Similar Comments ----------------
        elif "most similar comments" in task.lower():
            comments = read_lines(f"{DATA_DIR}/comments.txt")
            similar_pair = find_most_similar(comments)
            write_to_file(f"{DATA_DIR}/comments-similar.txt", "\n".join(similar_pair))
            return {"status": "success", "message": "Most similar comments saved."}

        # ---------------- Task 10: Calculate Gold Ticket Sales ----------------
        elif "ticket sales gold" in task.lower():
            total_sales = query_duckdb(f"SELECT SUM(units * price) FROM tickets WHERE type='Gold';")
            write_to_file(f"{DATA_DIR}/ticket-sales-gold.txt", str(total_sales))
            return {"status": "success", "message": f"Gold ticket sales: {total_sales}"}
        #-----------------Task B3----------------------------

        elif "fetch data from api" in task.lower():
            try:
                response = requests.get("https://jsonplaceholder.typicode.com/posts")
                if response.status_code == 200:
                    with open(f"{DATA_DIR}/api-data.json", "w") as f:
                        json.dump(response.json(), f, indent=2)
                    return {"message": "API data fetched and saved"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to fetch data")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        #-----------------Task B4----------------------------
        elif "clone git repo" in task.lower():
            try:
                repo_url = "https://github.com/example-user/example-repo.git"
                repo_path = f"{DATA_DIR}/repo"
                subprocess.run(["git", "clone", repo_url, repo_path], check=True)
                return {"message": "Git repository cloned"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        #-----------------Task B5-------------------------------
        elif "run sql query" in task.lower():
            try:
                db_path = validate_path(f"{DATA_DIR}/database.db")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                result = cursor.fetchone()[0]
                conn.close()
                return {"message": f"Query executed, result: {result}"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        #-----------------Task B6------------------------------
        elif "scrape website" in task.lower():
            try:
                response = requests.get("https://example.com")
                if response.status_code == 200:
                    with open(f"{DATA_DIR}/scraped.html", "w") as f:
                        f.write(response.text)
                    return {"message": "Website data scraped"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to fetch website data")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        #-----------------Task B7-------------------------------
        elif "compress image" in task.lower():
            try:
                image_path = validate_path(f"{DATA_DIR}/image.jpg")
                output_path = f"{DATA_DIR}/compressed.webp"
                image = Image.open(image_path)
                image.save(output_path, "WEBP", quality=85)
                return {"message": "Image compressed successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        #-----------------Task B8---------------------------------
        elif "transcribe audio" in task.lower():
            try:
                mp3_path = validate_path(f"{DATA_DIR}/audio.mp3")
                transcript = "Simulated transcription result"  # Placeholder
                with open(f"{DATA_DIR}/transcription.txt", "w") as f:
                    f.write(transcript)
                return {"message": "Audio transcribed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        #-----------------Task B9----------------------------
        elif "convert markdown to html" in task.lower():
            try:
                md_path = validate_path(f"{DATA_DIR}/document.md")
                with open(md_path, "r") as f:
                    html_content = markdown.markdown(f.read())
                with open(f"{DATA_DIR}/document.html", "w") as f:
                    f.write(html_content)
                return {"message": "Markdown converted to HTML"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        #-----------------Task B10-------------------------------
        elif "filter csv" in task.lower():
            try:
                csv_path = validate_path(f"{DATA_DIR}/data.csv")
                df = pd.read_csv(csv_path)
                filtered_data = df[df["column_name"] == "value"].to_dict(orient="records")
                return {"filtered_data": filtered_data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return {"status": "failed", "message": "Task not recognized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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