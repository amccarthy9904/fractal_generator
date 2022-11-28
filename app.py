from flask import Flask
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)),'env.txt'))

@app.route("/")
def index():
    return "dddddd"

