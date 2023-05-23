#!/usr/bin/env python3
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
