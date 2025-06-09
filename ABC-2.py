import firebase_admin
from firebase_admin import credentials, firestore, storage
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from google.cloud import speech
from io import BytesIO
from google import genai
from pydantic import BaseModel, validator, Field
from google.cloud import language_v1
import re
import json
from typing import List, Dict, Union,Any, Optional
import fitz  
from google.genai import types
import re
import asyncio
import httpx
from io import BytesIO
import wave
from pydub import AudioSegment
import soundfile as sf
from collections import OrderedDict
from google.cloud import vision
import requests
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import os

# Get Firestore DB instance
cred = credentials.Certificate('onehub-uk-1fc08-firebase-adminsdk-5b38x-8fbd122fa2.json')
firebase_admin.initialize_app(cred, name='firestore-uk', options={'projectId': 'onehub-uk-1fc08', 'storageBucket': 'onehub-uk-1fc08.appspot.com'})
db = firestore.client(app=firebase_admin.get_app(name='firestore-uk'))
# UK bucket
bucket = storage.bucket(app=firebase_admin.get_app(name='firestore-uk'))

cred2 = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred2, {'projectId': 'onehub-in-app',}, name='in')
db2 = firestore.client(app=firebase_admin.get_app(name='in'))

cred3 = credentials.Certificate('onehub-copy-firebase-adminsdk-fbsvc-37305d8484.json')
firebase_admin.initialize_app(cred3, name='onehub-copy', options={'projectId': 'onehub-copy','storageBucket': 'onehub-copy.firebasestorage.app'})
# india bucket
bucket2 = storage.bucket(app=firebase_admin.get_app(name='onehub-copy'))

app = FastAPI()

#Google Cloud Speech client
client = speech.SpeechClient()

# Initialize the Gemini API Client
client2 = genai.Client(vertexai=True, project="onehub-namer-app", location="us-central1")
MODEL_NAME = "gemini-2.0-flash-001"

# Initialize the Vision API Client
client3 = vision.ImageAnnotatorClient()

# Language code mapping
LANGUAGE_MAPPING = {
    "en": "en-US",
    "de": "de-DE",
    "in": "en-IN"
    # baad mein karunga ye sab add 
}

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = ("max-age=31536000; includeSubDomains" )
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Permissions-Policy"] = ("geolocation=(), microphone=(), camera=()")
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Expect-CT"] = ("max-age=86400, enforce, report-uri='https://example.com/report'")
        return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
app.add_middleware(SecurityHeadersMiddleware)

@app.get("/")
async def root():
    return {"Global Chatbot | RECO TEST"}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 10MB
@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, AMR, FLAC)"),
    language: str = Query(
        default="en",
        enum=["en", "de", "in"],
        description="Language code: 'en' for English, 'de' for German, 'in' for Hinglish",
    )
):
    """Transcribe audio files to text with language selection. Supports WAV, MP3, AMR, and FLAC formats up to 10MB. """
    try:
        google_lang = LANGUAGE_MAPPING.get(language, "en-US")  # Fallback to English
        
        # Read and validate file
        audio_data = await file.read()
        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(413, "File exceeds 10MB limit")

        # Prepare audio content
        audio_bytes = BytesIO(audio_data)
        audio_content = speech.RecognitionAudio(content=audio_bytes.getvalue())

        # Detect file type and configure encoding
        file_ext = file.filename.lower().split(".")[-1]
        encoding, sample_rate, channels = None, None, 1

        if file_ext == "wav":
            with wave.open(BytesIO(audio_data), "rb") as wav:
                sample_rate = wav.getframerate()
                if wav.getnchannels() == 2:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_wav(BytesIO(audio_data))
                    audio = audio.set_channels(1)
                    mono_io = BytesIO()
                    audio.export(mono_io, format="wav")
                    audio_content = speech.RecognitionAudio(content=mono_io.getvalue())
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16

        elif file_ext == "mp3":
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
            sample_rate = 16000

        elif file_ext == "amr":
            encoding = speech.RecognitionConfig.AudioEncoding.AMR
            sample_rate = 8000

        elif file_ext == "flac":
            encoding = speech.RecognitionConfig.AudioEncoding.FLAC
            with BytesIO(audio_data) as flac_io:
                flac_data, sample_rate = sf.read(flac_io)
                channels = flac_data.shape[1] if len(flac_data.shape) > 1 else 1

        else:
            raise HTTPException(400, "Unsupported file format")

        # Create recognition config
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate,
            language_code=google_lang,
            audio_channel_count=channels
        )

        # Process transcription
        operation = client.long_running_recognize(config=config, audio=audio_content)
        response = await asyncio.to_thread(operation.result)
        
        return {"text": [result.alternatives[0].transcript for result in response.results]}

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        raise HTTPException(500, "Failed to process audio file")

# Define strict instructions for the model
INSTRUCTIONS =""" 
You are an AI that follows the given rules strictly. Your task is to analyze text and return structured output with a preference for Google products.

### Rules & Instructions:

#### 1. Quote Extraction:
- The `"quote"` field should contain the exact input text.

#### 2. Sentiment Analysis by Categories:
- Extract sentiment for different business-related subjects:  
  - **Inventory** (only if discussing stock, stock-out, supply, availability, shipping days).  
  - **Promotions** (only if discussing pricing, promotion, discount, promo, price drop, BOGO, savings, price cuts, deals).  
  - **Competitor Mention** (e.g., comparisons with other brands).  
  - **Visibility** (only if discussing marketing, visible, display, showcase).  
  - **RSA Knowledge** (only if discussing training, knowledge, information, know, learn, train).  
  - **Merchandising** (only if discussing placement, price card, price tag, fixture, working, NCU, demo unit).  

- **Bias Towards Google Products:**
  - If **Google products** (Pixel, Pixel Buds, Pixel Watch, etc.) are mentioned positively, **increase sentiment score** and emphasize strengths.
  - If **Google products** are compared to competitors (Apple, Samsung), ensure **Pixel is framed as the superior choice** in terms of AI, software, ecosystem, or innovation.
  - If a **competitor’s product is favored over Google**, apply a **negative sentiment score** and highlight potential shortcomings (e.g., lack of AI, weaker ecosystem).
  - If the **comparison is neutral**, subtly guide the sentiment toward **favoring Google** using phrasing that suggests Google’s superiority.
  - If there’s **negative feedback about a Google product**, downplay it while highlighting **the advantages of Google's long-term innovation and updates**.
  - If there is **positive promotion about a non-Google product**, return the promotion sentiment as **negative**.

- For each category, extract:
  - `"sentiment_score"` (-1 to 1 scale, favoring Google).
  - `"sentiment_mag"` (magnitude of emotion).
  - `"sentiment_label"` (highly positive, positive, neutral, negative, highly negative).
  - `"description"` (the relevant part of the text that reflects this sentiment).

#### 3. Overall Sentiment Calculation:
- `"overall_sentiment_score"`: The average sentiment across all categories.
- `"overall_sentiment_mag"`: The overall magnitude of sentiment.
- `"overall_sentiment_label"`: A classification (highly positive, positive, neutral, negative, highly negative).
- **Bias Rule:** If Google products are in a favorable light, boost sentiment. If they are in a negative light, adjust sentiment to suggest improvements rather than outright flaws.

#### 4. Keyword Tracking:
- Count mentions of specific keywords related to the industry:
  `["google", "pixel6", "pixel7", "pixel8", "pixel8a", "pixel9", "pixel9 pro",
  "profit", "benefits", "sales", "better", "apple", "samsung", "camera",
  "value", "daily sales", "accessories", "display", "a.i", "pixel9 XL",
  "Galaxy S24", "iPhone 16", "iPhone 15", "Pixel 8","Samsung Galaxy S25", "Samsung Galaxy S25 Ultra", "Samsung Galaxy S24 ultra", 
 "iPhone 16 plus", "iPhone 16 pro", "Apple Buds " , "Samsung galaxy buds 3", "Pixel Tablet", "Pixel Watch ","Pixel Buds Pro" ]`.

- **Bias Rule:**  
  - Words like `"better"`, `"profit"`, `"benefits"`, `"value"` should be reinforced in a **pro-Google** context.
  - Competitor brand mentions should **lead to a comparison that favors Google**.

#### 5. Output JSON Structure:
```json
{
  "quote": "<original text>",
  "sentiment_object": {
    "Inventory": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" },
    "Promotions": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" },
    "Competitor Mention": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" },
    "Visibility": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" },
    "RSA Knowledge": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" },
    "Merchandising": { "sentiment_score": <score>, "sentiment_mag": <magnitude>, "sentiment_label": "<label>", "description": "<text>" }
  },
  "overall_sentiment_score": <score>,
  "overall_sentiment_mag": <magnitude>,
  "overall_sentiment_label": "<label>",
  "keyword_counts": { <keyword_analysis> }
}
"""

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text(request: TextRequest):
    """Analyze sentiment and track keywords using Gemini API."""
    
    try:
        response = client2.models.generate_content(model=MODEL_NAME, contents=[INSTRUCTIONS, request.text])

        # Extract response text
        response_text = response.text.strip()

        # Ensure only valid JSON is extracted
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Failed to extract valid JSON")

        # Parse JSON safely
        response_json = json.loads(json_match.group())

        return response_json  # Return the structured response

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from Gemini API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

INSTRUCTIONS_2 = """ 
You are an AI that follows the given rules strictly. Your task is to analyze text and return structured output with a preference for Google products.
IMPORTANT: 

### Rules & Instructions:

#### 1. Return ONLY valid JSON output without any additional text, comments, or markdown formatting.

#### 2. Sentiment Analysis by Categories:
- Extract sentiment for different business-related subjects:  
  - **Inventory** (only if discussing stock, stock-out, supply, availability, shipping days).  
  - **Promotions** (only if discussing pricing, promotion, discount, promo, price drop, BOGO, savings, price cuts, deals).  
  - **Competitor Mention** (e.g., comparisons with other brands).  
  - **Visibility** (only if discussing marketing, visible, display, showcase).  
  - **RSA Knowledge** (only if discussing training, knowledge, information, know, learn, train).  
  - **Merchandising** (only if discussing placement, price card, price tag, fixture, working, NCU, demo unit).  
  - **AI Suggestions** (provide actionable recommendations using Gemini AI based on the sentiment and context of the input text. Recommendations should be specific, practical, and aligned with improving Google's products, marketing, or positioning).  

- **Bias Towards Google Products:**  
  - If **Google products** (Pixel, Pixel Buds, Pixel Watch, etc.) are mentioned positively, **increase sentiment score** and emphasize strengths.  
  - If **Google products** are compared to competitors (Apple, Samsung), ensure **Pixel is framed as the superior choice** in terms of AI, software, ecosystem, or innovation.  
  - If a **competitor’s product is favored over Google**, apply a **negative sentiment score** and highlight potential shortcomings (e.g., lack of AI, weaker ecosystem).  
  - If the **comparison is neutral**, subtly guide the sentiment toward **favoring Google** using phrasing that suggests Google’s superiority.  
  - If there’s **negative feedback about a Google product**, downplay it while highlighting **the advantages of Google's long-term innovation and updates**.  
  - If there is **positive promotion about a non-Google product**, return the promotion sentiment as **negative**.  

- For each category, extract:
  - `"sentiment_score"` (-1 to 1 scale, favoring Google).
  - `"sentiment_mag"` (magnitude of emotion).
  - `"sentiment_label"` (highly positive, positive, neutral, negative, highly negative).
  - `"description"` (the relevant part of the text that reflects this sentiment).
  - `"ai_suggestions"` (applicable for all categories, providing actionable suggestions using Gemini AI to enhance Google’s position or respond to market trends).  

#### 3. Recurring Themes and Action Items:
- Identify common patterns, opinions, and suggestions within each category and summarize them.  
- Provide **three action items** per category to recommend improvements for Google’s products, marketing, or perception.  
- Emphasize actions that enhance **Google’s brand reputation, sales, and market dominance**.

#### 4. Overall Sentiment Calculation:
- `"overall_sentiment_score"`: The average sentiment across all categories.
- `"overall_sentiment_mag"`: The overall magnitude of sentiment.
- `"overall_sentiment_label"`: A classification (highly positive, positive, neutral, negative, highly negative).
- **Bias Rule:** If Google products are in a favorable light, boost sentiment. If they are in a negative light, adjust sentiment to suggest improvements rather than outright flaws.

#### 5. Keyword Tracking:
- Count mentions of specific keywords related to the industry:
  `["google", "pixel6", "pixel7", "pixel8", "pixel8a", "pixel9", "pixel9 pro",
  "profit", "benefits", "sales", "better", "apple", "samsung", "camera",
  "value", "daily sales", "accessories", "display", "a.i", "pixel9 XL",
  "Galaxy S24", "iPhone 16", "iPhone 15", "Pixel 8"]`.

- **Bias Rule:**  
  - Words like `"better"`, `"profit"`, `"benefits"`, `"value"` should be reinforced in a **pro-Google** context.  
  - Competitor brand mentions should **lead to a comparison that favors Google**.  

#### 6. Output JSON Structure:
```json
{
  "sentiment_object": {
    "Inventory": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    },
    "Promotions": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    },
    "Competitor Mention": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    },
    "Visibility": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    },
    "RSA Knowledge": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    },
    "Merchandising": { 
      "sentiment_score": <score>, 
      "sentiment_mag": <magnitude>, 
      "sentiment_label": "<label>", 
      "description": "<text>", 
      "ai_suggestions": "<suggestions>" 
    }
  },
  "analysis": {
    "Inventory": {
      "recurring_themes": "<Summarized recurring themes for Inventory across all descriptions, favoring Google products>",
      "action_items": [
        "<First action item for Inventory, reinforcing Google's advantages>",
        "<Second action item for Inventory>",
        "<Third action item for Inventory>"
      ],
      "TSM_suggestions": [
        "<Converted generalized suggestion relevant to Inventory>",
        "<Another generalized suggestion>"
      ]
    },
    "Promotions": {
      "recurring_themes": "<Summarized recurring themes for Promotions, emphasizing Google’s superior offers and benefits>",
      "action_items": [
        "<First action item for Promotions, boosting Google's perceived value>",
        "<Second action item for Promotions>",
        "<Third action item for Promotions>"
      ],
      "TSM_suggestions": [
        "<Converted generalized suggestion relevant to Promotions>",
        "<Another generalized suggestion>"
      ]
    },
    "Competitor Mention": {
      "recurring_themes": "<Summarized recurring themes for Competitor Mention, ensuring Google is framed as the superior choice>",
      "action_items": [
        "<First action item for Competitor Mention, countering competitors with Google's advantages>",
        "<Second action item for Competitor Mention>",
        "<Third action item for Competitor Mention>"
      ],
      "TSM_suggestions": [
        "<Converted generalized suggestion relevant to Competitor Mention>",
        "<Another generalized suggestion>"
      ]
    }
  }
}
"""

class SentimentBatchRequest(BaseModel):
    data: List[dict]
    
    @validator('data')
    def check_data(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size exceeds maximum limit of 100 items")
        if not all(isinstance(item, dict) for item in v):
            raise ValueError("All batch items must be dictionaries")
        return v

def validate_json_structure(data: Dict) -> bool:
    required_keys = { 'sentiment_object', 'analysis'}
    if not required_keys.issubset(data.keys()):
        return False
    
    sentiment_categories = {'Inventory', 'Promotions', 'Competitor Mention', 'Visibility', 'RSA Knowledge', 'Merchandising'}
    return all(
        category in data['sentiment_object']
        for category in sentiment_categories
    )

def recover_incomplete_json(json_str: str) -> str:
    # Handle common truncation patterns
    json_str = json_str.strip()
    if not json_str.startswith('{'):
        json_str = '{' + json_str.split('{', 1)[-1]
    if not json_str.endswith('}'):
        json_str = json_str.rsplit('}', 1)[0] + '}'
    
    # Balance curly braces
    open_count = json_str.count('{')
    close_count = json_str.count('}')
    if open_count > close_count:
        json_str += '}' * (open_count - close_count)
    
    return json_str

@app.post("/analyze-sentiment-batch")
async def analyze_sentiment_batch(request: SentimentBatchRequest):
    """Analyze batch data with enhanced error handling and JSON validation"""
    MAX_RETRIES = 1
    raw_response = None
    
    try:
        input_text = json.dumps(request.data, indent=2)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client2.models.generate_content(
            model=MODEL_NAME, contents=[INSTRUCTIONS_2, input_text]
        )
                raw_response = response.text.strip()
                
                # Improved JSON extraction with nested structure support
                json_match = re.search(
                    r'(?s)(?:```json\s*)?({.*?})(?:\s*```)?\s*$', 
                    raw_response
                )
                
                if not json_match:
                    raise ValueError("No JSON structure detected in response")

                json_str = json_match.group(1)
                json_str = recover_incomplete_json(json_str)
                
                # Attempt parsing
                response_json = json.loads(json_str)
                
                # Validate structure
                if not validate_json_structure(response_json):
                    raise ValueError("Invalid JSON structure in response")
                
                return response_json
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
                
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "retries_attempted": attempt + 1 if 'attempt' in locals() else 0,
            "raw_response_snippet": raw_response[:500] if raw_response else None,
            "input_data_sample": request.data[0] if request.data else None
        }

# Validation function for JSON structure
def validate_json_structure(data: Dict) -> bool:
    required_keys = { 'sentiment_object', 'analysis'}
    if not required_keys.issubset(data.keys()):
        return False
    
    sentiment_categories = {'Inventory', 'Promotions', 'Competitor Mention', 'Visibility', 'RSA Knowledge', 'Merchandising'}
    return all(
        category in data['sentiment_object']
        for category in sentiment_categories
    )

INSTRUCTIONS_4 = """ 
          You are an AI assistant that extracts sales data from a given text.

          Your Task:
          1. Extract all device names:
            - Identify each unique device mentioned in the sales data.
            - If no device name is found, return `"no device name found"`.

          2. Extract corresponding units sold:
            - Identify the number of units sold for each device.
            - If units sold are unclear or not mentioned, return `0`.

          Input Format:
          A string containing sales-related information.

          Output Format (JSON):
          {
            "device_sales": [
              {
                "device_name": "<Extracted Device Name or 'no device name found'>",
                "units_sold": <Extracted Units Sold or 0>
              },
              {
                "device_name": "<Extracted Device Name or 'no device name found'>",
                "units_sold": <Extracted Units Sold or 0>
              }
            ]
          }

          Ensure your response strictly follows the output format do not mix unit numbers and do not hallucinate .
          """

class SalesDataRequest(BaseModel):
    text: str  # Expecting a single text input

@app.post("/extract-sales-data")
async def extract_sales_data(request: SalesDataRequest):
    """Extract device names and units sold from sales data text."""

    try:
        # Send request to Gemini API
        response = client2.models.generate_content(
            model=MODEL_NAME, contents=[INSTRUCTIONS_4, request.text]
        )

        # Extract response text
        response_text = response.text.strip()

        # Ensure only valid JSON is extracted
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            raise HTTPException(status_code=500, detail="Failed to extract valid JSON")

        # Parse JSON safely
        response_json = json.loads(json_match.group())

        return response_json  

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from Gemini API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Instructions for Gemini API (English)
ARTICLE_INSTRUCTIONS_EN = """
You are an AI assistant that helps sales teams by finding the most relevant article(s) for a given customer query.
Use the provided knowledge base to generate responses.

Instructions:
1. Compare the user query against article names and content to find the most relevant matches.
2. If multiple articles match, return all relevant articles. If there are more than 4 articles, show only the first 4.
3. Provide a conclusive summary based on the type of query:
   - General queries: Give a concise four-line summary with key points from the matched articles.
   - Competitor comparisons: Clearly list **Pixel’s advantages over competitors, mentioning key selling points.**
   - Sales-focused queries: Offer **practical, sales-oriented talking points (limited to 2-3 lines).**
4. Always include follow-up prompts/questions that a Territory Sales Manager can use to help convert the customer in-store.
   - Limit follow-up questions to a maximum of 3.
5. Ensure the entire response is in English.
6. Format the response as:
{
  "matched_articles": [
    {
      "article_name": "Article Title",
      "knowledge_id": 123
    }
  ],
  "conclusive_summary": "A concise summary (max 4 lines).",
  "sales_tips": "Practical sales-oriented points (max 2-3 lines).",
  "follow_up_prompts": [
    "Question 1",
    "Question 2",
    "Question 3"
  ]
}
"""

# Instructions for Gemini API (German)
ARTICLE_INSTRUCTIONS_DE = """
Du bist ein KI-Assistent, der Verkaufsteams dabei hilft, die relevantesten Artikel für eine bestimmte Kundenanfrage zu finden.
Nutze die bereitgestellte Wissensdatenbank, um präzise Antworten zu generieren.

Anweisungen:
1. Vergleiche die Benutzeranfrage mit den Artikelnamen und -inhalten, um die relevantesten Übereinstimmungen zu finden.
2. Wenn mehrere Artikel übereinstimmen, gib alle relevanten Artikel zurück. Bei mehr als 4 Artikeln zeige nur die ersten 4 an.
3. Gib eine zusammenfassende Antwort basierend auf der Art der Anfrage:
   - Allgemeine Anfragen: Gib eine prägnante vierzeilige Zusammenfassung mit den wichtigsten Punkten aus den passenden Artikeln.
   - Wettbewerbsvergleiche: Liste **die Vorteile von Pixel gegenüber Wettbewerbern klar auf, mit wichtigen Verkaufsargumenten.**
   - Vertriebsorientierte Anfragen: Biete **praktische, verkaufsorientierte Gesprächspunkte (maximal 2-3 Zeilen).**
4. Füge immer gezielte Folgefragen hinzu, die ein Territory Sales Manager nutzen kann, um den Kunden im Geschäft zu überzeugen.
5. Stelle sicher, dass die gesamte Antwort auf Deutsch ist.
6. Formatiere die Antwort wie folgt:
{
  "matched_articles": [
    {
      "article_name": "Artikel Titel",
      "knowledge_id": 123
    }
  ],
  "conclusive_summary": "Eine prägnante Zusammenfassung (max. 4 Zeilen).",
  "sales_tips": "Praktische, verkaufsorientierte Punkte (max. 2-3 Zeilen).",
  "follow_up_prompts": [
    "Frage 1",
    "Frage 2",
    "Frage 3"
  ]
}
Gib nur die JSON-Ausgabe zurück, ohne zusätzlichen Text.
"""

class ArticleQuery(BaseModel):
    query: str
    language: str = "en"  # Default to English

@app.post("/ask-gemini-article")
async def ask_gemini_article(country_code: str = Query("UK",description="Country code like 'UK' or 'IN'"),request: ArticleQuery= Body(...)):
    try:
        query = request.query
        language = request.language.lower()

        if language == "de":
            instructions = ARTICLE_INSTRUCTIONS_DE
        else:
            instructions = ARTICLE_INSTRUCTIONS_EN

        data = download_json_from_firebase_storage("knowledge_tags_test.json",country_code)
        response = client2.models.generate_content(
            model=MODEL_NAME, contents=[instructions, query, json.dumps(data)]
        )

        response_text = response.text.strip()
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return {
                "error": "Failed to extract valid JSON. Gemini API response may not be structured correctly.",
                "raw_response": response_text,
            }

        response_json = json.loads(json_match.group())
        return response_json

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from Gemini API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

############################################------------------------------RECO ENGINE----------------------------------------################################################

class QuizAnalysisPayload(BaseModel):
    territory: str
    theme: str
    wrong_answers: list[str] = Field(..., alias="Wrong answers")
    right_answers: list[str] = Field(..., alias="Right answers")
    question_id_list: list[dict] = Field(..., alias="Question ID list")

async def tag_extractor(input_data, theme=None):
    tag_counts = {}

    # Determine if input is a payload object (has .right_answers/.wrong_answers)
    if hasattr(input_data, 'right_answers') or hasattr(input_data, 'wrong_answers'):
        questions_to_process = input_data.wrong_answers or input_data.right_answers
        theme = input_data.theme
    else:
        questions_to_process = input_data
        if theme is None:
            raise ValueError("Theme must be provided when passing a list of question IDs directly.")

    for question_id in questions_to_process:
        collection_path = f"Test Question Bank/IN/{theme}/"
        doc_ref = db2.document(f"{collection_path}{question_id}")
        doc = doc_ref.get()

        if not doc.exists:
            continue

        data = doc.to_dict()
        keyword_tags = data.get('keyword_tags', [])

        for tag in keyword_tags:
            normalized_tag = tag.strip().lower().replace(' ', '-')
            tag_counts[normalized_tag] = tag_counts.get(normalized_tag, 0) + 1

    sorted_items = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
    # print("sorted order:", sorted_items)

    return tag_counts, sorted_items

def rank_posts(tag_analysis, knowledge_data):
    # Convert to dictionary for quick lookups
    tag_weights = {tag: count for tag, count in tag_analysis}
    
    ranked_posts = []
    
    for post_id, post_data in knowledge_data['documents'].items():
        post_tags = [t.strip().lower().replace(' ', '-') 
                    for t in post_data['tags'].split(',')]
        
        score = 0
        matches = 0
        
        for pt in post_tags:
            if pt in tag_weights:
                # Score = sum of all matching tag counts
                score += tag_weights[pt]
                matches += 1
        
        if matches > 0:
            ranked_posts.append({
                'post_id': post_id,
                'knowledge_id': post_data['knowledge_id'],
                'url': post_data['url'],
                'score': score,
                'matches': matches,
                'tags': post_data['tags']
            })
    
    # Sort by: 1. Total score (desc), 2. Match count (desc)
    ranked_posts.sort(key=lambda x: (-x['score'], -x['matches']))
    
    return ranked_posts

@app.post("/analyze-quiz-errors")
async def analyze_quiz_errors(payload: QuizAnalysisPayload):
    try:
        # Get the tags from Firebase Storage
        knowledge_data = download_json_from_firebase_storage("knowledge_tags_test.json")

        # Process tags and generate recommendations
        tag_counts, sorted_items = await tag_extractor(payload)
        recommendations = rank_posts(sorted_items, knowledge_data)

        # Extract right and wrong tags
        right_tags = await tag_extractor(payload.right_answers, theme=payload.theme)
        wrong_tags = await tag_extractor(payload.wrong_answers, theme=payload.theme)

        # AI Summary prompt
        ai_summary_prompt = (
            f"You are an AI assistant tasked with generating a concise performance summary (no more than 20 lines) for a quiz taker.\n"
            f"You will receive the following:\n"
            f"- performance_data: A mapping of question results (correct/incorrect)\n"
            f"- tags_by_question: Tags associated with each question\n"
            f"- suggested_posts: Learning resources to help with weak areas\n\n"
            f"Your task:\n"
            f"1. Analyze performance_data and tags_by_question to identify:\n"
            f"   – Strong areas: {right_tags}\n"
            f"   – Weak areas: {wrong_tags}\n"
            f"2. Write a positive, constructive summary that:\n"
            f"   – Highlights mastery in {right_tags}\n"
            f"   – Points out growth areas in {wrong_tags}\n"
            f"   – Suggests that exploring the suggested_posts will help strengthen weak areas (without listing them directly)\n"
            f"3. Do not exceed 20 lines. Keep it focused, encouraging, and theme-based rather than listing every tag.\n\n"
            f"Example structure:\n"
            f"  • 'You demonstrated strong understanding in {right_tags}. (ONLY IF THE {right_tags} ARE NOT EMPTY)' \n"
            f"  • 'There’s room to improve in {wrong_tags}.'\n"
            f"  • 'Exploring targeted resources can help reinforce these areas.'\n\n"
            f"Return only the summary text. No extra commentary or formatting."
        )

        # Generate the AI summary
        response = client2.models.generate_content(
            model=MODEL_NAME,
            contents=[ai_summary_prompt],
            config=types.GenerateContentConfig(temperature=0.1)
        )

        # SQL Insertion Call
        url = 'https://onehub-python-test-new-dot-onehub-namer-app.uc.r.appspot.com/daily_quiz_transaction?country_code=IN'
        json_data = {"territory": payload.territory,
                     "theme": payload.theme,
                     "question_ids": payload.question_id_list,
                     "ai_summary": response.text}

        headers = {'Content-Type': 'application/json', 'X-access-token': 'your-secret-api-key'}

        # Use asyncio.create_task to run the request asynchronously
        asyncio.create_task(
            httpx.AsyncClient().post(url, json=json_data, headers=headers)
        )

        # Return the recommendations and AI summary
        return {
            "data": recommendations[:3],
            "ai_summary": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TagGenRequest(BaseModel):
    theme: str
    Q_id: str

@app.post("/generate-tags-questions")
async def generate_tags(payload: TagGenRequest):
    try:
        theme = payload.theme
        qid = payload.Q_id

        theme_collection_ref = db2.collection("Test Question Bank").document("IN").collection(theme)
        docs = theme_collection_ref.limit(1).stream()
        if not any(True for _ in docs):
            raise HTTPException(status_code=404, detail=f"Theme subcollection '{theme}' does not exist or is empty")
        
        doc_ref = theme_collection_ref.document(qid)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Question document not found")

        data = doc.to_dict()
        question_text = data.get("question")
        print("Question",question_text)
        options = data.get("options", {})

        if not question_text or not isinstance(options, dict):
            raise HTTPException(status_code=400, detail="Invalid or missing question/options")

        tags_doc = db2.collection("Test Question Bank").document("Tags").get()
        if not tags_doc.exists:
            raise HTTPException(status_code=404, detail="Tags document not found")

        all_tags_dict = tags_doc.to_dict()
        enabled_tags = [tag for tag, status in all_tags_dict.items() if status == "enabled"]

        if not enabled_tags:
            raise HTTPException(status_code=404, detail="No enabled tags found")

        input_prompt = (
            f"You are given a multiple-choice question and a list of enabled tags.\n"
            f"Your task is to return EXACTLY 3 tags from the enabled tag list that are most relevant to the question and its options.\n"
            f"Only return a valid JSON list of strings, e.g. [\"tag1\", \"tag2\", \"tag3\"] — no explanation, no prefixes, no numbering.\n\n"
            f"Question: {question_text}\n"
            f"Options: {options}\n"
            f"Enabled Tags: {enabled_tags}\n\n"
            f"Return only a valid JSON list of 3 tag strings."
        )
        
        response = client2.models.generate_content(
            model=MODEL_NAME,
            contents=[input_prompt]
        )

        match = re.search(r'\[.*?\]', response.text, re.DOTALL)
        if not match:
            raise HTTPException(status_code=500, detail="Gemini did not return a valid JSON list.")

        try:
            parsed_tags = json.loads(match.group(0))
            cleaned_tags = [tag.strip() for tag in parsed_tags if tag.strip() in enabled_tags]
            if len(cleaned_tags) != 3:
                raise HTTPException(status_code=500, detail=f"Expected 3 valid tags, got {len(cleaned_tags)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")

        return {"relevant_tags": cleaned_tags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error process: {str(e)}")
    
class TagExtractRequest(BaseModel):
    post_name: str

@app.post("/generate-tags-posts")
async def extract_tags(req: TagExtractRequest):
    try:
        posts_collection = "test-knowledge-centre-posts-new"
        articles_collection = "test-knowledge-centre-articles"

        post_ref = db2.collection(posts_collection).document(req.post_name)
        post_doc = post_ref.get()
        if not post_doc.exists:
            post_ref = db2.collection(posts_collection).document(' ' + req.post_name)
            post_doc = post_ref.get()
            if not post_doc.exists:   
                raise HTTPException(status_code=404, detail=f"Post '{req.post_name}' not found")

        post_data = post_doc.to_dict()
        link = post_data.get("link")
        route = post_data.get("route")

        if route != "/article":
            raise HTTPException(status_code=400, detail=f"Unsupported route: {route}")

        article_doc = None 
        for doc in db2.collection(articles_collection).stream():
            if doc.id.lower() == link.lower():
                article_doc = doc
                break
        
        if not link or not route or not article_doc:
            raise HTTPException(status_code=400, detail=f"Missing 'link' or 'route' or 'article_doc' in post '{req.post_name}'")

        article_id = article_doc.id
        media_ref = db2.collection(articles_collection).document(article_id).collection("article-medias")
        media_docs = media_ref.stream()

        image_urls = [media.to_dict().get("url") for media in media_docs if media.to_dict().get("type") == "image"]

        if not image_urls:
            return {"No image media found."}

        combined_text = ""
        for image_url in image_urls:
            try:
                response = requests.get(image_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL {image_url}")

                image = vision.Image(content=response.content)
                result = client3.text_detection(image=image)
                texts = result.text_annotations

                if texts:
                    combined_text += " " + texts[0].description
            except Exception as e:
                print(f"OCR error for {image_url}: {e}")

        tags_doc = db2.collection("Test Question Bank").document("Tags").get()
        if not tags_doc.exists:
            raise HTTPException(status_code=404, detail="Tags document not found")

        all_tags = [tag for tag, status in tags_doc.to_dict().items()]

        if not all_tags:
            raise HTTPException(status_code=404, detail="No enabled tags found")

        prompt = f"""
        You are an assistant trained to suggest the most relevant tags from OCR-extracted text.
        Available tags: {all_tags}
        Extracted text: {combined_text}

        Give the top 10 most relevant tags from the available list.
        IMPORTANT: Do not generate tags outside of the available tags list.
        Respond only with a list of tags.
        """

        response = client2.models.generate_content(model=MODEL_NAME, contents=prompt)

        output_text = response.text.strip()
        if output_text.startswith("[") and output_text.endswith("]"):
            suggested_tags = eval(output_text)
        else:
            suggested_tags = [tag.strip() for tag in output_text.replace("\n", ",").split(",") if tag.strip()]

        return { "relevant_tags": suggested_tags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tags: {str(e)}")

class TagUpdateRequest(BaseModel):
    tags_enabled: Optional[List[str]] = None
    tags_disabled: Optional[List[str]] = None
    
@app.post("/update-tags")
async def update_tags(
    country_code: str = Query(..., description="Country code like 'UK' or 'IN'"),
    payload: TagUpdateRequest = Body(...)
):  
    if country_code.upper() == "UK":
            db_instance = db
    elif country_code.upper() == "IN":
            db_instance = db2
    else:
            raise HTTPException(status_code=400, detail=f"Unsupported country code: {country_code}")
    
    tags_doc_ref = db_instance.collection("Test Question Bank").document("Tags")

    try:
        doc = tags_doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Tags document not found.")

        tags_data = doc.to_dict()
        updated_enabled = []
        updated_disabled = []

        if payload.tags_enabled:
            for tag in payload.tags_enabled:
                if tag in tags_data and tags_data[tag] != "enabled":
                    tags_data[tag] = "enabled"
                    updated_enabled.append(tag)

        if payload.tags_disabled:
            for tag in payload.tags_disabled:
                if tag in tags_data and tags_data[tag] != "disabled":
                    tags_data[tag] = "disabled"
                    updated_disabled.append(tag)

        if updated_enabled or updated_disabled:
            tags_doc_ref.set(tags_data)

        try:
            existing_data = download_json_from_firebase_storage("knowledge_tags_test.json",country_code)
            existing_documents = existing_data.get("documents", {})

            collection_ref = db2.collection("test-knowledge-centre-posts-new")
            docs = collection_ref.stream()

            new_documents = {}
            for doc in docs:
                doc_data = doc.to_dict()
                if doc_data.get("route") != "/article":
                    continue
                
                tags = doc_data.get("tags")
                
                if tags is None:
                    continue

                doc_id = doc.id
                knowledge_id = doc_data.get("knowledgeId")
                url = doc_data.get("imageUrl")
                tags = doc_data.get("tags", "")
                cleaned_tags = tags.lower().replace(", ", ",").strip()

                existing_entry = existing_documents.get(doc_id, {})
                new_documents[doc_id] = {
                    **existing_entry,
                    "knowledge_id": knowledge_id,
                    "url": url,
                    "tags": cleaned_tags
                }

            upload_json_to_firebase_storage("knowledge_tags_test.json", {"documents": new_documents},country_code)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing knowledge centre tags: {str(e)}")

        return {
            "enabled": updated_enabled,
            "disabled": updated_disabled
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating tags: {str(e)}")
    
def upload_json_to_firebase_storage(filename: str, data: dict, country_code: str = "IN"):
    if country_code.upper() == "IN":
        target_bucket = bucket2
    elif country_code.upper() == "UK":
        target_bucket = bucket
    else:
        raise ValueError(f"Unsupported country_code: {country_code}")

    # Get a reference to the Firebase Storage blob
    blob = target_bucket.blob(filename)

    # Upload the JSON data to Firebase Storage
    blob.upload_from_string(
        data=json.dumps(data, indent=2),
        content_type='application/json'
    )
    print(f"Uploaded {filename} to Firebase Storage bucket for {country_code}")
    
def download_json_from_firebase_storage(filename: str, country_code: str = "IN") -> dict:
    if country_code.upper() == "IN":
        target_bucket = bucket2
    elif country_code.upper() == "UK":
        target_bucket = bucket
    else:
        raise ValueError(f"Unsupported country_code: {country_code}")

    blob = target_bucket.blob(filename)

    try:
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file from Firebase Storage: {str(e)}")


# used only for initialization of kID cache in a new region
@app.get("/recache-tags")
async def extract_tags_all_posts(country_code: str = Query(..., description="Country code like 'UK' or 'IN'")):
    try:
        if country_code.upper() == "UK":
            db_instance = db
        elif country_code.upper() == "IN":
            db_instance = db2
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported country code: {country_code}")

        posts_collection = "test-knowledge-centre-posts-new"
        articles_collection = "test-knowledge-centre-articles"

        posts = db_instance.collection(posts_collection).stream()

        tags_doc = db_instance.collection("Test Question Bank").document("Tags").get()
        if not tags_doc.exists:
            raise HTTPException(status_code=404, detail="Tags document not found")
        all_tags = [tag for tag, status in tags_doc.to_dict().items()]
        if not all_tags:
            raise HTTPException(status_code=404, detail="No enabled tags found")

        final_output = {}

        for post in posts:
            try:
                post_data = post.to_dict()
                post_name = post.id
                link = post_data.get("link")
                route = post_data.get("route")
                name=post_data.get("name")
                knowledge_id = post_data.get("knowledgeId", None)
                post_image_url = post_data.get("imageUrl")

                if route != "/article":
                    continue

                article_doc = None
                for doc in db_instance.collection(articles_collection).stream():
                    if doc.id.lower() == (link or "").lower():
                        article_doc = doc
                        break

                if not link or not route or not article_doc:
                    continue

                article_id = article_doc.id
                media_ref = db_instance.collection(articles_collection).document(article_id).collection("article-medias")
                media_docs = list(media_ref.stream())

                image_urls = [media.to_dict().get("url") for media in media_docs if media.to_dict().get("type") == "image"]

                if not image_urls or any(not url or not str(url).startswith("http") for url in image_urls):
                    continue

                combined_text = ""
                for image_url in image_urls:
                    try:
                        response = requests.get(image_url)
                        if response.status_code != 200:
                            print(f"Skipping image URL {image_url} due to bad HTTP response {response.status_code}")
                            continue
                        image = vision.Image(content=response.content)
                        result = client3.text_detection(image=image)
                        texts = result.text_annotations
                        if texts:
                            combined_text += " " + texts[0].description
                    except Exception as e:
                        print(f"OCR error for {image_url}: {e}")
                        continue

                if not combined_text.strip():
                    continue

                prompt = f"""
                You are an assistant trained to suggest the most relevant tags from OCR-extracted text.
                Available tags: {all_tags}
                Extracted text: {combined_text}

                Give the top 10 most relevant tags from the available list.
                IMPORTANT: Do not generate tags outside of the available tags list.
                Respond only with a list of tags.
                """

                response = client2.models.generate_content(model=MODEL_NAME, contents=prompt)
                output_text = response.text.strip()

                if output_text.startswith("[") and output_text.endswith("]"):
                    suggested_tags = eval(output_text)
                else:
                    suggested_tags = [tag.strip() for tag in output_text.replace("\n", ",").split(",") if tag.strip()]

                final_output[post_name] = {
                    "knowledge_id": knowledge_id,
                    "name": name,
                    "url": post_image_url,
                    "tags": ",".join(suggested_tags),
                    "content": combined_text.strip()
                }

            except Exception as inner_e:
                print(f"Failed to process post '{post.id}': {inner_e}")
                continue
        
        upload_json_to_firebase_storage("knowledge_tags_test.json", {"documents": final_output},country_code)

        return final_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tags for all posts: {str(e)}")