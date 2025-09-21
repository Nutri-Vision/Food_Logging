# Nutri-Vision Unified API

Nutri-Vision Unified API is a FastAPI-based service for nutrition analysis from text, image, and voice inputs. It provides endpoints for extracting food items and their nutritional information using NLP and external APIs.

## Features
- **Text Analysis**: Extracts food items and nutrition from text using a hybrid NLP pipeline and USDA lookup.
- **Image Analysis**: Uses LogMeal API to detect food items in images and retrieve nutrition data.
- **Voice Analysis**: (Planned) Will support nutrition extraction from voice input.
- **Unified JSON Response**: Standardized output for all input types.
- **Legacy Endpoint**: Backward compatibility for older frontends.

## Endpoints
- `GET /` — API info and available endpoints
- `GET /health` — Health check for services
- `POST /analyze/text` — Analyze nutrition from text
- `POST /analyze/image` — Analyze nutrition from image
- `POST /analyze/voice` — Analyze nutrition from voice (planned)
- `POST /analyze-text` — Legacy text analysis endpoint

## Request Examples
### Text Analysis
```json
POST /analyze/text
{
	"text": "2 eggs and a slice of bread",
	"include_usda": true
}
```
### Image Analysis
Send an image file as form-data to `/analyze/image`.

### Voice Analysis
```json
POST /analyze/voice
{
	"audio_data": "<base64-encoded-audio>",
	"audio_format": "wav",
	"language": "en"
}
```

## Response Structure
All analysis endpoints return a unified JSON structure:
```json
{
	"success": true,
	"input_type": "text|image|voice",
	"raw_input": "...",
	"items": [
		{
			"name": "Egg",
			"quantity": 2,
			"unit": "piece",
			"macros": {
				"calories": 155,
				"protein": 13,
				"carbs": 1.1,
				"fats": 11
			},
			"confidence": 0.95,
			"source": "text_nlp|logmeal_api",
			"notes": null
		}
	],
	"totals": {
		"calories": 310,
		"protein": 26,
		"carbs": 2.2,
		"fats": 22
	},
	"processing_time": 0.12,
	"warnings": [],
	"metadata": {}
}
```

## Setup & Usage
1. **Install dependencies**:
	 ```bash
	 pip install -r requirements.txt
	 ```
2. **Set environment variables**:
	 - `LOGMEAL_API_TOKEN` for image analysis
3. **Run the API**:
	 ```bash
	 python main.py
	 ```
	 Or with Uvicorn:
	 ```bash
	 uvicorn main:app --host 0.0.0.0 --port 8000
	 ```

## Configuration
- CORS is enabled for all origins (adjust for production).
- USDA lookup can be toggled in text analysis requests.
- LogMeal API token must be set for image analysis.

## Folder Structure
- `main.py` — API entry point
- `nlp/` — NLP extraction logic
- `usda/` — USDA nutrition lookup
- `models/food_ner/` — spaCy model files

## License
See `LICENSE` file for details.

---
For more details, see code comments in `main.py`.