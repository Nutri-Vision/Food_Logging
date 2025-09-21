# Nutri-Vision Unified API v2.1.0

Nutri-Vision Unified API is an enhanced FastAPI-based service for comprehensive nutrition analysis from text, image, and voice inputs. It integrates LogMeal API for advanced food recognition and USDA FoodData Central for authoritative nutrition data.

## 🆕 What's New in v2.1.0

- **🔍 Enhanced LogMeal Integration**: Complete image segmentation and food recognition
- **🥗 USDA FoodData Central Integration**: Authoritative nutrition data from official government database
- **🧠 Improved NLP Pipeline**: Better food extraction with quantity detection
- **📊 Extended Nutrition Data**: Now includes fiber, sugar, and detailed micronutrients
- **🔄 Seamless API Orchestration**: Combines multiple services for comprehensive analysis
- **🧪 Testing Endpoints**: Built-in API testing and validation tools
- **⚡ Enhanced Error Handling**: Robust fallbacks and meaningful error messages

## Features

### 🔤 **Text Analysis (Enhanced)**
- Extracts food items and quantities using hybrid NLP pipeline
- USDA nutrition lookup with confidence scoring
- Supports complex meal descriptions with multiple ingredients
- Quantity detection (e.g., "2 apples", "1 cup rice")

### 📸 **Image Analysis (Fully Implemented)**
- **LogMeal API Integration**: Advanced food segmentation and recognition
- **Multi-item Detection**: Identifies multiple foods in single image
- **USDA Nutrition Lookup**: Gets authoritative nutrition data for detected foods
- **Confidence Scoring**: Provides recognition confidence for each item

### 🎤 **Voice Analysis**
- Planned feature for speech-to-text nutrition analysis
- Will support multiple languages and audio formats

### 📋 **Unified Response Format**
- Standardized JSON output across all input types
- Comprehensive metadata and processing information
- Detailed nutrition breakdown with macros and micronutrients

### 🔧 **API Management**
- Health monitoring and service status
- Configuration validation
- Built-in testing endpoints
- Legacy endpoint compatibility

## API Endpoints

### **Core Analysis Endpoints**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and available endpoints |
| `GET` | `/health` | Health check with service status |
| `GET` | `/config` | Configuration and API key status |
| `POST` | `/analyze/text` | Enhanced text analysis with USDA integration |
| `POST` | `/analyze/image` | Image analysis with LogMeal + USDA |
| `POST` | `/analyze/voice` | Voice analysis (planned) |
| `POST` | `/analyze-text` | Legacy text analysis (backward compatibility) |

### **Testing & Utility Endpoints**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/test/usda-search?query={food}` | Test USDA food search |
| `GET` | `/test/usda-nutrition/{food_id}` | Test USDA nutrition lookup |

## Request Examples

### **Enhanced Text Analysis**
```json
POST /analyze/text
Content-Type: application/json

{
  "text": "I had 2 medium apples, 150g grilled chicken breast, and 1 cup of brown rice for lunch",
  "include_usda": true
}
```

### **Image Analysis with LogMeal + USDA**
```bash
POST /analyze/image
Content-Type: multipart/form-data

image: [food_image.jpg]
include_nutrition: true
```

### **Legacy Text Analysis (Backward Compatible)**
```json
POST /analyze-text
Content-Type: application/json

{
  "description": "2 eggs and a slice of bread"
}
```

### **USDA Testing**
```bash
GET /test/usda-search?query=chicken%20breast
GET /test/usda-nutrition/171077
```

## Enhanced Response Structure

All analysis endpoints return a comprehensive unified JSON structure:

```json
{
  "success": true,
  "input_type": "text|image|voice",
  "raw_input": "I had 2 medium apples...",
  "items": [
    {
      "name": "Apple",
      "quantity": 2.0,
      "unit": "medium",
      "macros": {
        "calories": 190.0,
        "protein": 1.0,
        "carbs": 50.0,
        "fats": 0.6,
        "fiber": 8.0,
        "sugar": 38.0
      },
      "confidence": 0.92,
      "source": "text_usda",
      "notes": null,
      "usda_food_id": "171688",
      "logmeal_food_id": null
    },
    {
      "name": "Chicken breast",
      "quantity": 150.0,
      "unit": "grams",
      "macros": {
        "calories": 248.0,
        "protein": 46.5,
        "carbs": 0.0,
        "fats": 5.4,
        "fiber": 0.0,
        "sugar": 0.0
      },
      "confidence": 0.88,
      "source": "text_usda",
      "notes": null,
      "usda_food_id": "171077",
      "logmeal_food_id": null
    }
  ],
  "totals": {
    "calories": 688.0,
    "protein": 52.7,
    "carbs": 78.0,
    "fats": 7.3,
    "fiber": 12.0,
    "sugar": 40.0
  },
  "processing_time": 2.34,
  "warnings": [],
  "metadata": {
    "usda_configured": true,
    "logmeal_configured": true,
    "items_with_usda": 2,
    "items_with_mock": 0,
    "usda_lookup_enabled": true
  }
}
```

## Setup & Configuration

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Environment Variables**
```bash
# Required for LogMeal image analysis
export LOGMEAL_TOKEN="your_logmeal_api_token"

# Required for USDA nutrition lookup
export USDA_API_KEY="your_usda_api_key"

# Optional: Custom port
export PORT="8000"
```

### **3. API Key Setup**

#### **LogMeal API Token**
1. Sign up at [LogMeal.es](https://logmeal.es)
2. Get your API token from the dashboard
3. Set `LOGMEAL_TOKEN` environment variable

#### **USDA FoodData Central API Key**
1. Visit [USDA FDC API Guide](https://fdc.nal.usda.gov/api-guide.html)
2. Sign up for a free API key
3. Set `USDA_API_KEY` environment variable

### **4. Run the API**

**Development:**
```bash
python main.py
```

**Production with Uvicorn:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**With Docker:**
```bash
docker build -t nutri-vision-api .
docker run -p 8000:8000 -e LOGMEAL_TOKEN="your_token" -e USDA_API_KEY="your_key" nutri-vision-api
```

## Service Integration Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│ Nutri-Vision │───▶│   Response  │
│ (Text/Image)│    │     API      │    │   (JSON)    │
└─────────────┘    └──────┬───────┘    └─────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌─────────────┐  ┌──────────┐
    │   NLP    │   │  LogMeal    │  │   USDA   │
    │ Pipeline │   │     API     │  │   API    │
    └──────────┘   └─────────────┘  └──────────┘
```

## Configuration Status Checking

Check your API configuration:
```bash
curl -X GET "https://your-api-url.com/config"
```

Response shows service availability:
```json
{
  "version": "2.1.0-enhanced",
  "services": {
    "nlp_module": {"available": true, "status": "available"},
    "logmeal_api": {"configured": true, "status": "ready"},
    "usda_api": {"configured": true, "status": "ready"}
  },
  "features": {
    "text_analysis": true,
    "image_analysis": true,
    "voice_analysis": false,
    "usda_integration": true,
    "logmeal_integration": true
  }
}
```

## Testing Your Deployment

### **Quick Health Check**
```bash
curl -X GET "https://your-api-url.com/health"
```

### **Test Text Analysis**
```bash
curl -X POST "https://your-api-url.com/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "apple and banana", "include_usda": true}'
```

### **Test USDA Integration**
```bash
curl -X GET "https://your-api-url.com/test/usda-search?query=apple"
```

### **Full Testing Suite**
See the comprehensive testing guide in `/docs/testing.md` for detailed testing instructions.

## Enhanced Features

### **🎯 Improved Accuracy**
- USDA government database for authoritative nutrition data
- LogMeal computer vision for precise food recognition
- Confidence scoring for all detected items

### **🔄 Seamless Fallbacks**
- Mock nutrition data when APIs are unavailable
- Graceful degradation with meaningful error messages
- Service health monitoring and alerts

### **📊 Comprehensive Data**
- Extended nutrition profile (calories, protein, carbs, fats, fiber, sugar)
- Quantity detection and scaling
- Source attribution and confidence metrics

### **🧪 Developer Tools**
- Built-in API testing endpoints
- Configuration validation
- Service status monitoring

## Folder Structure

```
nutri-vision-api/
├── main.py                 # Enhanced API entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── nlp/                   # NLP extraction logic
│   └── hybrid_extractor.py
├── usda/                  # USDA API integration
│   └── fooddata_api.py
├── models/                # ML models
│   └── food_ner/
├── tests/                 # Test suite
├── docs/                  # Documentation
└── deploy/                # Deployment configs
```

## Production Deployment

### **Environment Setup**
- Set production API keys
- Configure CORS for your domain
- Enable request rate limiting
- Set up monitoring and logging

### **Performance Considerations**
- USDA API: 1000 requests/hour (free tier)
- LogMeal API: Check your plan limits
- Image processing: ~2-5 seconds per image
- Text processing: ~0.5-2 seconds per request

### **Security**
- API keys stored as environment variables
- Input validation and sanitization
- Rate limiting recommended for production
- HTTPS required for image uploads

## Troubleshooting

### **Common Issues**

**1. API Keys Not Working**
- Check `/config` endpoint for service status
- Verify environment variables are set
- Check API key validity with test endpoints

**2. Image Analysis Failing**
- Ensure LogMeal token is configured
- Check image format (JPEG, PNG supported)
- Verify image size (< 10MB recommended)

**3. Nutrition Data Missing**
- USDA API might be down or rate-limited
- Food item not found in USDA database
- Fallback to mock nutrition data

**4. Slow Response Times**
- Multiple API calls for complex meals
- Network latency to external services
- Consider caching for repeated requests

## Support & Contributing

- **Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive API docs at `/docs`
- **Testing**: Built-in test endpoints for validation
- **Updates**: Check version compatibility

## License

See `LICENSE` file for details.

---

**🚀 Ready for comprehensive nutrition analysis with industry-leading accuracy!**

For detailed testing instructions and examples, see the complete testing guide included in your deployment.