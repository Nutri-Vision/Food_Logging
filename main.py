import httpx
from io import BytesIO
from PIL import Image as PILImage
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any
import logging
import os
import time
import asyncio
from urllib.parse import quote

# Import your existing NLP components with error handling
try:
    from nlp.hybrid_extractor import hybrid_extract
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"NLP import failed: {e}")
    NLP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nutri-Vision Unified API",
    description="Enhanced nutrition analysis API with LogMeal and USDA integration",
    version="2.1.0-enhanced"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# LogMeal API Configuration
LOGMEAL_API_TOKEN = os.getenv('LOGMEAL_TOKEN', '01837cb41c6d9314170d28d15b3103a9e89d0987')
LOGMEAL_BASE_URL = 'https://api.logmeal.es'

# USDA API Configuration  
USDA_API_KEY = os.getenv('USDA_API_KEY', 'ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk')
USDA_BASE_URL = 'https://api.nal.usda.gov/fdc/v1'

# =============================================================================
# MODELS
# =============================================================================

class MacroInfo(BaseModel):
    """Standardized macronutrient information"""
    calories: float = Field(default=0.0, ge=0, description="Calories in kcal")
    protein: float = Field(default=0.0, ge=0, description="Protein in grams")
    carbs: float = Field(default=0.0, ge=0, description="Carbohydrates in grams")
    fats: float = Field(default=0.0, ge=0, description="Fats in grams")
    fiber: Optional[float] = Field(default=None, ge=0, description="Fiber in grams")
    sugar: Optional[float] = Field(default=None, ge=0, description="Sugar in grams")

class FoodItem(BaseModel):
    """Individual food item with nutrition info"""
    name: str = Field(..., description="Food item name")
    quantity: float = Field(default=1.0, gt=0, description="Quantity amount")
    unit: str = Field(default="serving", description="Measurement unit")
    macros: MacroInfo
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Recognition confidence (0-1)")
    source: str = Field(default="api", description="Data source")
    notes: Optional[str] = Field(None, description="Additional notes or warnings")
    usda_food_id: Optional[str] = Field(None, description="USDA Food ID if found")
    logmeal_food_id: Optional[str] = Field(None, description="LogMeal Food ID if found")

class NutritionAnalysis(BaseModel):
    """Unified nutrition analysis response"""
    success: bool = Field(..., description="Analysis success status")
    input_type: str = Field(..., description="Input type: text, image, or voice")
    raw_input: str = Field(..., description="Original input description")
    items: List[FoodItem] = Field(default=[], description="Detected food items")
    totals: MacroInfo = Field(..., description="Total macronutrients")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    warnings: List[str] = Field(default=[], description="Processing warnings")
    metadata: dict = Field(default={}, description="Additional metadata")

class TextAnalysisRequest(BaseModel):
    """Text-based nutrition analysis request"""
    text: str = Field(..., min_length=1, max_length=2000, description="Food description text")
    include_usda: bool = Field(True, description="Whether to include USDA nutrition lookup")

# =============================================================================
# USDA API INTEGRATION
# =============================================================================

async def search_usda_food(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for food items in USDA FoodData Central"""
    try:
        if USDA_API_KEY == 'your_usda_api_key_here':
            logger.warning("USDA API key not configured")
            return []
            
        params = {
            'api_key': USDA_API_KEY,
            'query': query,
            'pageSize': limit,
            'dataType': ['Foundation', 'SR Legacy', 'Survey (FNDDS)']
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{USDA_BASE_URL}/foods/search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('foods', [])
            else:
                logger.error(f"USDA search failed: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"USDA search error: {str(e)}")
        return []

async def get_usda_nutrition(food_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed nutrition information from USDA for a specific food ID"""
    try:
        if USDA_API_KEY == 'your_usda_api_key_here':
            return None
            
        params = {'api_key': USDA_API_KEY}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{USDA_BASE_URL}/food/{food_id}", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"USDA nutrition lookup failed: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"USDA nutrition error: {str(e)}")
        return None

def extract_usda_macros(usda_food: Dict[str, Any]) -> MacroInfo:
    """Extract macronutrients from USDA food data"""
    nutrients = {}
    
    # Map USDA nutrient names to our standard names
    nutrient_map = {
        'Energy': 'calories',
        'Protein': 'protein', 
        'Carbohydrate, by difference': 'carbs',
        'Total lipid (fat)': 'fats',
        'Fiber, total dietary': 'fiber',
        'Sugars, total including NLEA': 'sugar'
    }
    
    for nutrient in usda_food.get('foodNutrients', []):
        nutrient_name = nutrient.get('nutrient', {}).get('name', '')
        
        if nutrient_name in nutrient_map:
            value = nutrient.get('amount', 0)
            # Convert kcal to calories for energy
            if nutrient_name == 'Energy':
                unit = nutrient.get('nutrient', {}).get('unitName', '').upper()
                if unit == 'KCAL':
                    nutrients[nutrient_map[nutrient_name]] = float(value)
                elif unit == 'KJ':
                    # Convert kJ to kcal (1 kcal = 4.184 kJ)
                    nutrients[nutrient_map[nutrient_name]] = float(value) / 4.184
            else:
                nutrients[nutrient_map[nutrient_name]] = float(value)
    
    return MacroInfo(
        calories=nutrients.get('calories', 0.0),
        protein=nutrients.get('protein', 0.0),
        carbs=nutrients.get('carbs', 0.0),
        fats=nutrients.get('fats', 0.0),
        fiber=nutrients.get('fiber'),
        sugar=nutrients.get('sugar')
    )

# =============================================================================
# LOGMEAL API INTEGRATION
# =============================================================================

async def logmeal_detect_food(image_data: bytes, filename: str) -> Dict[str, Any]:
    """Use LogMeal API to detect and segment food in image"""
    try:
        if LOGMEAL_API_TOKEN == 'your_logmeal_token_here':
            logger.warning("LogMeal API token not configured")
            return {"error": "LogMeal API token not configured"}
            
        headers = {
            "Authorization": f"Bearer {LOGMEAL_API_TOKEN}",
        }
        
        files = {
            "image": (filename, image_data, "image/jpeg")
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Use LogMeal's complete segmentation endpoint
            response = await client.post(
                f"{LOGMEAL_BASE_URL}/v2/image/segmentation/complete",
                headers=headers,
                files=files
            )
            
            if response.status_code == 401:
                return {"error": "LogMeal API authentication failed"}
            elif response.status_code != 200:
                return {"error": f"LogMeal API error: HTTP {response.status_code}"}
            
            return response.json()
            
    except Exception as e:
        logger.error(f"LogMeal detection error: {str(e)}")
        return {"error": str(e)}

def parse_logmeal_results(logmeal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse LogMeal API response to extract food items"""
    detected_items = []
    
    try:
        # Handle segmentation results format
        if "segmentation_results" in logmeal_data:
            for segment in logmeal_data["segmentation_results"]:
                recognition_results = segment.get("recognition_results", [])
                
                if recognition_results:
                    # Get the best recognition result
                    best_result = recognition_results[0]
                    
                    detected_items.append({
                        "name": best_result.get("name", "Unknown Food"),
                        "confidence": best_result.get("prob", 0.0),
                        "food_id": best_result.get("food_id"),
                        "category": best_result.get("food_type", "unknown"),
                        "source": "logmeal"
                    })
        
        # Handle direct recognition results format  
        elif "recognition_results" in logmeal_data:
            for result in logmeal_data["recognition_results"][:5]:  # Top 5 results
                if result.get("prob", 0) > 0.1:  # Filter low confidence
                    detected_items.append({
                        "name": result.get("name", "Unknown Food"),
                        "confidence": result.get("prob", 0.0),
                        "food_id": result.get("food_id"),
                        "category": result.get("food_type", "unknown"), 
                        "source": "logmeal"
                    })
                    
        logger.info(f"Parsed {len(detected_items)} items from LogMeal response")
        return detected_items
        
    except Exception as e:
        logger.error(f"Error parsing LogMeal results: {str(e)}")
        return []

# =============================================================================
# ENHANCED FOOD PROCESSING
# =============================================================================

async def get_nutrition_for_ingredient(ingredient_data: Dict[str, Any]) -> FoodItem:
    """Get comprehensive nutrition info for a detected ingredient"""
    
    ingredient_name = ingredient_data.get("name", "Unknown Food")
    confidence = ingredient_data.get("confidence", 0.0)
    logmeal_food_id = ingredient_data.get("food_id")
    
    logger.info(f"Processing nutrition for: {ingredient_name}")
    
    # Step 1: Try to get nutrition from USDA
    usda_macros = None
    usda_food_id = None
    notes = []
    
    if USDA_API_KEY != 'your_usda_api_key_here':
        try:
            # Search USDA for this ingredient
            usda_results = await search_usda_food(ingredient_name, limit=3)
            
            if usda_results:
                # Get nutrition from the best match
                best_match = usda_results[0]
                usda_food_id = str(best_match.get('fdcId', ''))
                
                # Get detailed nutrition info
                usda_detail = await get_usda_nutrition(usda_food_id)
                if usda_detail:
                    usda_macros = extract_usda_macros(usda_detail)
                    logger.info(f"Got USDA nutrition for {ingredient_name}: {usda_macros.calories} cal")
                else:
                    notes.append("USDA nutrition lookup failed")
            else:
                notes.append("No USDA match found")
                
        except Exception as e:
            logger.error(f"USDA lookup error for {ingredient_name}: {str(e)}")
            notes.append(f"USDA error: {str(e)}")
    
    # Step 2: Fallback to mock nutrition if USDA failed
    if not usda_macros:
        mock_nutrition = get_mock_nutrition_by_food_name(ingredient_name)
        usda_macros = MacroInfo(**mock_nutrition)
        notes.append("Using estimated nutrition values")
        
    # Step 3: Create FoodItem
    return FoodItem(
        name=ingredient_name,
        quantity=1.0,
        unit="portion",
        macros=usda_macros,
        confidence=confidence,
        source="logmeal_usda" if usda_food_id else "logmeal_mock",
        notes="; ".join(notes) if notes else None,
        usda_food_id=usda_food_id,
        logmeal_food_id=logmeal_food_id
    )

def get_mock_nutrition_by_food_name(food_name: str) -> dict:
    """Enhanced mock nutrition database"""
    food_name_lower = food_name.lower()
    
    nutrition_db = {
        # Fruits
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fats": 0.3, "fiber": 4.0, "sugar": 19},
        "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.4, "fiber": 3.1, "sugar": 14},
        "orange": {"calories": 65, "protein": 1.3, "carbs": 16, "fats": 0.2, "fiber": 3.4, "sugar": 13},
        "strawberry": {"calories": 49, "protein": 1.0, "carbs": 12, "fats": 0.5, "fiber": 3.3, "sugar": 7},
        
        # Vegetables
        "broccoli": {"calories": 55, "protein": 4.6, "carbs": 11, "fats": 0.6, "fiber": 5.1, "sugar": 2.6},
        "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fats": 0.2, "fiber": 2.8, "sugar": 4.7},
        "tomato": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fats": 0.2, "fiber": 1.4, "sugar": 3.2},
        "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fats": 0.2, "fiber": 1.3, "sugar": 0.8},
        
        # Proteins
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6, "fiber": 0, "sugar": 0},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fats": 15, "fiber": 0, "sugar": 0},
        "fish": {"calories": 206, "protein": 22, "carbs": 0, "fats": 12, "fiber": 0, "sugar": 0},
        "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fats": 13, "fiber": 0, "sugar": 0},
        "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fats": 11, "fiber": 0, "sugar": 1.1},
        
        # Carbohydrates
        "bread": {"calories": 265, "protein": 9, "carbs": 49, "fats": 3.2, "fiber": 2.7, "sugar": 5.0},
        "rice": {"calories": 365, "protein": 7.1, "carbs": 80, "fats": 0.9, "fiber": 1.6, "sugar": 0.1},
        "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fats": 1.1, "fiber": 2.5, "sugar": 2.7},
        "potato": {"calories": 161, "protein": 4.3, "carbs": 37, "fats": 0.1, "fiber": 2.1, "sugar": 1.7},
        "oats": {"calories": 389, "protein": 16.9, "carbs": 66, "fats": 6.9, "fiber": 10.6, "sugar": 0.99},
        
        # Popular dishes
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fats": 10, "fiber": 2.3, "sugar": 3.8},
        "burger": {"calories": 295, "protein": 17, "carbs": 23, "fats": 14, "fiber": 2.0, "sugar": 4.0},
        "salad": {"calories": 65, "protein": 5, "carbs": 7, "fats": 4, "fiber": 3.0, "sugar": 4.0},
        "sandwich": {"calories": 230, "protein": 10, "carbs": 30, "fats": 8, "fiber": 3.0, "sugar": 4.0},
        "soup": {"calories": 85, "protein": 4, "carbs": 12, "fats": 2.5, "fiber": 2.0, "sugar": 3.0},
        
        # dairy & snacks
        "cheese": {"calories": 113, "protein": 7, "carbs": 1, "fats": 9, "fiber": 0, "sugar": 0.5},
        "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fats": 0.4, "fiber": 0, "sugar": 3.6},
        "nuts": {"calories": 607, "protein": 20, "carbs": 16, "fats": 54, "fiber": 8.0, "sugar": 4.0},
        "avocado": {"calories": 160, "protein": 2, "carbs": 9, "fats": 15, "fiber": 7.0, "sugar": 0.7},
    }
    
    # Try exact match first
    if food_name_lower in nutrition_db:
        return nutrition_db[food_name_lower]
    
    # Try partial matches
    for key, nutrition in nutrition_db.items():
        if key in food_name_lower or food_name_lower in key:
            return nutrition
    
    # Default for unknown foods
    return {"calories": 150, "protein": 8, "carbs": 20, "fats": 5, "fiber": 2.0, "sugar": 5.0}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_totals(items: List[FoodItem]) -> MacroInfo:
    """Calculate total macronutrients from food items"""
    totals = MacroInfo(
        calories=sum(item.macros.calories * item.quantity for item in items),
        protein=sum(item.macros.protein * item.quantity for item in items),
        carbs=sum(item.macros.carbs * item.quantity for item in items),
        fats=sum(item.macros.fats * item.quantity for item in items),
        fiber=sum((item.macros.fiber or 0) * item.quantity for item in items),
        sugar=sum((item.macros.sugar or 0) * item.quantity for item in items)
    )
    
    # Round values
    totals.calories = round(totals.calories, 1)
    totals.protein = round(totals.protein, 1) 
    totals.carbs = round(totals.carbs, 1)
    totals.fats = round(totals.fats, 1)
    if totals.fiber: totals.fiber = round(totals.fiber, 1)
    if totals.sugar: totals.sugar = round(totals.sugar, 1)
    
    return totals

def simple_food_extraction(text: str) -> List[dict]:
    """Simple fallback extraction if NLP module fails"""
    import re
    
    common_foods = [
        "apple", "banana", "orange", "bread", "chicken", "rice", 
        "egg", "cheese", "milk", "beef", "fish", "potato", 
        "tomato", "lettuce", "pasta", "pizza", "burger", "salad"
    ]
    
    found_foods = []
    text_lower = text.lower()
    
    for food in common_foods:
        if food in text_lower:
            # Try to extract quantity
            quantity = 1.0
            qty_patterns = [
                rf"(\d+(?:\.\d+)?)\s*{food}",
                rf"{food}\s*(\d+(?:\.\d+)?)",
                rf"(\d+(?:\.\d+)?)\s*(?:slices?\s+of\s+|cups?\s+of\s+|pieces?\s+of\s+)?{food}",
            ]
            
            for pattern in qty_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        quantity = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        pass
            
            found_foods.append({
                "ingredient": food,
                "quantity": quantity,
                "unit": "serving"
            })
    
    if not found_foods:
        found_foods.append({
            "ingredient": "mixed food",
            "quantity": 1.0,
            "unit": "serving"
        })
    
    return found_foods

async def process_text_analysis(text: str, include_usda: bool = True) -> List[FoodItem]:
    """Enhanced text processing with USDA integration"""
    try:
        if NLP_AVAILABLE:
            logger.info("Using hybrid NLP extraction")
            extracted_items = hybrid_extract(text)
        else:
            logger.warning("NLP not available, using simple extraction")
            extracted_items = simple_food_extraction(text)
            
    except Exception as e:
        logger.error(f"NLP extraction failed: {e}, using fallback")
        extracted_items = simple_food_extraction(text)
    
    if not extracted_items:
        return []
    
    processed_items = []
    
    for item in extracted_items:
        try:
            item_name = item.get("ingredient", "unknown")
            item_quantity = float(item.get("quantity", 1.0))
            item_unit = item.get("unit", "serving")
            
            # Get nutrition from USDA if enabled
            macros = None
            notes = []
            usda_food_id = None
            
            if include_usda and USDA_API_KEY != 'your_usda_api_key_here':
                try:
                    usda_results = await search_usda_food(item_name, limit=1)
                    if usda_results:
                        usda_food = usda_results[0]
                        usda_food_id = str(usda_food.get('fdcId', ''))
                        
                        usda_detail = await get_usda_nutrition(usda_food_id)
                        if usda_detail:
                            macros = extract_usda_macros(usda_detail)
                            # Scale by quantity
                            macros.calories *= item_quantity
                            macros.protein *= item_quantity
                            macros.carbs *= item_quantity
                            macros.fats *= item_quantity
                            if macros.fiber: macros.fiber *= item_quantity
                            if macros.sugar: macros.sugar *= item_quantity
                        else:
                            notes.append("USDA nutrition lookup failed")
                    else:
                        notes.append("No USDA match found")
                        
                except Exception as e:
                    logger.error(f"USDA processing error: {e}")
                    notes.append(f"USDA error: {str(e)}")
            
            # Fallback to mock data
            if not macros:
                mock_nutrition = get_mock_nutrition_by_food_name(item_name)
                scaled_nutrition = {k: v * item_quantity for k, v in mock_nutrition.items()}
                macros = MacroInfo(**scaled_nutrition)
                notes.append("Using estimated nutrition values")
            
            food_item = FoodItem(
                name=item_name,
                quantity=item_quantity,
                unit=item_unit,
                macros=macros,
                confidence=0.8 if usda_food_id else 0.5,
                source="text_usda" if usda_food_id else "text_mock",
                notes="; ".join(notes) if notes else None,
                usda_food_id=usda_food_id
            )
            
            processed_items.append(food_item)
            
        except Exception as e:
            logger.error(f"Error processing text item {item}: {str(e)}")
            continue
    
    return processed_items

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Nutri-Vision Unified API",
        "version": "2.1.0-enhanced",
        "description": "Enhanced nutrition analysis with LogMeal and USDA integration",
        "status": "running",
        "services": {
            "nlp": "available" if NLP_AVAILABLE else "fallback_mode",
            "logmeal": "configured" if LOGMEAL_API_TOKEN != 'your_logmeal_token_here' else "not_configured",
            "usda": "configured" if USDA_API_KEY != 'your_usda_api_key_here' else "not_configured"
        },
        "endpoints": {
            "text": "/analyze/text",
            "image": "/analyze/image", 
            "legacy": "/analyze-text",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "version": "2.1.0-enhanced",
        "services": {
            "api": "active",
            "nlp": "available" if NLP_AVAILABLE else "fallback",
            "logmeal": "configured" if LOGMEAL_API_TOKEN != 'your_logmeal_token_here' else "needs_token",
            "usda": "configured" if USDA_API_KEY != 'your_usda_api_key_here' else "needs_token"
        },
        "integration_status": "enhanced"
    }

@app.post("/analyze/text", response_model=NutritionAnalysis)
async def analyze_text(request: TextAnalysisRequest):
    """Enhanced text analysis with USDA integration"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing enhanced text analysis: '{request.text}'")
        
        items = await process_text_analysis(request.text, request.include_usda)
        totals = calculate_totals(items)
        
        processing_time = round(time.time() - start_time, 3)
        warnings = []
        
        if not items:
            warnings.append("No food items could be identified")
        
        if not NLP_AVAILABLE:
            warnings.append("Using simplified extraction (NLP module unavailable)")
            
        if USDA_API_KEY == 'your_usda_api_key_here':
            warnings.append("Using mock nutrition data (USDA API key not configured)")
        
        return NutritionAnalysis(
            success=True,
            input_type="text",
            raw_input=request.text,
            items=items,
            totals=totals,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "nlp_available": NLP_AVAILABLE,
                "usda_configured": USDA_API_KEY != 'your_usda_api_key_here',
                "usda_lookup_enabled": request.include_usda,
                "items_with_usda": sum(1 for item in items if item.usda_food_id),
                "items_with_mock": sum(1 for item in items if not item.usda_food_id)
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced text analysis failed: {str(e)}")
        
        return NutritionAnalysis(
            success=False,
            input_type="text",
            raw_input=request.text,
            items=[],
            totals=MacroInfo(),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Analysis failed: {str(e)}"],
            metadata={"error": str(e)}
        )

@app.post("/analyze/image", response_model=NutritionAnalysis)
async def analyze_image(
    image: UploadFile = File(..., description="Image file containing food"),
    include_nutrition: bool = Form(True, description="Include nutrition lookup")
):
    """Enhanced image analysis with LogMeal segmentation and USDA nutrition"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing enhanced image analysis: {image.filename}")
        
        # Validate image
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image data
        image_data = await image.read()
        
        try:
            img = PILImage.open(BytesIO(image_data))
            img.verify()
            logger.info(f"Image validated: {img.format} {img.size}")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_error)}")
        
        # Check LogMeal API configuration
        if LOGMEAL_API_TOKEN == 'your_logmeal_token_here':
            logger.warning("LogMeal API not configured, using mock detection")
            
            # Mock food detection for demo purposes
            mock_items = [
                {
                    "name": "Mixed Food Item",
                    "confidence": 0.5,
                    "food_id": None,
                    "category": "unknown",
                    "source": "mock"
                }
            ]
            
            warnings = ["LogMeal API token not configured - using mock food detection"]
        else:
            # Step 1: Use LogMeal for food detection and segmentation
            logger.info("Calling LogMeal API for food detection")
            logmeal_result = await logmeal_detect_food(image_data, image.filename)
            
            if "error" in logmeal_result:
                logger.error(f"LogMeal API error: {logmeal_result['error']}")
                return NutritionAnalysis(
                    success=False,
                    input_type="image",
                    raw_input=f"Image file: {image.filename}",
                    items=[],
                    totals=MacroInfo(),
                    processing_time=round(time.time() - start_time, 3),
                    warnings=[f"LogMeal API error: {logmeal_result['error']}"],
                    metadata={"logmeal_error": logmeal_result['error']}
                )
            
            # Step 2: Parse LogMeal results to extract food items
            detected_items = parse_logmeal_results(logmeal_result)
            
            if not detected_items:
                logger.warning("No food items detected by LogMeal")
                # Fallback to generic food item
                mock_items = [
                    {
                        "name": "Unidentified Food",
                        "confidence": 0.3,
                        "food_id": None,
                        "category": "unknown", 
                        "source": "logmeal_fallback"
                    }
                ]
                warnings = ["No specific food items detected in image"]
            else:
                mock_items = detected_items
                warnings = []
        
        # Step 3: Get nutrition information for each detected item
        processed_items = []
        
        for item_data in mock_items:
            try:
                if include_nutrition:
                    # Get comprehensive nutrition using both LogMeal and USDA
                    food_item = await get_nutrition_for_ingredient(item_data)
                else:
                    # Just use basic mock nutrition
                    mock_nutrition = get_mock_nutrition_by_food_name(item_data["name"])
                    food_item = FoodItem(
                        name=item_data["name"],
                        quantity=1.0,
                        unit="portion",
                        macros=MacroInfo(**mock_nutrition),
                        confidence=item_data.get("confidence", 0.5),
                        source=item_data.get("source", "logmeal"),
                        notes="Nutrition lookup disabled",
                        logmeal_food_id=item_data.get("food_id")
                    )
                
                processed_items.append(food_item)
                logger.info(f"Processed food item: {food_item.name} - {food_item.macros.calories} cal")
                
            except Exception as item_error:
                logger.error(f"Error processing food item {item_data}: {str(item_error)}")
                
                # Add error item with basic nutrition
                error_nutrition = get_mock_nutrition_by_food_name(item_data.get("name", "unknown"))
                error_item = FoodItem(
                    name=item_data.get("name", "Unknown Food"),
                    quantity=1.0,
                    unit="portion",
                    macros=MacroInfo(**error_nutrition),
                    confidence=0.1,
                    source="error",
                    notes=f"Processing error: {str(item_error)}",
                    logmeal_food_id=item_data.get("food_id")
                )
                processed_items.append(error_item)
        
        # Calculate totals
        totals = calculate_totals(processed_items)
        
        processing_time = round(time.time() - start_time, 3)
        
        # Add configuration warnings
        if LOGMEAL_API_TOKEN == 'your_logmeal_token_here':
            warnings.append("LogMeal API token not configured - set LOGMEAL_TOKEN environment variable")
        
        if USDA_API_KEY == 'your_usda_api_key_here':
            warnings.append("USDA API key not configured - using estimated nutrition values")
        
        logger.info(f"Successfully processed image with {len(processed_items)} food items")
        
        return NutritionAnalysis(
            success=True,
            input_type="image",
            raw_input=f"Image file: {image.filename} ({image.content_type})",
            items=processed_items,
            totals=totals,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "image_filename": image.filename,
                "image_type": image.content_type,
                "logmeal_configured": LOGMEAL_API_TOKEN != 'your_logmeal_token_here',
                "usda_configured": USDA_API_KEY != 'your_usda_api_key_here',
                "nutrition_lookup_enabled": include_nutrition,
                "detected_items_count": len(processed_items),
                "items_with_usda": sum(1 for item in processed_items if item.usda_food_id),
                "items_with_logmeal_id": sum(1 for item in processed_items if item.logmeal_food_id)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced image analysis failed: {str(e)}")
        
        return NutritionAnalysis(
            success=False,
            input_type="image",
            raw_input=f"Image file: {image.filename}",
            items=[],
            totals=MacroInfo(),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Image analysis failed: {str(e)}"],
            metadata={"error": str(e)}
        )

@app.post("/analyze-text")
async def legacy_analyze_text(payload: dict):
    """Legacy endpoint for backward compatibility"""
    try:
        text = payload.get("description", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Description is required")
        
        # Use the enhanced analysis method
        request = TextAnalysisRequest(text=text, include_usda=True)
        result = await analyze_text(request)
        
        if not result.success:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Convert to legacy format
        legacy_items = []
        for item in result.items:
            legacy_items.append({
                "ingredient": item.name,
                "quantity": item.quantity,
                "unit": item.unit,
                "macros": {
                    "calories": item.macros.calories,
                    "protein_g": item.macros.protein,
                    "carbs_g": item.macros.carbs,
                    "fat_g": item.macros.fats,
                    "fiber_g": item.macros.fiber,
                    "sugar_g": item.macros.sugar
                },
                "usda_match_score": item.confidence,
                "usda_food_id": item.usda_food_id,
                "note": item.notes
            })
        
        return {
            "input": text,
            "items": legacy_items,
            "totals": {
                "calories": result.totals.calories,
                "protein_g": result.totals.protein,
                "carbs_g": result.totals.carbs,
                "fat_g": result.totals.fats,
                "fiber_g": result.totals.fiber,
                "sugar_g": result.totals.sugar
            },
            "processing_time": result.processing_time,
            "api_version": "2.1.0-enhanced"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze/voice")
async def analyze_voice_placeholder(payload: dict):
    """Voice analysis placeholder - future implementation"""
    return {
        "success": False,
        "message": "Voice analysis not yet implemented",
        "note": "This endpoint will support speech-to-text and nutrition analysis"
    }

# =============================================================================
# ADDITIONAL API ENDPOINTS FOR TESTING AND UTILITIES
# =============================================================================

@app.get("/test/usda-search")
async def test_usda_search(query: str):
    """Test USDA food search functionality"""
    if USDA_API_KEY == 'your_usda_api_key_here':
        raise HTTPException(status_code=503, detail="USDA API key not configured")
    
    try:
        results = await search_usda_food(query, limit=5)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"USDA search error: {str(e)}")

@app.get("/test/usda-nutrition/{food_id}")
async def test_usda_nutrition(food_id: str):
    """Test USDA nutrition lookup for a specific food ID"""
    if USDA_API_KEY == 'your_usda_api_key_here':
        raise HTTPException(status_code=503, detail="USDA API key not configured")
    
    try:
        nutrition_data = await get_usda_nutrition(food_id)
        if not nutrition_data:
            raise HTTPException(status_code=404, detail="Food ID not found")
        
        macros = extract_usda_macros(nutrition_data)
        
        return {
            "food_id": food_id,
            "food_name": nutrition_data.get("description", "Unknown"),
            "macros": {
                "calories": macros.calories,
                "protein": macros.protein,
                "carbs": macros.carbs,
                "fats": macros.fats,
                "fiber": macros.fiber,
                "sugar": macros.sugar
            },
            "raw_nutrients": nutrition_data.get("foodNutrients", [])[:10]  # First 10 for brevity
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"USDA nutrition error: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current API configuration status"""
    return {
        "version": "2.1.0-enhanced",
        "services": {
            "nlp_module": {
                "available": NLP_AVAILABLE,
                "status": "available" if NLP_AVAILABLE else "using_fallback"
            },
            "logmeal_api": {
                "configured": LOGMEAL_API_TOKEN != 'your_logmeal_token_here',
                "status": "ready" if LOGMEAL_API_TOKEN != 'your_logmeal_token_here' else "needs_token",
                "base_url": LOGMEAL_BASE_URL
            },
            "usda_api": {
                "configured": USDA_API_KEY != 'your_usda_api_key_here',
                "status": "ready" if USDA_API_KEY != 'your_usda_api_key_here' else "needs_token",
                "base_url": USDA_BASE_URL
            }
        },
        "features": {
            "text_analysis": True,
            "image_analysis": True,
            "voice_analysis": False,
            "usda_integration": USDA_API_KEY != 'your_usda_api_key_here',
            "logmeal_integration": LOGMEAL_API_TOKEN != 'your_logmeal_token_here'
        },
        "setup_instructions": {
            "logmeal": "Set LOGMEAL_TOKEN environment variable with your LogMeal API token",
            "usda": "Set USDA_API_KEY environment variable with your USDA FoodData Central API key",
            "nlp": "Ensure nlp.hybrid_extractor module is available for advanced text parsing"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("="*60)
    logger.info("Starting Enhanced Nutri-Vision API v2.1.0")
    logger.info("="*60)
    logger.info(f"🚀 Server starting on port {port}")
    logger.info(f"📊 NLP Module: {'✅ Available' if NLP_AVAILABLE else '⚠️  Fallback mode'}")
    logger.info(f"🍎 LogMeal API: {'✅ Configured' if LOGMEAL_API_TOKEN != 'your_logmeal_token_here' else '❌ Not configured'}")
    logger.info(f"🥗 USDA API: {'✅ Configured' if USDA_API_KEY != 'your_usda_api_key_here' else '❌ Not configured'}")
    logger.info("="*60)
    logger.info("Available endpoints:")
    logger.info("  • GET  /           - API information")
    logger.info("  • GET  /health     - Health check")
    logger.info("  • GET  /config     - Configuration status")
    logger.info("  • POST /analyze/text  - Enhanced text analysis")
    logger.info("  • POST /analyze/image - Image analysis with LogMeal + USDA")
    logger.info("  • POST /analyze-text  - Legacy text endpoint")
    logger.info("  • GET  /test/usda-search - Test USDA search")
    logger.info("  • GET  /test/usda-nutrition/{id} - Test USDA nutrition")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
