import httpx
from io import BytesIO
from PIL import Image as PILImage
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union
import logging
import os

# Import your existing NLP components with error handling
try:
    from nlp.hybrid_extractor import hybrid_extract
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"NLP import failed: {e}")
    NLP_AVAILABLE = False

try:
    from usda.fooddata_api import get_nutrition_for_item
    USDA_AVAILABLE = True
except ImportError as e:
    print(f"USDA import failed: {e}")
    USDA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nutri-Vision Unified API",
    description="Unified nutrition analysis API supporting text, image, and voice inputs",
    version="2.0.1-fixed"
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
# FIXED MODELS 
# =============================================================================

class MacroInfo(BaseModel):
    """Standardized macronutrient information - FIXED"""
    calories: float = Field(default=0.0, ge=0, description="Calories in kcal")
    protein: float = Field(default=0.0, ge=0, description="Protein in grams")
    carbs: float = Field(default=0.0, ge=0, description="Carbohydrates in grams")
    fats: float = Field(default=0.0, ge=0, description="Fats in grams")

class FoodItem(BaseModel):
    """Individual food item with nutrition info"""
    name: str = Field(..., description="Food item name")
    quantity: float = Field(default=1.0, gt=0, description="Quantity amount")
    unit: str = Field(default="serving", description="Measurement unit")
    macros: MacroInfo
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Recognition confidence (0-1)")
    source: str = Field(default="api", description="Data source")
    notes: Optional[str] = Field(None, description="Additional notes or warnings")

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
# UTILITY FUNCTIONS - FIXED
# =============================================================================

def safe_normalize_macros(macros_dict: dict) -> MacroInfo:
    """Safely convert macro dict to MacroInfo, handling field name variations"""
    try:
        # Handle different possible field names from your USDA API
        calories = float(macros_dict.get("calories", 0.0))
        protein = float(macros_dict.get("protein_g", macros_dict.get("protein", 0.0)))
        carbs = float(macros_dict.get("carbs_g", macros_dict.get("carbohydrates", macros_dict.get("carbs", 0.0))))
        fats = float(macros_dict.get("fat_g", macros_dict.get("fats", macros_dict.get("fat", 0.0))))
        
        return MacroInfo(
            calories=calories,
            protein=protein,
            carbs=carbs,
            fats=fats
        )
    except Exception as e:
        logger.error(f"Error normalizing macros {macros_dict}: {e}")
        return MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0)

def calculate_totals(items: List[FoodItem]) -> MacroInfo:
    """Calculate total macronutrients from food items"""
    total_calories = sum(item.macros.calories for item in items)
    total_protein = sum(item.macros.protein for item in items)
    total_carbs = sum(item.macros.carbs for item in items)
    total_fats = sum(item.macros.fats for item in items)
    
    return MacroInfo(
        calories=round(total_calories, 2),
        protein=round(total_protein, 2),
        carbs=round(total_carbs, 2),
        fats=round(total_fats, 2)
    )

def simple_food_extraction(text: str) -> List[dict]:
    """Simple fallback extraction if NLP module fails"""
    import re
    
    # Basic food detection
    common_foods = {
        "apple": {"calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3},
        "banana": {"calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4},
        "bread": {"calories": 80, "protein_g": 3, "carbs_g": 15, "fat_g": 1},
        "chicken": {"calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
        "rice": {"calories": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3},
        "egg": {"calories": 70, "protein_g": 6, "carbs_g": 0.5, "fat_g": 5},
    }
    
    found_foods = []
    text_lower = text.lower()
    
    for food, nutrition in common_foods.items():
        if food in text_lower:
            # Try to extract quantity
            quantity = 1.0
            qty_patterns = [
                rf"(\d+(?:\.\d+)?)\s*{food}",
                rf"{food}\s*(\d+(?:\.\d+)?)",
                rf"(\d+(?:\.\d+)?)\s*(?:slices?\s+of\s+)?{food}",
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
    
    # If no foods found, return generic item
    if not found_foods:
        found_foods.append({
            "ingredient": "mixed food",
            "quantity": 1.0,
            "unit": "serving"
        })
    
    return found_foods

# =============================================================================
# MAIN PROCESSING FUNCTION - FIXED
# =============================================================================

async def process_text_analysis(text: str, include_usda: bool = True) -> List[FoodItem]:
    """Process text input with improved error handling"""
    
    try:
        # Try to use your NLP system
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
        logger.warning(f"No food items extracted from: '{text}'")
        return []
    
    logger.info(f"Extracted {len(extracted_items)} items")
    processed_items = []
    
    for item in extracted_items:
        try:
            item_name = item.get("ingredient", "unknown")
            item_quantity = item.get("quantity", 1.0)
            item_unit = item.get("unit", "serving")
            
            # Get nutrition info
            if include_usda and USDA_AVAILABLE:
                try:
                    nutrition_result = get_nutrition_for_item(item)
                    
                    if nutrition_result.get("error"):
                        # Use mock data for failed lookups
                        macros = MacroInfo(calories=100.0, protein=5.0, carbs=15.0, fats=3.0)
                        notes = f"USDA lookup failed: {nutrition_result['error']}"
                        confidence = 0.3
                    else:
                        # Safely normalize the macros
                        macros_raw = nutrition_result.get("macros", {})
                        macros = safe_normalize_macros(macros_raw)
                        notes = None
                        confidence = nutrition_result.get("score", 0.8)
                        
                except Exception as usda_error:
                    logger.error(f"USDA API error: {usda_error}")
                    macros = MacroInfo(calories=100.0, protein=5.0, carbs=15.0, fats=3.0)
                    notes = f"USDA API error: {str(usda_error)}"
                    confidence = 0.3
            else:
                # Use mock nutrition data
                mock_nutrition = {
                    "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fats": 0.3},
                    "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.4},
                    "bread": {"calories": 80, "protein": 3, "carbs": 15, "fats": 1},
                    "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6},
                    "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fats": 0.3},
                }.get(item_name.lower(), {"calories": 100, "protein": 5, "carbs": 15, "fats": 3})
                
                # Scale by quantity
                scaled_nutrition = {k: v * item_quantity for k, v in mock_nutrition.items()}
                macros = MacroInfo(**scaled_nutrition)
                notes = "Mock nutrition data (USDA disabled or unavailable)"
                confidence = 0.5
            
            # Create food item
            food_item = FoodItem(
                name=item_name,
                quantity=item_quantity,
                unit=item_unit,
                macros=macros,
                confidence=confidence,
                source="text_nlp" if NLP_AVAILABLE else "simple_extraction",
                notes=notes
            )
            
            processed_items.append(food_item)
            logger.info(f"Processed item: {item_name} - {macros.calories} cal")
            
        except Exception as e:
            logger.error(f"Error processing item {item}: {str(e)}")
            
            # Add error item with zero macros
            error_item = FoodItem(
                name=item.get("ingredient", "unknown"),
                quantity=item.get("quantity", 1.0),
                unit=item.get("unit", "serving"),
                macros=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
                confidence=0.0,
                source="error",
                notes=f"Processing error: {str(e)}"
            )
            processed_items.append(error_item)
    
    return processed_items

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Nutri-Vision Unified API",
        "version": "2.0.1-fixed",
        "description": "Fixed nutrition analysis API",
        "status": "running",
        "services": {
            "nlp": "available" if NLP_AVAILABLE else "fallback_mode",
            "usda": "available" if USDA_AVAILABLE else "mock_data"
        },
        "endpoints": {
            "text": "/analyze/text",
            "legacy": "/analyze-text",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.1-fixed",
        "services": {
            "api": "active",
            "nlp": "available" if NLP_AVAILABLE else "fallback",
            "usda": "available" if USDA_AVAILABLE else "mock",
        },
        "message": "API is running with fixes applied"
    }

@app.post("/analyze/text", response_model=NutritionAnalysis)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze nutrition from text description - FIXED"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing text analysis: '{request.text}'")
        
        # Process text
        items = await process_text_analysis(request.text, request.include_usda)
        
        # Calculate totals
        totals = calculate_totals(items)
        
        processing_time = round(time.time() - start_time, 3)
        warnings = []
        
        if not items:
            warnings.append("No food items could be identified")
        
        if not NLP_AVAILABLE:
            warnings.append("Using simplified extraction (NLP module unavailable)")
            
        if not USDA_AVAILABLE:
            warnings.append("Using mock nutrition data (USDA API unavailable)")
        
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
                "usda_available": USDA_AVAILABLE,
                "usda_lookup_enabled": request.include_usda
            }
        )
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        
        # Return error response instead of raising exception
        return NutritionAnalysis(
            success=False,
            input_type="text",
            raw_input=request.text,
            items=[],
            totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Analysis failed: {str(e)}"],
            metadata={"error": str(e)}
        )

@app.post("/analyze-text")
async def legacy_analyze_text(payload: dict):
    """Legacy endpoint for backward compatibility - FIXED"""
    try:
        text = payload.get("description", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Description is required")
        
        # Use the new analysis method
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
                    "fat_g": item.macros.fats
                },
                "usda_match_score": item.confidence,
                "note": item.notes
            })
        
        return {
            "input": text,
            "items": legacy_items,
            "totals": {
                "calories": result.totals.calories,
                "protein_g": result.totals.protein,
                "carbs_g": result.totals.carbs,
                "fat_g": result.totals.fats
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Placeholder endpoints
# Replace the placeholder endpoint with this implementation
@app.post("/analyze/image", response_model=NutritionAnalysis)
async def analyze_image(
    image: UploadFile = File(..., description="Image file containing food"),
    include_nutrition: bool = Form(True, description="Include nutrition lookup")
):
    """Analyze nutrition from food image using LogMeal API"""
    import time
    start_time = time.time()
    
    # LogMeal API Configuration
    LOGMEAL_API_TOKEN = os.getenv("LOGMEAL_API_TOKEN")
    LOGMEAL_BASE_URL = "https://api.logmeal.es"
    
    if not LOGMEAL_API_TOKEN:
        logger.warning("LogMeal API token not configured")
        return NutritionAnalysis(
            success=False,
            input_type="image",
            raw_input=f"Image file: {image.filename}",
            items=[],
            totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
            processing_time=round(time.time() - start_time, 3),
            warnings=["LogMeal API token not configured. Set LOGMEAL_API_TOKEN environment variable."],
            metadata={"error": "api_token_missing"}
        )
    
    try:
        logger.info(f"Processing image analysis: {image.filename}")
        
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Validate image format using PIL
        try:
            from PIL import Image as PILImage
            img = PILImage.open(BytesIO(image_data))
            img.verify()  # Verify it's a valid image
            logger.info(f"Image validated: {img.format} {img.size}")
        except Exception as img_error:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_error)}")
        
        # Prepare LogMeal API headers
        headers = {
            "Authorization": f"Bearer {LOGMEAL_API_TOKEN}",
        }
        
        # Prepare multipart form data for image upload
        files = {
            "image": (image.filename, image_data, image.content_type)
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info("Calling LogMeal food detection API")
            
            # Step 1: Food Detection and Recognition
            # Using LogMeal's main detection endpoint
            detection_response = await client.post(
                f"{LOGMEAL_BASE_URL}/v2/image/segmentation/complete",
                headers=headers,
                files=files
            )
            
            if detection_response.status_code == 401:
                logger.error("LogMeal API authentication failed")
                return NutritionAnalysis(
                    success=False,
                    input_type="image",
                    raw_input=f"Image file: {image.filename}",
                    items=[],
                    totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
                    processing_time=round(time.time() - start_time, 3),
                    warnings=["LogMeal API authentication failed. Check your API token."],
                    metadata={"error": "authentication_failed", "status_code": 401}
                )
            
            if detection_response.status_code != 200:
                logger.error(f"LogMeal API error: {detection_response.status_code} - {detection_response.text}")
                return NutritionAnalysis(
                    success=False,
                    input_type="image",
                    raw_input=f"Image file: {image.filename}",
                    items=[],
                    totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
                    processing_time=round(time.time() - start_time, 3),
                    warnings=[f"LogMeal API error: HTTP {detection_response.status_code}"],
                    metadata={"error": "api_request_failed", "status_code": detection_response.status_code}
                )
            
            detection_data = detection_response.json()
            logger.info(f"LogMeal detection response keys: {list(detection_data.keys())}")
            
            # Process detection results
            processed_items = []
            
            # Handle different response formats from LogMeal API
            if "segmentation_results" in detection_data:
                # Standard segmentation response
                for segment in detection_data["segmentation_results"]:
                    try:
                        # Extract recognition results
                        recognition_results = segment.get("recognition_results", [])
                        if not recognition_results:
                            continue
                            
                        # Get the best recognition result
                        best_result = recognition_results[0]  # Usually sorted by confidence
                        
                        food_name = best_result.get("name", "Unknown Food")
                        confidence = best_result.get("prob", 0.0)
                        food_id = best_result.get("food_id")
                        
                        logger.info(f"Detected food: {food_name} (confidence: {confidence})")
                        
                        # Get nutrition information if requested
                        nutrition_macros = MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0)
                        notes = None
                        
                        if include_nutrition and food_id:
                            try:
                                # Call LogMeal nutrition endpoint
                                nutrition_response = await client.get(
                                    f"{LOGMEAL_BASE_URL}/v2/nutrition/recipe/nutritionalInfo",
                                    headers=headers,
                                    params={"food_id": food_id}
                                )
                                
                                if nutrition_response.status_code == 200:
                                    nutrition_data = nutrition_response.json()
                                    logger.info(f"Nutrition data keys: {list(nutrition_data.keys())}")
                                    
                                    # Extract nutrition information
                                    # LogMeal typically returns nutrition per 100g
                                    nutrition_macros = MacroInfo(
                                        calories=float(nutrition_data.get("calories", 0.0)),
                                        protein=float(nutrition_data.get("protein", 0.0)),
                                        carbs=float(nutrition_data.get("carbohydrates", 0.0)),
                                        fats=float(nutrition_data.get("fat", 0.0))
                                    )
                                else:
                                    logger.warning(f"Nutrition lookup failed: {nutrition_response.status_code}")
                                    notes = f"Nutrition lookup failed (HTTP {nutrition_response.status_code})"
                                    
                            except Exception as nutrition_error:
                                logger.error(f"Nutrition lookup error: {str(nutrition_error)}")
                                notes = f"Nutrition lookup error: {str(nutrition_error)}"
                        
                        # If no nutrition data retrieved, use mock values based on food type
                        if nutrition_macros.calories == 0.0:
                            mock_nutrition = get_mock_nutrition_by_food_name(food_name)
                            nutrition_macros = MacroInfo(**mock_nutrition)
                            notes = "Using estimated nutrition values"
                        
                        # Create food item
                        food_item = FoodItem(
                            name=food_name,
                            quantity=1.0,  # LogMeal typically detects portions, could be enhanced
                            unit="portion",
                            macros=nutrition_macros,
                            confidence=confidence,
                            source="logmeal_api",
                            notes=notes
                        )
                        
                        processed_items.append(food_item)
                        
                    except Exception as item_error:
                        logger.error(f"Error processing food item: {str(item_error)}")
                        continue
                        
            elif "recognition_results" in detection_data:
                # Direct recognition response format
                recognition_results = detection_data["recognition_results"]
                
                for result in recognition_results[:5]:  # Limit to top 5 results
                    try:
                        food_name = result.get("name", "Unknown Food")
                        confidence = result.get("prob", 0.0)
                        
                        # Skip low confidence results
                        if confidence < 0.1:
                            continue
                            
                        logger.info(f"Detected food: {food_name} (confidence: {confidence})")
                        
                        # Use mock nutrition data
                        mock_nutrition = get_mock_nutrition_by_food_name(food_name)
                        nutrition_macros = MacroInfo(**mock_nutrition)
                        
                        food_item = FoodItem(
                            name=food_name,
                            quantity=1.0,
                            unit="serving",
                            macros=nutrition_macros,
                            confidence=confidence,
                            source="logmeal_api",
                            notes="Using estimated nutrition values"
                        )
                        
                        processed_items.append(food_item)
                        
                    except Exception as item_error:
                        logger.error(f"Error processing recognition result: {str(item_error)}")
                        continue
            
            else:
                logger.warning("Unexpected LogMeal API response format")
                return NutritionAnalysis(
                    success=False,
                    input_type="image",
                    raw_input=f"Image file: {image.filename}",
                    items=[],
                    totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
                    processing_time=round(time.time() - start_time, 3),
                    warnings=["Unexpected API response format from LogMeal"],
                    metadata={"logmeal_response_keys": list(detection_data.keys())}
                )
            
            # Calculate totals
            totals = calculate_totals(processed_items)
            
            processing_time = round(time.time() - start_time, 3)
            warnings = []
            
            if not processed_items:
                warnings.append("No food items could be detected in the image")
            
            logger.info(f"Successfully processed {len(processed_items)} food items from image")
            
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
                    "logmeal_api": "enabled",
                    "nutrition_lookup_enabled": include_nutrition,
                    "detected_items_count": len(processed_items)
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        return NutritionAnalysis(
            success=False,
            input_type="image",
            raw_input=f"Image file: {image.filename}",
            items=[],
            totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Image analysis failed: {str(e)}"],
            metadata={"error": str(e)}
        )


def get_mock_nutrition_by_food_name(food_name: str) -> dict:
    """Get mock nutrition data based on food name"""
    food_name_lower = food_name.lower()
    
    # Extended mock nutrition database
    nutrition_db = {
        # Fruits
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fats": 0.3},
        "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.4},
        "orange": {"calories": 65, "protein": 1.3, "carbs": 16, "fats": 0.2},
        
        # Vegetables  
        "broccoli": {"calories": 55, "protein": 4.6, "carbs": 11, "fats": 0.6},
        "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fats": 0.2},
        "tomato": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fats": 0.2},
        
        # Proteins
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fats": 15},
        "fish": {"calories": 206, "protein": 22, "carbs": 0, "fats": 12},
        "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fats": 11},
        
        # Carbs
        "bread": {"calories": 265, "protein": 9, "carbs": 49, "fats": 3.2},
        "rice": {"calories": 365, "protein": 7.1, "carbs": 80, "fats": 0.9},
        "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fats": 1.1},
        "potato": {"calories": 161, "protein": 4.3, "carbs": 37, "fats": 0.1},
        
        # Popular dishes
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fats": 10},
        "burger": {"calories": 295, "protein": 17, "carbs": 23, "fats": 14},
        "salad": {"calories": 65, "protein": 5, "carbs": 7, "fats": 4},
        "sandwich": {"calories": 230, "protein": 10, "carbs": 30, "fats": 8},
        "soup": {"calories": 85, "protein": 4, "carbs": 12, "fats": 2.5},
        
        # Snacks & Others
        "cheese": {"calories": 113, "protein": 7, "carbs": 1, "fats": 9},
        "nuts": {"calories": 607, "protein": 20, "carbs": 16, "fats": 54},
        "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fats": 0.4},
    }
    
    # Try to find matching food
    for key, nutrition in nutrition_db.items():
        if key in food_name_lower or food_name_lower in key:
            return nutrition
    
    # Default nutrition for unknown foods
    return {"calories": 150, "protein": 8, "carbs": 20, "fats": 5}

@app.post("/analyze/voice")
async def analyze_voice_placeholder(payload: dict):
    """Voice analysis placeholder"""
    return {
        "success": False,
        "message": "Voice analysis not yet implemented"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting fixed API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)