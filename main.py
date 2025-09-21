from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union
import logging
import os
import httpx
import base64
from io import BytesIO
from PIL import Image
import json

# Import your existing NLP components
from nlp.hybrid_extractor import hybrid_extract
from usda.fooddata_api import get_nutrition_for_item, _normalize_macros_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nutri-Vision Unified API",
    description="Unified nutrition analysis API supporting text, image, and voice inputs",
    version="2.0.0"
)

# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS (Unified JSON Structure)
# =============================================================================

class MacroInfo(BaseModel):
    """Standardized macronutrient information"""
    calories: float = Field(..., ge=0, description="Calories in kcal")
    protein: float = Field(..., ge=0, description="Protein in grams", alias="protein_g")
    carbs: float = Field(..., ge=0, description="Carbohydrates in grams", alias="carbs_g")
    fats: float = Field(..., ge=0, description="Fats in grams", alias="fat_g")

class FoodItem(BaseModel):
    """Individual food item with nutrition info"""
    name: str = Field(..., description="Food item name")
    quantity: float = Field(..., gt=0, description="Quantity amount")
    unit: str = Field(..., description="Measurement unit")
    macros: MacroInfo
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Recognition confidence (0-1)")
    source: str = Field(..., description="Data source (text_nlp, logmeal_api, etc.)")
    notes: Optional[str] = Field(None, description="Additional notes or warnings")

class NutritionAnalysis(BaseModel):
    """Unified nutrition analysis response"""
    success: bool = Field(..., description="Analysis success status")
    input_type: str = Field(..., description="Input type: text, image, or voice")
    raw_input: str = Field(..., description="Original input description or metadata")
    items: List[FoodItem] = Field(default=[], description="Detected food items")
    totals: MacroInfo = Field(..., description="Total macronutrients across all items")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    warnings: List[str] = Field(default=[], description="Processing warnings")
    metadata: dict = Field(default={}, description="Additional metadata")

# =============================================================================
# INPUT MODELS
# =============================================================================

class TextAnalysisRequest(BaseModel):
    """Text-based nutrition analysis request"""
    text: str = Field(..., min_length=1, max_length=2000, description="Food description text")
    include_usda: bool = Field(True, description="Whether to include USDA nutrition lookup")

class VoiceAnalysisRequest(BaseModel):
    """Voice-based nutrition analysis request (future implementation)"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field("wav", description="Audio format (wav, mp3, etc.)")
    language: str = Field("en", description="Language code")

# =============================================================================
# CONFIGURATION
# =============================================================================

# LogMeal API Configuration
LOGMEAL_API_URL = "https://api.logmeal.es/v2"
LOGMEAL_TOKEN = os.getenv("LOGMEAL_API_TOKEN")  # Set this in your Render environment

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_macros_to_unified(macros_dict: dict) -> MacroInfo:
    """Convert various macro formats to unified MacroInfo"""
    normalized = _normalize_macros_map(macros_dict)
    
    return MacroInfo(
        calories=normalized.get("calories", 0.0),
        protein=normalized.get("protein_g", 0.0),
        carbs=normalized.get("carbs_g", 0.0),
        fats=normalized.get("fat_g", 0.0)
    )

def calculate_totals(items: List[FoodItem]) -> MacroInfo:
    """Calculate total macronutrients from food items"""
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    
    for item in items:
        totals["calories"] += item.macros.calories
        totals["protein"] += item.macros.protein
        totals["carbs"] += item.macros.carbs
        totals["fats"] += item.macros.fats
    
    return MacroInfo(
        calories=round(totals["calories"], 2),
        protein=round(totals["protein"], 2),
        carbs=round(totals["carbs"], 2),
        fats=round(totals["fats"], 2)
    )

# =============================================================================
# TEXT ANALYSIS (Your existing NLP)
# =============================================================================

async def process_text_analysis(text: str, include_usda: bool = True) -> List[FoodItem]:
    """Process text input using your existing NLP pipeline"""
    
    # Extract food items using your hybrid approach
    extracted_items = hybrid_extract(text)
    
    if not extracted_items:
        logger.warning(f"No food items extracted from text: '{text}'")
        return []
    
    logger.info(f"Extracted {len(extracted_items)} items from text")
    
    processed_items = []
    
    for item in extracted_items:
        try:
            # Get nutrition info if USDA lookup is enabled
            if include_usda:
                nutrition_result = get_nutrition_for_item(item)
                
                if nutrition_result.get("error"):
                    # Use zero macros with error note
                    macros = MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0)
                    notes = f"USDA lookup failed: {nutrition_result['error']}"
                    confidence = 0.0
                else:
                    macros = normalize_macros_to_unified(nutrition_result.get("macros", {}))
                    notes = None
                    confidence = nutrition_result.get("score", 0.8)
            else:
                # Mock data for demo without USDA
                macros = MacroInfo(calories=100.0, protein=5.0, carbs=15.0, fats=3.0)
                notes = "Mock nutrition data (USDA lookup disabled)"
                confidence = 0.5
            
            food_item = FoodItem(
                name=item.get("ingredient", "unknown"),
                quantity=item.get("quantity", 1.0),
                unit=item.get("unit", "serving"),
                macros=macros,
                confidence=confidence,
                source="text_nlp",
                notes=notes
            )
            
            processed_items.append(food_item)
            
        except Exception as e:
            logger.error(f"Error processing text item {item}: {str(e)}")
            
            # Add item with zero macros and error
            error_item = FoodItem(
                name=item.get("ingredient", "unknown"),
                quantity=item.get("quantity", 1.0),
                unit=item.get("unit", "serving"),
                macros=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
                confidence=0.0,
                source="text_nlp",
                notes=f"Processing error: {str(e)}"
            )
            processed_items.append(error_item)
    
    return processed_items

# =============================================================================
# IMAGE ANALYSIS (LogMeal API)
# =============================================================================

async def process_image_analysis(image_file: UploadFile) -> List[FoodItem]:
    """Process image using LogMeal API"""
    
    if not LOGMEAL_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="LogMeal API token not configured. Set LOGMEAL_API_TOKEN environment variable."
        )
    
    try:
        # Read and validate image
        image_data = await image_file.read()
        
        # Validate image format
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()  # Verify it's a valid image
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Prepare LogMeal API request
        headers = {
            "Authorization": f"Bearer {LOGMEAL_TOKEN}",
        }
        
        # LogMeal expects form data with image file
        files = {
            "image": (image_file.filename, image_data, image_file.content_type)
        }
        
        # Call LogMeal Food Detection API
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Food Detection
            detection_response = await client.post(
                f"{LOGMEAL_API_URL}/image/segmentation/complete",
                headers=headers,
                files=files
            )
            
            if detection_response.status_code != 200:
                logger.error(f"LogMeal detection failed: {detection_response.status_code} - {detection_response.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"LogMeal API error: {detection_response.status_code}"
                )
            
            detection_data = detection_response.json()
            
            # Step 2: Get Nutrition Info
            nutrition_items = []
            
            if "segmentation_results" in detection_data:
                for food_item in detection_data["segmentation_results"]:
                    try:
                        # Extract food info from LogMeal response
                        food_name = food_item.get("recognition_results", [{}])[0].get("name", "Unknown Food")
                        confidence = food_item.get("recognition_results", [{}])[0].get("prob", 0.0)
                        
                        # Get nutrition data for detected food
                        food_id = food_item.get("recognition_results", [{}])[0].get("food_id")
                        
                        if food_id:
                            # Call LogMeal nutrition endpoint
                            nutrition_response = await client.get(
                                f"{LOGMEAL_API_URL}/food/{food_id}/nutritional_info",
                                headers=headers
                            )
                            
                            if nutrition_response.status_code == 200:
                                nutrition_data = nutrition_response.json()
                                
                                # Map LogMeal nutrition to our format
                                macros = MacroInfo(
                                    calories=nutrition_data.get("calories", 0.0),
                                    protein=nutrition_data.get("protein", 0.0),
                                    carbs=nutrition_data.get("carbohydrates", 0.0),
                                    fats=nutrition_data.get("fat", 0.0)
                                )
                            else:
                                # Fallback nutrition
                                macros = MacroInfo(calories=150.0, protein=8.0, carbs=20.0, fats=5.0)
                        else:
                            macros = MacroInfo(calories=150.0, protein=8.0, carbs=20.0, fats=5.0)
                        
                        food_item_obj = FoodItem(
                            name=food_name,
                            quantity=1.0,  # LogMeal typically detects portions
                            unit="portion",
                            macros=macros,
                            confidence=confidence,
                            source="logmeal_api",
                            notes=None
                        )
                        
                        nutrition_items.append(food_item_obj)
                        
                    except Exception as e:
                        logger.error(f"Error processing LogMeal food item: {str(e)}")
                        continue
        
        return nutrition_items
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Nutri-Vision Unified API",
        "version": "2.0.0",
        "description": "Unified nutrition analysis supporting text, image, and voice",
        "endpoints": {
            "text": "/analyze/text",
            "image": "/analyze/image", 
            "voice": "/analyze/voice",
            "health": "/health"
        },
        "features": ["NLP text analysis", "LogMeal image analysis", "Voice analysis (planned)"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "nlp": "active",
            "logmeal": "configured" if LOGMEAL_TOKEN else "not_configured",
            "voice": "planned"
        }
    }

@app.post("/analyze/text", response_model=NutritionAnalysis)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze nutrition from text description"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing text analysis: '{request.text}'")
        
        # Process using your existing NLP
        items = await process_text_analysis(request.text, request.include_usda)
        
        # Calculate totals
        totals = calculate_totals(items)
        
        processing_time = round(time.time() - start_time, 3)
        warnings = []
        
        if not items:
            warnings.append("No food items could be identified from the text")
        
        return NutritionAnalysis(
            success=True,
            input_type="text",
            raw_input=request.text,
            items=items,
            totals=totals,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "usda_lookup_enabled": request.include_usda,
                "extraction_method": "hybrid_nlp"
            }
        )
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze/image", response_model=NutritionAnalysis)
async def analyze_image(
    image: UploadFile = File(..., description="Image file containing food"),
    include_nutrition: bool = Form(True, description="Include nutrition lookup")
):
    """Analyze nutrition from food image using LogMeal API"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing image analysis: {image.filename}")
        
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        items = await process_image_analysis(image)
        
        # Calculate totals
        totals = calculate_totals(items)
        
        processing_time = round(time.time() - start_time, 3)
        warnings = []
        
        if not items:
            warnings.append("No food items could be detected in the image")
        
        return NutritionAnalysis(
            success=True,
            input_type="image",
            raw_input=f"Image file: {image.filename} ({image.content_type})",
            items=items,
            totals=totals,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "image_filename": image.filename,
                "image_type": image.content_type,
                "logmeal_api": "enabled" if LOGMEAL_TOKEN else "disabled"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze/voice", response_model=NutritionAnalysis)
async def analyze_voice(request: VoiceAnalysisRequest):
    """Analyze nutrition from voice input (future implementation)"""
    
    # Placeholder for voice analysis
    # In the future, you would:
    # 1. Decode base64 audio data
    # 2. Use speech-to-text service (Google Speech, Azure, etc.)
    # 3. Pass transcribed text to text analysis
    
    return NutritionAnalysis(
        success=False,
        input_type="voice",
        raw_input="Voice analysis not yet implemented",
        items=[],
        totals=MacroInfo(calories=0.0, protein=0.0, carbs=0.0, fats=0.0),
        warnings=["Voice analysis feature is not yet implemented"],
        metadata={
            "status": "planned",
            "audio_format": request.audio_format,
            "language": request.language
        }
    )

# =============================================================================
# LEGACY COMPATIBILITY (Optional)
# =============================================================================

@app.post("/analyze-text")
async def legacy_analyze_text(payload: dict):
    """Legacy endpoint for backward compatibility with your current frontend"""
    try:
        text = payload.get("description", "")
        
        request = TextAnalysisRequest(text=text, include_usda=True)
        result = await analyze_text(request)
        
        # Convert to legacy format if needed
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)