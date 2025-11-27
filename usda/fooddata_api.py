import requests
import json
from difflib import SequenceMatcher

# USDA FoodData Central API configuration
USDA_API_KEY = "ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk"
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# Common unit conversions to grams
UNIT_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "oz": 28.35,
    "ounce": 28.35,
    "ounces": 28.35,
    "lb": 453.59,
    "pound": 453.59,
    "pounds": 453.59,
    "cup": 240.0,
    "cups": 240.0,
    "ml": 1.0,
    "l": 1000.0,
    "litre": 1000.0,
    "tbsp": 15.0,
    "tablespoon": 15.0,
    "tsp": 5.0,
    "teaspoon": 5.0,
    "slice": 25.0,
    "slices": 25.0,
    "piece": 100.0,
    "pieces": 100.0,
    "serving": 100.0,
    "servings": 100.0,
    "glass": 240.0,
    "medium": 150.0,  # Added for medium size items
    "small": 100.0,   # Added for small size items
    "large": 200.0,   # Added for large size items
}

def search_food(query, limit=5):
    """Search for foods in USDA database"""
    if not USDA_API_KEY or USDA_API_KEY == "YOUR_API_KEY_HERE":
        return mock_search_food(query)
    
    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "query": query,
        "pageSize": limit,
        "api_key": USDA_API_KEY,
        "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"]
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"USDA API error: {e}")
        return {"foods": []}

def mock_search_food(query):
    """Mock data for testing when API key is not available"""
    mock_foods = {
        "apple": {
            "fdcId": 171688,
            "description": "Apples, raw, with skin",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 52.0, "unitName": "kcal"},
                {"nutrientId": 1003, "nutrientName": "Protein", "value": 0.26, "unitName": "g"},
                {"nutrientId": 1005, "nutrientName": "Carbohydrate, by difference", "value": 13.81, "unitName": "g"},
                {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 0.17, "unitName": "g"},
                {"nutrientId": 1079, "nutrientName": "Fiber, total dietary", "value": 2.4, "unitName": "g"},
                {"nutrientId": 2000, "nutrientName": "Sugars, total including NLEA", "value": 10.39, "unitName": "g"}
            ]
        },
        "chicken breast": {
            "fdcId": 171477,
            "description": "Chicken, broilers or fryers, breast, meat only, raw",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 165.0, "unitName": "kcal"},
                {"nutrientId": 1003, "nutrientName": "Protein", "value": 31.02, "unitName": "g"},
                {"nutrientId": 1005, "nutrientName": "Carbohydrate, by difference", "value": 0.0, "unitName": "g"},
                {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 3.57, "unitName": "g"},
                {"nutrientId": 1079, "nutrientName": "Fiber, total dietary", "value": 0.0, "unitName": "g"},
                {"nutrientId": 2000, "nutrientName": "Sugars, total including NLEA", "value": 0.0, "unitName": "g"}
            ]
        },
        "whole wheat bread": {
            "fdcId": 172687,
            "description": "Bread, whole-wheat, commercially prepared",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 247.0, "unitName": "kcal"},
                {"nutrientId": 1003, "nutrientName": "Protein", "value": 12.95, "unitName": "g"},
                {"nutrientId": 1005, "nutrientName": "Carbohydrate, by difference", "value": 41.29, "unitName": "g"},
                {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 4.17, "unitName": "g"},
                {"nutrientId": 1079, "nutrientName": "Fiber, total dietary", "value": 6.0, "unitName": "g"},
                {"nutrientId": 2000, "nutrientName": "Sugars, total including NLEA", "value": 5.0, "unitName": "g"}
            ]
        },
        "rice": {
            "fdcId": 169704,
            "description": "Rice, white, long-grain, regular, cooked",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 130.0, "unitName": "kcal"},
                {"nutrientId": 1003, "nutrientName": "Protein", "value": 2.69, "unitName": "g"},
                {"nutrientId": 1005, "nutrientName": "Carbohydrate, by difference", "value": 28.17, "unitName": "g"},
                {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 0.28, "unitName": "g"},
                {"nutrientId": 1079, "nutrientName": "Fiber, total dietary", "value": 0.4, "unitName": "g"},
                {"nutrientId": 2000, "nutrientName": "Sugars, total including NLEA", "value": 0.05, "unitName": "g"}
            ]
        },
    }
    
    best_match = None
    best_score = 0
    
    for food_name, food_data in mock_foods.items():
        score = SequenceMatcher(None, query.lower(), food_name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = food_data
    
    if best_match and best_score > 0.3:
        return {"foods": [best_match]}
    else:
        return {"foods": []}

def get_food_details(fdc_id):
    """Get detailed nutrition information for a specific food"""
    if not USDA_API_KEY or USDA_API_KEY == "YOUR_API_KEY_HERE":
        return None
    
    url = f"{USDA_BASE_URL}/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"USDA API error: {e}")
        return None

def _get_nutrient_value(nutrient_entry):
    """Extract numeric value from nutrient entry"""
    if nutrient_entry is None:
        return 0.0
    
    if isinstance(nutrient_entry, (int, float)):
        return float(nutrient_entry)
    
    if isinstance(nutrient_entry, dict):
        for key in ("value", "amount", "nutrientValue"):
            if key in nutrient_entry and nutrient_entry[key] is not None:
                try:
                    return float(nutrient_entry[key])
                except (ValueError, TypeError):
                    continue
    
    try:
        return float(nutrient_entry)
    except (ValueError, TypeError):
        return 0.0

def estimate_sugar_from_food_type(food_description, carbs_g):
    """
    Estimate sugar content based on food type and carbs
    Returns a reasonable sugar estimate
    """
    food_lower = food_description.lower()
    
    # High sugar foods (fruits, sweets, desserts)
    high_sugar_keywords = ['apple', 'banana', 'orange', 'strawberry', 'grape', 'mango', 
                          'pineapple', 'watermelon', 'candy', 'chocolate', 'cake', 
                          'cookie', 'dessert', 'ice cream', 'soda', 'juice', 'honey']
    
    # Medium sugar foods (some vegetables, dairy)
    medium_sugar_keywords = ['carrot', 'tomato', 'beet', 'milk', 'yogurt', 'corn']
    
    # Low/no sugar foods (meats, grains, fats)
    low_sugar_keywords = ['chicken', 'beef', 'pork', 'fish', 'egg', 'rice', 'bread', 
                         'pasta', 'oil', 'butter', 'cheese']
    
    # Check food type and estimate sugar percentage of carbs
    for keyword in high_sugar_keywords:
        if keyword in food_lower:
            return carbs_g * 0.75  # 75% of carbs are sugar in fruits/sweets
    
    for keyword in medium_sugar_keywords:
        if keyword in food_lower:
            return carbs_g * 0.35  # 35% of carbs are sugar
    
    for keyword in low_sugar_keywords:
        if keyword in food_lower:
            return carbs_g * 0.05  # 5% of carbs are sugar (minimal)
    
    # Default: moderate estimate
    return carbs_g * 0.30  # 30% of carbs are sugar (reasonable default)

def extract_core_macros(food_detail):
    """
    Extract calories, protein, carbs, fat, fiber, and sugar from food detail
    ALWAYS returns sugar value (estimated if not available)
    """
    macros = {
        "calories": 0.0, 
        "protein_g": 0.0, 
        "carbs_g": 0.0, 
        "fat_g": 0.0,
        "fiber_g": 0.0,
        "sugar_g": 0.0
    }
    
    nutrients = food_detail.get("foodNutrients", [])
    food_description = food_detail.get("description", "")
    
    sugar_found = False
    
    for nutrient in nutrients:
        nutrient_id = nutrient.get("nutrientId")
        nutrient_name = (nutrient.get("nutrientName") or "").lower()
        value = _get_nutrient_value(nutrient.get("value"))
        
        # Map nutrients by ID (most reliable)
        if nutrient_id == 1008:  # Energy
            macros["calories"] = value
        elif nutrient_id == 1003:  # Protein
            macros["protein_g"] = value
        elif nutrient_id == 1005:  # Carbohydrate
            macros["carbs_g"] = value
        elif nutrient_id == 1004:  # Total lipid (fat)
            macros["fat_g"] = value
        elif nutrient_id == 1079:  # Fiber, total dietary
            macros["fiber_g"] = value
        elif nutrient_id == 2000:  # Sugars, total including NLEA (primary)
            macros["sugar_g"] = value
            sugar_found = True
        elif nutrient_id == 269 and not sugar_found:  # Sugars, total (fallback)
            macros["sugar_g"] = value
            sugar_found = True
        
        # Fallback to name matching if IDs don't work
        elif not sugar_found and "sugar" in nutrient_name and "added" not in nutrient_name:
            macros["sugar_g"] = value
            sugar_found = True
    
    # CRITICAL: If sugar not found or is zero, estimate it
    if not sugar_found or macros["sugar_g"] <= 0.0:
        if macros["carbs_g"] > 0:
            macros["sugar_g"] = estimate_sugar_from_food_type(food_description, macros["carbs_g"])
        else:
            # Even if no carbs, give a minimal sugar value
            macros["sugar_g"] = 0.5
    
    return macros

def convert_to_grams(quantity, unit):
    """Convert quantity and unit to grams"""
    unit_lower = unit.lower()
    
    # Handle special cases
    if unit_lower in ("serving", "servings"):
        return quantity * 100.0
    
    multiplier = UNIT_TO_GRAMS.get(unit_lower, 100.0)  # Default to 100g if unknown
    return quantity * multiplier

def scale_macros(macros_map, grams, base=100.0):
    """Scale macros from base grams to target grams"""
    scaled = {}
    factor = grams / base
    
    for key, value in macros_map.items():
        scaled[key] = round(value * factor, 2)
    
    return scaled

def _normalize_macros_map(macros):
    """Ensure all macro keys exist and are properly formatted"""
    normalized = {
        "calories": round(macros.get("calories", 0.0), 2),
        "protein_g": round(macros.get("protein_g", 0.0), 2),
        "carbs_g": round(macros.get("carbs_g", 0.0), 2),
        "fat_g": round(macros.get("fat_g", 0.0), 2),
        "fiber_g": round(macros.get("fiber_g", 0.0), 2),
        "sugar_g": round(macros.get("sugar_g", 0.1), 2)  # Minimum 0.1g if missing
    }
    return normalized

def get_nutrition_for_item(item):
    """
    Get nutrition information for a food item
    item: dict with keys 'ingredient', 'quantity', 'unit'
    """
    ingredient = item.get("ingredient", "")
    quantity = item.get("quantity", 1.0)
    unit = item.get("unit", "serving")
    
    if not ingredient:
        return {"error": "No ingredient specified"}
    
    # Search for the food
    search_results = search_food(ingredient)
    foods = search_results.get("foods", [])
    
    if not foods:
        return {"error": f"No USDA match found for '{ingredient}'"}
    
    # Take the best match (first result)
    best_match = foods[0]
    
    # Extract macros (nutrients are per 100g in USDA data)
    macros_per_100g = extract_core_macros(best_match)
    
    # Convert user's quantity to grams
    total_grams = convert_to_grams(quantity, unit)
    
    # Scale macros to actual quantity
    scaled_macros = scale_macros(macros_per_100g, total_grams, base=100.0)
    
    # Calculate match score (simplified)
    ingredient_lower = ingredient.lower()
    description_lower = best_match.get("description", "").lower()
    score = SequenceMatcher(None, ingredient_lower, description_lower).ratio()
    
    return {
        "candidate": best_match.get("description"),
        "grams": total_grams,
        "macros": _normalize_macros_map(scaled_macros),
        "score": round(score, 3)
    }
