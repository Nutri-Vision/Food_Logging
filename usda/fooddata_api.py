import requests
import json
import os
from difflib import SequenceMatcher

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# USDA FoodData Central API configuration
USDA_API_KEY = "ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk"  # Get from https://fdc.nal.usda.gov/api-guide.html
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# ---------------------------------------------------------
# UNIT CONVERSION LOGIC
# ---------------------------------------------------------
# Note: "cup" to grams depends on density. These are averages.
UNIT_TO_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "pound": 453.59, "pounds": 453.59,
    "cup": 240.0, "cups": 240.0, # Water density assumption
    "ml": 1.0, "l": 1000.0,
    "tbsp": 15.0, "tablespoon": 15.0,
    "tsp": 5.0, "teaspoon": 5.0,
    "slice": 35.0,  # Average bread slice
    "piece": 100.0, # Generic medium piece
    "serving": 100.0,
    "servings": 100.0,
    "glass": 240.0,
    "bottle": 500.0, # Common soda size
    "can": 355.0,    # Common soda can
}

def search_food(query, limit=3):
    """Search for foods in USDA database"""
    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "query": query,
        "pageSize": limit,
        "api_key": USDA_API_KEY,
        # 'Survey (FNDDS)' usually has better prepared food data (pizzas, burgers)
        # 'Foundation' is better for raw ingredients (chicken breast, egg)
        "dataType": ["Foundation", "Survey (FNDDS)", "SR Legacy"] 
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"USDA API error: {e}")
        return {"foods": []}

def _get_nutrient_value(nutrient_entry):
    """Safely extract value from weird USDA JSON structure"""
    if not nutrient_entry: return 0.0
    # USDA sometimes nests value, sometimes it's direct
    if isinstance(nutrient_entry, (int, float)): return float(nutrient_entry)
    if isinstance(nutrient_entry, dict):
        return float(nutrient_entry.get("value", 0.0))
    return 0.0

def extract_core_macros(food_detail):
    """Extracts macros including Fiber and Sugar"""
    # Initialize with 0
    macros = {
        "calories": 0.0, 
        "protein_g": 0.0, 
        "carbs_g": 0.0, 
        "fat_g": 0.0, 
        "fiber_g": 0.0, 
        "sugar_g": 0.0
    }
    
    nutrients = food_detail.get("foodNutrients", [])
    
    for n in nutrients:
        n_id = n.get("nutrientId")
        val = _get_nutrient_value(n.get("value", n)) # Handle flat or nested structure

        # --- MACRO MAPPING ---
        if n_id == 1008: macros["calories"] = val    # Energy (kcal)
        elif n_id == 1003: macros["protein_g"] = val # Protein
        elif n_id == 1005: macros["carbs_g"] = val   # Carbs
        elif n_id == 1004: macros["fat_g"] = val     # Fat
        elif n_id == 291:  macros["fiber_g"] = val   # Fiber (Total Dietary)
        elif n_id == 269:  macros["sugar_g"] = val   # Sugars (Total)
    
    return macros

def convert_to_grams(quantity, unit):
    """Convert user unit to grams. Defaults to 100g if unit unknown."""
    unit_lower = unit.lower().rstrip('s') # remove plural 's'
    
    # Direct lookup
    if unit_lower in UNIT_TO_GRAMS:
        return quantity * UNIT_TO_GRAMS[unit_lower]
    
    # Partial match (e.g. "teaspoons" -> "teaspoon")
    for key in UNIT_TO_GRAMS:
        if key in unit_lower:
             return quantity * UNIT_TO_GRAMS[key]
             
    # FALLBACK: If unit is unknown (e.g., "bowl", "plate"), 
    # assume it's a "serving" (100g) rather than 1g.
    print(f"Warning: Unit '{unit}' not found. Assuming standard serving (100g).")
    return quantity * 100.0

def scale_macros(macros_100g, target_grams):
    """Scales macros from 100g base to target weight"""
    factor = target_grams / 100.0
    return {k: round(v * factor, 1) for k, v in macros_100g.items()}

def get_nutrition_for_item(item):
    """
    Main handler
    item: {'ingredient': 'chicken', 'quantity': 4, 'unit': 'piece'}
    """
    query = item.get("ingredient", "")
    qty = float(item.get("quantity", 1.0))
    unit = item.get("unit", "serving") # Default to serving if missing

    # 1. Search USDA
    data = search_food(query)
    foods = data.get("foods", [])
    
    if not foods:
        return {"error": "Food not found"}

    # 2. Get Best Match
    best_match = foods[0]
    
    # 3. Extract Macros (Per 100g)
    macros_100g = extract_core_macros(best_match)
    
    # 4. Calculate Total Grams
    total_grams = convert_to_grams(qty, unit)
    
    # 5. Scale
    final_macros = scale_macros(macros_100g, total_grams)
    
    return {
        "food_name": best_match.get("description"),
        "logged_qty": qty,
        "logged_unit": unit,
        "total_grams": total_grams,
        "macros": final_macros
    }

# --- TEST ---
# item = {"ingredient": "Coke", "quantity": 1, "unit": "bottle"}
# print(json.dumps(get_nutrition_for_item(item), indent=2))
