"""
Enhanced simple extractor for when enhanced NLP is unavailable
Uses basic pattern matching with improved accuracy
"""
import re

# Comprehensive food list
COMMON_FOODS = {
    # Proteins
    "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp",
    "egg", "eggs", "tofu", "bacon", "ham", "turkey",
    
    # Dairy
    "milk", "cheese", "yogurt", "butter", "cream",
    
    # Grains
    "rice", "bread", "pasta", "noodles", "quinoa", "oats", "cereal",
    "toast", "bagel", "tortilla", "wrap",
    
    # Vegetables
    "broccoli", "carrot", "carrots", "tomato", "tomatoes", "lettuce",
    "spinach", "cucumber", "pepper", "peppers", "onion", "garlic",
    "potato", "potatoes", "corn", "peas", "beans",
    
    # Fruits
    "apple", "apples", "banana", "bananas", "orange", "oranges",
    "strawberry", "strawberries", "grape", "grapes", "mango",
    "pineapple", "watermelon", "peach", "pear", "avocado",
    
    # Common dishes
    "pizza", "burger", "sandwich", "salad", "soup", "burrito", "taco",
    
    # Nuts
    "nuts", "almonds", "peanuts", "walnuts", "cashews"
}

# Unit keywords
UNITS = {
    "g", "grams", "kg", "oz", "ounces", "lb", "pounds",
    "ml", "l", "liters", "cup", "cups", "tbsp", "tsp",
    "slice", "slices", "piece", "pieces", "serving", "servings",
    "bowl", "plate", "glass", "small", "medium", "large"
}

def parse_number(num_str):
    """Parse number string to float"""
    try:
        return float(num_str)
    except ValueError:
        # Word numbers
        word_nums = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "half": 0.5, "quarter": 0.25, "a": 1, "an": 1
        }
        return word_nums.get(num_str.lower(), 1.0)

def simple_extract(text: str) -> list:
    """
    Simple extraction using basic pattern matching
    Fallback when enhanced NLP is unavailable
    """
    if not text:
        return []
    
    text = text.lower()
    results = []
    
    # Pattern 1: Number + Food (e.g., "2 apples", "3 eggs")
    pattern1 = r'(\d+(?:\.\d+)?|one|two|three|four|five|half|a|an)\s+([a-z]+)'
    matches1 = re.findall(pattern1, text)
    
    for qty_str, food in matches1:
        if food in COMMON_FOODS:
            quantity = parse_number(qty_str)
            results.append({
                "ingredient": food,
                "quantity": quantity,
                "unit": "servings"
            })
    
    # Pattern 2: Number + Unit + Food (e.g., "100g chicken", "1 cup rice")
    pattern2 = r'(\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:of\s+)?([a-z]+)'
    matches2 = re.findall(pattern2, text)
    
    for qty_str, unit, food in matches2:
        if food in COMMON_FOODS and unit in UNITS:
            quantity = parse_number(qty_str)
            results.append({
                "ingredient": food,
                "quantity": quantity,
                "unit": unit
            })
    
    # Pattern 3: Food + Number + Unit (e.g., "chicken 200g", "rice 2 cups")
    pattern3 = r'([a-z]+)\s+(\d+(?:\.\d+)?)\s*([a-z]+)?'
    matches3 = re.findall(pattern3, text)
    
    for food, qty_str, unit in matches3:
        if food in COMMON_FOODS:
            quantity = parse_number(qty_str)
            unit = unit if unit in UNITS else "servings"
            results.append({
                "ingredient": food,
                "quantity": quantity,
                "unit": unit
            })
    
    # If no matches with patterns, look for just food names
    if not results:
        for food in COMMON_FOODS:
            if food in text:
                results.append({
                    "ingredient": food,
                    "quantity": 1.0,
                    "unit": "servings"
                })
    
    # Remove duplicates
    seen = set()
    unique_results = []
    for item in results:
        key = (item["ingredient"], item["unit"])
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    
    # If still no results, return generic item
    return unique_results if unique_results else [{
        "ingredient": "mixed food",
        "quantity": 1.0,
        "unit": "servings"
    }]