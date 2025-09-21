import re
from word2number import w2n

# Expanded unit normalization map
UNIT_ALIASES = {
    "grams": "g", "gram": "g", "g": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "l": "l", "litre": "l", "liter": "l",
    "cup": "cup", "cups": "cup",
    "slice": "slice", "slices": "slice",
    "piece": "piece", "pieces": "piece",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "glass": "glass", "glasses": "glass",
    "serving": "serving", "servings": "serving",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "pound": "lb", "pounds": "lb",
    "bowl": "bowl", "bowls": "bowl",
    "plate": "plate", "plates": "plate",
    "portion": "serving", "portions": "serving"
}

# Expanded common foods list
COMMON_FOODS = [
    # Grains and starches
    "rice", "bread", "pasta", "noodles", "quinoa", "oats", "wheat", "barley",
    "potato", "sweet potato", "corn", "tortilla", "bagel", "cereal",
    
    # Proteins
    "chicken", "beef", "pork", "fish", "salmon", "tuna", "egg", "eggs",
    "tofu", "beans", "lentils", "chickpeas", "paneer", "cheese",
    
    # Fruits
    "apple", "banana", "orange", "grapes", "strawberry", "blueberry",
    "mango", "pineapple", "watermelon", "peach", "pear",
    
    # Vegetables
    "broccoli", "spinach", "carrot", "tomato", "onion", "garlic",
    "lettuce", "cucumber", "bell pepper", "mushroom",
    
    # Dairy
    "milk", "yogurt", "butter", "cream",
    
    # Others
    "oil", "salt", "sugar", "honey", "nuts", "almonds", "peanuts"
]

# Improved regex patterns for different input formats
PATTERNS = [
    # Pattern 1: "two slices of bread", "200 g of rice", "1 cup rice"
    re.compile(
        r"(?P<qty>(?:\d+\.?\d*|half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|a|an))\s+(?P<unit>slice|slices|cup|cups|g|gram|grams|kg|ml|l|tbsp|tsp|glass|glasses|bowl|bowls|piece|pieces|serving|servings|oz|ounce|ounces|lb|pound|pounds)s?\s+(?:of\s+)?(?P<ingredient>(?:[a-zA-Z]+(?:\s+[a-zA-Z]+)*))(?=\s*(?:and|,|with|$))",
        flags=re.I
    ),
    
    # Pattern 2: "chicken breast 200g", "rice 2 cups"
    re.compile(
        r"(?P<ingredient>(?:[a-zA-Z]+(?:\s+[a-zA-Z]+)*?))\s+(?P<qty>(?:\d+\.?\d*|half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|a|an))\s*(?P<unit>slice|slices|cup|cups|g|gram|grams|kg|ml|l|tbsp|tsp|glass|glasses|bowl|bowls|piece|pieces|serving|servings|oz|ounce|ounces|lb|pound|pounds)?s?(?=\s*(?:and|,|with|$))",
        flags=re.I
    ),
    
    # Pattern 3: Just quantity + ingredient (no unit): "2 eggs", "3 apples"
    re.compile(
        r"(?P<qty>(?:\d+\.?\d*|half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|a|an))\s+(?P<ingredient>(?:[a-zA-Z]+(?:\s+[a-zA-Z]+)*))(?=\s*(?:and|,|with|$))",
        flags=re.I
    ),
    
    # Pattern 4: Just ingredient name
    re.compile(
        r"(?P<ingredient>(?:[a-zA-Z]+(?:\s+[a-zA-Z]+)*))(?=\s*(?:and|,|with|$|\.|\n))",
        flags=re.I
    )
]

def parse_number(qty_str):
    """Convert quantity string to float"""
    if not qty_str:
        return 1.0
    
    qty_str = qty_str.strip().lower()
    
    try:
        # Try direct float conversion first
        return float(qty_str)
    except ValueError:
        pass
    
    # Handle fractions
    if "/" in qty_str:
        try:
            parts = qty_str.split("/")
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
    
    # Handle word numbers
    try:
        # Common word-to-number mappings
        word_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "half": 0.5, "quarter": 0.25, "a": 1, "an": 1
        }
        
        if qty_str in word_numbers:
            return float(word_numbers[qty_str])
        
        # Try word2number library
        return float(w2n.word_to_num(qty_str))
    except (ValueError, AttributeError):
        pass
    
    # Default to 1 if all parsing fails
    return 1.0

def clean_ingredient(ingredient):
    """Clean and normalize ingredient name"""
    if not ingredient:
        return ""
    
    # Remove extra whitespace and convert to lowercase
    ingredient = re.sub(r'\s+', ' ', ingredient.strip().lower())
    
    # Remove common prefixes/suffixes
    prefixes = ["some", "a", "an", "the", "fresh", "organic", "raw", "cooked"]
    suffixes = ["meat", "fish", "vegetable", "fruit"]
    
    words = ingredient.split()
    
    # Remove prefixes
    while words and words[0] in prefixes:
        words.pop(0)
    
    # Remove suffixes (but keep if it's the only word)
    while len(words) > 1 and words[-1] in suffixes:
        words.pop()
    
    return " ".join(words).strip()

def is_likely_food(ingredient):
    """Check if ingredient is likely a food item"""
    if not ingredient:
        return False
    
    ingredient_lower = ingredient.lower()
    
    # Check against known foods
    for food in COMMON_FOODS:
        if food in ingredient_lower or ingredient_lower in food:
            return True
    
    # Additional heuristics
    food_indicators = [
        "chicken", "beef", "pork", "fish", "meat", "bread", "rice", "pasta",
        "apple", "banana", "fruit", "vegetable", "milk", "cheese", "egg",
        "potato", "tomato", "onion", "garlic", "oil", "butter", "yogurt"
    ]
    
    for indicator in food_indicators:
        if indicator in ingredient_lower:
            return True
    
    # If ingredient has reasonable length and contains only letters/spaces
    if 2 <= len(ingredient_lower) <= 50 and re.match(r'^[a-z\s\-]+$', ingredient_lower):
        return True
    
    return False

def parse_clause(clause):
    """Parse a single clause to extract ingredient, quantity, and unit"""
    clause = clause.strip()
    if not clause:
        return None
    
    # Clean up the clause - remove common non-food words at the beginning
    clause = re.sub(r'^(i\s+had|i\s+ate|we\s+ordered|lunch\s+was|breakfast\s+was|dinner\s+was|for\s+breakfast|for\s+lunch|for\s+dinner)\s+', '', clause, flags=re.I)
    clause = re.sub(r'^(a|an|some|the)\s+', '', clause, flags=re.I)
    
    best_match = None
    best_score = 0
    
    # Try each pattern and pick the best match
    for i, pattern in enumerate(PATTERNS):
        match = pattern.search(clause)
        if match:
            groups = match.groupdict()
            
            ingredient = clean_ingredient(groups.get("ingredient", ""))
            qty_str = groups.get("qty", "1")
            unit = groups.get("unit", "serving")
            
            # Skip if no valid ingredient
            if not ingredient or len(ingredient) < 2:
                continue
            
            # Score this match based on completeness and food likelihood
            score = 0
            if is_likely_food(ingredient):
                score += 3
            if qty_str and qty_str != "1":
                score += 2
            if unit and unit != "serving":
                score += 1
            
            # Pattern preference (earlier patterns are better)
            score += (4 - i) * 0.1
            
            if score > best_score:
                best_score = score
                quantity = parse_number(qty_str)
                unit_norm = UNIT_ALIASES.get(unit.lower() if unit else "serving", 
                                           unit.lower() if unit else "serving")
                
                best_match = {
                    "ingredient": ingredient,
                    "quantity": quantity,
                    "unit": unit_norm
                }
    
    return best_match

def rule_based_extraction(text):
    """Extract food items from text using rule-based approach"""
    if not text:
        return []
    
    # Normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split on common separators, but be more careful
    separators = r'(?:,\s*|\s+and\s+|\s+with\s+|;\s*)'
    parts = re.split(separators, text, flags=re.I)
    
    items = []
    seen_ingredients = set()
    
    for part in parts:
        part = part.strip()
        if not part or len(part) < 2:
            continue
        
        # Skip common non-food phrases
        skip_phrases = ['for breakfast', 'for lunch', 'for dinner', 'i had', 'i ate', 'we ordered', 'lunch was', 'breakfast was']
        if any(phrase in part.lower() for phrase in skip_phrases):
            continue
        
        parsed = parse_clause(part)
        if parsed and parsed["ingredient"] not in seen_ingredients:
            items.append(parsed)
            seen_ingredients.add(parsed["ingredient"])
    
    # If no items found with splitting, try the whole text
    if not items:
        # Try some manual patterns for common cases
        
        # Pattern for "X and Y"
        and_match = re.search(r'(.+?)\s+and\s+(.+)', text, re.I)
        if and_match:
            part1, part2 = and_match.groups()
            for part in [part1.strip(), part2.strip()]:
                parsed = parse_clause(part)
                if parsed and parsed["ingredient"] not in seen_ingredients:
                    items.append(parsed)
                    seen_ingredients.add(parsed["ingredient"])
        
        # If still no items, try the whole text as one item
        if not items:
            parsed = parse_clause(text)
            if parsed:
                items.append(parsed)
    
    return items