import re
from word2number import w2n

# ============================================================================
# ENHANCED UNIT NORMALIZATION
# ============================================================================

UNIT_ALIASES = {
    # Weight
    "g": "g", "gram": "g", "grams": "g", "gm": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    
    # Volume
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "cup": "cup", "cups": "cup", "c": "cup",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp", "tbs": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "glass": "glass", "glasses": "glass",
    "bottle": "bottle", "bottles": "bottle",
    "can": "can", "cans": "can",
    
    # Count/Serving
    "slice": "slice", "slices": "slice",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece",
    "serving": "serving", "servings": "serving", "serve": "serving",
    "portion": "portion", "portions": "portion",
    "bowl": "bowl", "bowls": "bowl",
    "plate": "plate", "plates": "plate",
    
    # Size descriptors (treated as units)
    "small": "small", "medium": "medium", "large": "large",
    "whole": "whole", "half": "half", "quarter": "quarter"
}

# ============================================================================
# COMPREHENSIVE FOOD DATABASE
# ============================================================================

FOOD_DATABASE = {
    # Proteins - Meat
    "chicken": ["chicken", "chicken breast", "chicken thigh", "chicken leg", "chicken wing",
                "grilled chicken", "fried chicken", "roasted chicken", "baked chicken",
                "chicken meat", "poultry"],
    "beef": ["beef", "steak", "ground beef", "beef patty", "roast beef", "beef steak",
             "sirloin", "ribeye", "t-bone", "filet mignon", "beef mince"],
    "pork": ["pork", "pork chop", "bacon", "ham", "pork loin", "pork ribs",
             "pulled pork", "pork belly", "sausage", "hot dog"],
    "lamb": ["lamb", "lamb chop", "mutton"],
    "turkey": ["turkey", "turkey breast", "ground turkey"],
    
    # Proteins - Seafood
    "fish": ["fish", "fish fillet"],
    "salmon": ["salmon", "salmon fillet", "smoked salmon"],
    "tuna": ["tuna", "tuna steak", "canned tuna"],
    "shrimp": ["shrimp", "prawns", "prawn"],
    "cod": ["cod", "cod fillet"],
    "tilapia": ["tilapia"],
    "mackerel": ["mackerel"],
    
    # Proteins - Eggs & Dairy
    "egg": ["egg", "eggs", "scrambled eggs", "fried egg", "boiled egg",
            "poached egg", "omelette", "omelet"],
    "cheese": ["cheese", "cheddar", "mozzarella", "parmesan", "swiss cheese",
               "feta", "cottage cheese", "cream cheese"],
    "milk": ["milk", "whole milk", "skim milk", "2% milk", "almond milk",
             "soy milk", "oat milk", "coconut milk"],
    "yogurt": ["yogurt", "yoghurt", "greek yogurt", "frozen yogurt"],
    
    # Proteins - Plant-based
    "tofu": ["tofu", "bean curd"],
    "beans": ["beans", "black beans", "kidney beans", "pinto beans", "baked beans"],
    "lentils": ["lentils", "lentil"],
    "chickpeas": ["chickpeas", "garbanzo beans", "chickpea"],
    
    # Grains & Carbs
    "rice": ["rice", "white rice", "brown rice", "jasmine rice", "basmati rice",
             "fried rice", "steamed rice", "wild rice"],
    "bread": ["bread", "toast", "whole wheat bread", "white bread", "wheat bread",
              "sourdough", "rye bread", "multigrain bread", "bagel", "roll", "bun"],
    "pasta": ["pasta", "spaghetti", "noodles", "macaroni", "penne", "fettuccine",
              "linguine", "ravioli", "lasagna", "angel hair"],
    "potato": ["potato", "potatoes", "mashed potato", "baked potato", "roasted potato",
               "french fries", "fries", "chips", "hash browns", "tater tots"],
    "sweet potato": ["sweet potato", "sweet potatoes", "yam", "yams"],
    "quinoa": ["quinoa"],
    "oats": ["oats", "oatmeal", "porridge", "rolled oats"],
    "cereal": ["cereal", "cornflakes", "granola", "muesli"],
    "corn": ["corn", "maize", "corn on the cob", "sweet corn"],
    "tortilla": ["tortilla", "tortillas", "wrap", "wraps"],
    
    # Fruits
    "apple": ["apple", "apples", "green apple", "red apple"],
    "banana": ["banana", "bananas"],
    "orange": ["orange", "oranges", "mandarin", "tangerine", "clementine"],
    "strawberry": ["strawberry", "strawberries"],
    "blueberry": ["blueberry", "blueberries"],
    "raspberry": ["raspberry", "raspberries"],
    "grape": ["grape", "grapes"],
    "mango": ["mango", "mangoes"],
    "pineapple": ["pineapple"],
    "watermelon": ["watermelon"],
    "peach": ["peach", "peaches"],
    "pear": ["pear", "pears"],
    "plum": ["plum", "plums"],
    "avocado": ["avocado", "avocados"],
    "kiwi": ["kiwi", "kiwifruit"],
    
    # Vegetables
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "tomato": ["tomato", "tomatoes", "cherry tomato"],
    "lettuce": ["lettuce", "salad greens", "mixed greens"],
    "spinach": ["spinach"],
    "kale": ["kale"],
    "cucumber": ["cucumber", "cucumbers"],
    "bell pepper": ["bell pepper", "peppers", "capsicum", "red pepper", "green pepper"],
    "onion": ["onion", "onions", "red onion", "white onion"],
    "garlic": ["garlic", "garlic clove"],
    "mushroom": ["mushroom", "mushrooms", "button mushroom"],
    "zucchini": ["zucchini", "courgette"],
    "eggplant": ["eggplant", "aubergine"],
    "cauliflower": ["cauliflower"],
    "celery": ["celery"],
    "asparagus": ["asparagus"],
    
    # Nuts & Seeds
    "nuts": ["nuts", "mixed nuts"],
    "almonds": ["almonds", "almond"],
    "peanuts": ["peanuts", "peanut"],
    "cashews": ["cashews", "cashew"],
    "walnuts": ["walnuts", "walnut"],
    
    # Common Dishes
    "pizza": ["pizza", "pepperoni pizza", "cheese pizza"],
    "burger": ["burger", "hamburger", "cheeseburger"],
    "sandwich": ["sandwich", "sub", "panini", "hoagie"],
    "salad": ["salad", "caesar salad", "greek salad", "garden salad"],
    "soup": ["soup", "broth", "chicken soup", "tomato soup"],
    "burrito": ["burrito", "burrito bowl"],
    "taco": ["taco", "tacos"],
    "nachos": ["nachos"],
    
    # Snacks & Desserts
    "cookie": ["cookie", "cookies", "biscuit", "biscuits"],
    "cake": ["cake", "chocolate cake"],
    "ice cream": ["ice cream", "icecream"],
    "chocolate": ["chocolate", "chocolate bar"],
    
    # Beverages (for context)
    "coffee": ["coffee", "espresso", "latte", "cappuccino"],
    "tea": ["tea", "green tea", "black tea"],
    "juice": ["juice", "orange juice", "apple juice"],
    "soda": ["soda", "coke", "pepsi", "soft drink"],
    "water": ["water"]
}

# Create reverse lookup dictionary
FOOD_LOOKUP = {}
for category, variations in FOOD_DATABASE.items():
    for variation in variations:
        FOOD_LOOKUP[variation.lower()] = category

# ============================================================================
# ENHANCED NUMBER PARSING
# ============================================================================

def parse_number(qty_str):
    """Convert quantity string to float with comprehensive handling"""
    if not qty_str:
        return 1.0
    
    qty_str = qty_str.strip().lower()
    
    # Direct float conversion
    try:
        return float(qty_str)
    except ValueError:
        pass
    
    # Handle fractions (1/2, 3/4, etc.)
    if "/" in qty_str:
        try:
            parts = qty_str.split("/")
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError, IndexError):
            pass
    
    # Handle mixed fractions (1 1/2, 2 3/4, etc.)
    mixed_match = re.match(r'(\d+)\s+(\d+)/(\d+)', qty_str)
    if mixed_match:
        try:
            whole, numerator, denominator = mixed_match.groups()
            return float(whole) + (float(numerator) / float(denominator))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Word to number mappings
    word_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "half": 0.5, "quarter": 0.25, "third": 0.33,
        "a": 1, "an": 1, "couple": 2, "few": 3, "several": 4, "dozen": 12
    }
    
    if qty_str in word_numbers:
        return float(word_numbers[qty_str])
    
    # Try word2number library
    try:
        return float(w2n.word_to_num(qty_str))
    except (ValueError, AttributeError):
        pass
    
    # Default to 1
    return 1.0

# ============================================================================
# UNIT NORMALIZATION
# ============================================================================

def normalize_unit(unit):
    """Normalize unit to standard form"""
    if not unit:
        return "serving"
    
    unit_lower = unit.lower().strip()
    return UNIT_ALIASES.get(unit_lower, unit_lower)

# ============================================================================
# FOOD IDENTIFICATION WITH FUZZY MATCHING
# ============================================================================

def identify_food(text):
    """Identify food item from text using intelligent matching"""
    text_lower = text.lower().strip()
    
    # Remove common descriptors that don't affect food identity
    descriptors = [
        "fresh", "raw", "cooked", "grilled", "fried", "baked", "roasted",
        "boiled", "steamed", "sauteed", "organic", "frozen", "canned",
        "dried", "hot", "cold", "warm", "spicy", "sweet", "sour", "salty",
        "tender", "crispy", "juicy", "lean", "extra", "super", "best"
    ]
    
    for descriptor in descriptors:
        text_lower = re.sub(rf'\b{descriptor}\b', '', text_lower).strip()
    
    # Direct exact match
    if text_lower in FOOD_LOOKUP:
        return FOOD_LOOKUP[text_lower]
    
    # Check for partial matches (longer matches first for better accuracy)
    sorted_foods = sorted(FOOD_LOOKUP.items(), key=lambda x: len(x[0]), reverse=True)
    
    for food_variant, food_name in sorted_foods:
        # Full word match
        if re.search(rf'\b{re.escape(food_variant)}\b', text_lower):
            return food_name
    
    # Substring match as fallback
    for food_variant, food_name in sorted_foods:
        if food_variant in text_lower or text_lower in food_variant:
            return food_name
    
    # Return cleaned text if no match
    return text_lower

# ============================================================================
# INGREDIENT CLEANING
# ============================================================================

def clean_ingredient(ingredient):
    """Clean and normalize ingredient name"""
    if not ingredient:
        return ""
    
    # Remove extra whitespace
    ingredient = re.sub(r'\s+', ' ', ingredient.strip())
    
    # Remove articles and common prefixes
    ingredient = re.sub(r'^(the|a|an|some|my|your|our)\s+', '', ingredient, flags=re.I)
    
    # Remove trailing punctuation
    ingredient = re.sub(r'[.,;!?]+$', '', ingredient)
    
    # Remove measurement-related words that might be caught
    ingredient = re.sub(r'\b(of|about|approximately|roughly|around)\b', '', ingredient, flags=re.I)
    
    return ingredient.strip()

# ============================================================================
# ENHANCED PATTERN MATCHING
# ============================================================================

# Comprehensive regex patterns for different input formats
PATTERNS = [
    # Pattern 1: Quantity + Unit + "of" + Food
    # Examples: "2 cups of rice", "100g of chicken", "half cup of milk"
    re.compile(
        r'(?P<qty>[\d.]+(?:\s+[\d.]+/[\d.]+)?|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|half|quarter|third|couple|few|several|a|an)\s+'
        r'(?P<unit>g|grams?|kg|kilograms?|mg|oz|ounces?|lbs?|pounds?|ml|l|liters?|litres?|'
        r'cups?|c|tbsp|tablespoons?|tsp|teaspoons?|slices?|pieces?|pcs?|servings?|'
        r'portions?|bowls?|plates?|glasses?|bottles?|cans?|small|medium|large)\s+'
        r'(?:of\s+)?'
        r'(?P<ingredient>[a-zA-Z][\w\s]+?)(?=\s*(?:and|with|,|\.|;|for|$))',
        re.I
    ),
    
    # Pattern 2: Quantity + Food + Unit
    # Examples: "2 apples", "chicken breast 200g", "banana medium"
    re.compile(
        r'(?P<qty>[\d.]+(?:\s+[\d.]+/[\d.]+)?|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'half|quarter|couple|few|a|an)\s+'
        r'(?P<ingredient>[a-zA-Z][\w\s]+?)\s*'
        r'(?P<unit>g|grams?|kg|oz|lbs?|ml|l|cups?|slices?|pieces?|servings?|small|medium|large)?'
        r'(?=\s*(?:and|with|,|\.|;|for|$))',
        re.I
    ),
    
    # Pattern 3: Food + Quantity + Unit
    # Examples: "rice 2 cups", "chicken 150g", "milk 1 glass"
    re.compile(
        r'(?P<ingredient>[a-zA-Z][\w\s]+?)\s+'
        r'(?P<qty>[\d.]+(?:\s+[\d.]+/[\d.]+)?|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'half|quarter|couple|few)\s*'
        r'(?P<unit>g|grams?|kg|oz|lbs?|ml|l|cups?|slices?|pieces?|servings?)?'
        r'(?=\s*(?:and|with|,|\.|;|for|$))',
        re.I
    ),
    
    # Pattern 4: Just food (no explicit quantity or unit)
    # Examples: "apple", "chicken breast", "salad"
    re.compile(
        r'(?P<ingredient>[a-zA-Z][\w\s]{2,})(?=\s*(?:and|with|,|\.|;|for|$))',
        re.I
    )
]

# ============================================================================
# FOOD VALIDATION
# ============================================================================

def is_likely_food(ingredient):
    """Check if ingredient is likely a food item"""
    if not ingredient or len(ingredient) < 2:
        return False
    
    ingredient_lower = ingredient.lower()
    
    # Check against known foods
    if ingredient_lower in FOOD_LOOKUP:
        return True
    
    # Check for partial matches
    for food_key in FOOD_LOOKUP.keys():
        if food_key in ingredient_lower or ingredient_lower in food_key:
            return True
    
    # Additional food indicators
    food_indicators = [
        "meat", "chicken", "beef", "pork", "fish", "seafood",
        "bread", "rice", "pasta", "grain", "cereal",
        "fruit", "berry", "apple", "banana", "orange",
        "vegetable", "veggie", "salad", "greens",
        "cheese", "milk", "dairy", "yogurt",
        "egg", "tofu", "bean", "nut"
    ]
    
    for indicator in food_indicators:
        if indicator in ingredient_lower:
            return True
    
    # Non-food keywords (blacklist)
    non_food_keywords = [
        "breakfast", "lunch", "dinner", "meal", "snack",
        "plate", "bowl", "glass", "cup", "serving",
        "had", "ate", "consumed", "enjoyed",
        "today", "yesterday", "morning", "afternoon", "evening"
    ]
    
    for non_food in non_food_keywords:
        if ingredient_lower == non_food:
            return False
    
    # If it has reasonable length and only letters/spaces
    if 2 <= len(ingredient_lower) <= 50 and re.match(r'^[a-z\s\-]+$', ingredient_lower):
        return True
    
    return False

# ============================================================================
# CLAUSE PARSING
# ============================================================================

def parse_clause(clause):
    """Parse a single clause to extract ingredient, quantity, and unit"""
    clause = clause.strip()
    if not clause or len(clause) < 2:
        return None
    
    # Clean up clause - remove common prefixes
    prefixes = [
        r'^(i\s+had|i\s+ate|i\s+consumed|i\s+enjoyed|we\s+ordered|'
        r'lunch\s+was|breakfast\s+was|dinner\s+was|'
        r'for\s+breakfast|for\s+lunch|for\s+dinner|for\s+snack)\s+',
        r'^(a|an|some|the|my|our)\s+'
    ]
    
    for prefix_pattern in prefixes:
        clause = re.sub(prefix_pattern, '', clause, flags=re.I)
    
    best_match = None
    best_score = 0
    
    # Try each pattern
    for i, pattern in enumerate(PATTERNS):
        match = pattern.search(clause)
        if not match:
            continue
        
        groups = match.groupdict()
        ingredient = clean_ingredient(groups.get("ingredient", ""))
        qty_str = groups.get("qty", "1")
        unit = groups.get("unit", "serving")
        
        # Skip if no valid ingredient
        if not ingredient or len(ingredient) < 2:
            continue
        
        # Score this match
        score = 0
        
        # Food likelihood (most important)
        if is_likely_food(ingredient):
            score += 5
        
        # Has explicit quantity (not just "1" or "a")
        if qty_str and qty_str not in ["1", "a", "an"]:
            score += 3
        
        # Has explicit unit (not just "serving")
        if unit and unit not in ["serving", "servings"]:
            score += 2
        
        # Pattern preference (earlier patterns are more specific)
        score += (4 - i) * 0.5
        
        if score > best_score:
            best_score = score
            quantity = parse_number(qty_str)
            unit_norm = normalize_unit(unit)
            
            # Identify the actual food
            food_name = identify_food(ingredient)
            
            best_match = {
                "ingredient": food_name,
                "quantity": quantity,
                "unit": unit_norm
            }
    
    return best_match

# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def rule_based_extraction(text):
    """Extract food items from text using enhanced rule-based approach"""
    if not text or not text.strip():
        return []
    
    # Normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split on common separators
    separators = r'(?:,\s*|\s+and\s+|\s+with\s+|;\s*|\s+\+\s+)'
    parts = re.split(separators, text, flags=re.I)
    
    items = []
    seen_ingredients = set()
    
    for part in parts:
        part = part.strip()
        if not part or len(part) < 2:
            continue
        
        # Skip common non-food phrases
        skip_phrases = [
            'for breakfast', 'for lunch', 'for dinner', 'for snack',
            'i had', 'i ate', 'i consumed', 'we ordered',
            'this morning', 'this afternoon', 'this evening'
        ]
        
        if any(phrase in part.lower() for phrase in skip_phrases):
            # Try to extract food after these phrases
            for phrase in skip_phrases:
                if phrase in part.lower():
                    part = part.lower().split(phrase)[-1].strip()
                    break
        
        parsed = parse_clause(part)
        if parsed and parsed["ingredient"] not in seen_ingredients:
            items.append(parsed)
            seen_ingredients.add(parsed["ingredient"])
    
    # If no items found with splitting, try the whole text
    if not items:
        parsed = parse_clause(text)
        if parsed:
            items.append(parsed)
    
    return items