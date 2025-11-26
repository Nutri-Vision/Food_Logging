from .rules import rule_based_extraction, identify_food, is_likely_food
from .spacy_model import load_trained_model
from difflib import SequenceMatcher
import re

# Load spaCy model if available
nlp_model = load_trained_model()

# ============================================================================
# SPACY EXTRACTION
# ============================================================================

def spacy_extract(nlp, text):
    """Extract food entities using spaCy NER"""
    try:
        doc = nlp(text)
        results = []
        
        for ent in doc.ents:
            if ent.label_ == "FOOD":
                food_name = ent.text.lower().strip()
                if food_name and len(food_name) > 1:
                    # Skip if it's just a number or unit
                    if not re.match(r'^\d+', food_name):
                        results.append(food_name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
            if item not in seen:
                seen.add(item)
                unique_results.append(item)
        
        return unique_results
    
    except Exception as e:
        print(f"spaCy extraction error: {e}")
        return []

# ============================================================================
# INTELLIGENT MERGING OF EXTRACTIONS
# ============================================================================

def merge_extractions(rule_items, spacy_foods):
    """Intelligently merge rule-based and spaCy extractions"""
    if not spacy_foods:
        return rule_items
    
    # Clean spaCy foods - remove non-food items
    clean_spacy_foods = []
    for food in spacy_foods:
        food = food.strip().lower()
        
        # Skip if it's just a number, unit, or common non-food word
        skip_words = [
            'cups', 'cup', 'slices', 'slice', 'glass', 'glasses',
            'bowl', 'bowls', 'plate', 'plates', 'tbsp', 'tsp',
            'grams', 'gram', 'ounces', 'ounce', 'serving', 'servings',
            'breakfast', 'lunch', 'dinner', 'meal', 'snack'
        ]
        
        if (re.match(r'^\d+', food) or 
            food in skip_words or 
            len(food) < 3):
            continue
        
        # Check if it's likely a food
        if is_likely_food(food):
            clean_spacy_foods.append(food)
    
    merged_items = []
    used_spacy_foods = set()
    
    # Match rule-based items with spaCy foods
    for rule_item in rule_items:
        rule_ingredient = rule_item["ingredient"].lower()
        best_match = None
        best_score = 0.0
        
        # Find the best spaCy match for this rule-based item
        for spacy_food in clean_spacy_foods:
            if spacy_food in used_spacy_foods:
                continue
            
            # Calculate similarity score
            if spacy_food == rule_ingredient:
                score = 1.0  # Perfect match
            elif spacy_food in rule_ingredient:
                score = 0.9  # SpaCy food is contained in rule ingredient
            elif rule_ingredient in spacy_food:
                score = 0.85  # Rule ingredient is contained in spaCy food
            else:
                # Use sequence matching for similarity
                score = SequenceMatcher(None, rule_ingredient, spacy_food).ratio()
            
            if score > best_score and score > 0.75:  # Threshold for matching
                best_match = spacy_food
                best_score = score
        
        # Use spaCy match if it's a good match, otherwise use rule-based
        if best_match and best_score > 0.75:
            # Use the more specific/complete name
            if len(best_match) > len(rule_ingredient):
                final_ingredient = identify_food(best_match)
            else:
                final_ingredient = rule_ingredient
            used_spacy_foods.add(best_match)
        else:
            final_ingredient = rule_ingredient
        
        merged_items.append({
            "ingredient": final_ingredient,
            "quantity": rule_item["quantity"],
            "unit": rule_item["unit"]
        })
    
    # Add any unused high-quality spaCy foods that weren't matched
    for spacy_food in clean_spacy_foods:
        if spacy_food not in used_spacy_foods:
            # Only add if it's clearly a food item
            identified_food = identify_food(spacy_food)
            
            # Skip if it's generic or already exists
            if identified_food != spacy_food.lower():
                # It was recognized as a known food
                merged_items.append({
                    "ingredient": identified_food,
                    "quantity": 1.0,
                    "unit": "serving"
                })
    
    return merged_items

# ============================================================================
# CONSOLIDATION
# ============================================================================

def consolidate_items(items):
    """Consolidate duplicate ingredients by summing quantities"""
    if not items:
        return []
    
    consolidated = {}
    
    for item in items:
        ingredient = item["ingredient"].lower().strip()
        quantity = item.get("quantity", 1.0)
        unit = item.get("unit", "serving")
        
        # Create key combining ingredient and unit
        key = f"{ingredient}|{unit}"
        
        if key in consolidated:
            # Same ingredient and unit - sum quantities
            consolidated[key]["quantity"] += quantity
        else:
            consolidated[key] = {
                "ingredient": ingredient,
                "quantity": quantity,
                "unit": unit
            }
    
    # Convert back to list and round quantities
    result = []
    for item in consolidated.values():
        item["quantity"] = round(item["quantity"], 2)
        result.append(item)
    
    return result

# ============================================================================
# POST-PROCESSING VALIDATION
# ============================================================================

def validate_and_filter(items):
    """Validate and filter extracted items"""
    valid_items = []
    
    for item in items:
        ingredient = item["ingredient"]
        quantity = item["quantity"]
        
        # Skip if ingredient is too generic or invalid
        generic_terms = [
            "food", "item", "thing", "stuff", "meal",
            "breakfast", "lunch", "dinner", "snack"
        ]
        
        if ingredient.lower() in generic_terms:
            continue
        
        # Skip if quantity is unreasonable
        if quantity <= 0 or quantity > 100:
            continue
        
        # Skip if ingredient is just a number
        if re.match(r'^\d+$', ingredient):
            continue
        
        valid_items.append(item)
    
    return valid_items

# ============================================================================
# MAIN HYBRID EXTRACTION
# ============================================================================

def hybrid_extract(text):
    """
    Main extraction function combining rule-based and spaCy approaches
    with enhanced accuracy and validation
    """
    if not text or not text.strip():
        return []
    
    # Step 1: Run rule-based extraction (provides quantities and units)
    rule_items = rule_based_extraction(text)
    
    # Step 2: If spaCy model is available, use it to enhance extraction
    spacy_foods = []
    if nlp_model:
        try:
            spacy_foods = spacy_extract(nlp_model, text)
            if spacy_foods:
                print(f"[DEBUG] SpaCy extracted: {spacy_foods}")
        except Exception as e:
            print(f"[DEBUG] SpaCy model error: {e}")
            spacy_foods = []
    
    # Step 3: Merge the extractions
    if spacy_foods:
        merged_items = merge_extractions(rule_items, spacy_foods)
    else:
        merged_items = rule_items
    
    # Step 4: Consolidate duplicate ingredients
    consolidated_items = consolidate_items(merged_items)
    
    # Step 5: Validate and filter
    final_items = validate_and_filter(consolidated_items)
    
    # Debug output
    print(f"[DEBUG] Rule-based: {len(rule_items)} items")
    print(f"[DEBUG] After merge: {len(merged_items)} items")
    print(f"[DEBUG] After consolidation: {len(consolidated_items)} items")
    print(f"[DEBUG] Final valid: {len(final_items)} items")
    print(f"[DEBUG] Final items: {final_items}")
    
    return final_items

# ============================================================================
# TESTING UTILITY
# ============================================================================

def test_hybrid_extract(test_cases):
    """Test function for hybrid extraction with detailed output"""
    print("\n" + "="*70)
    print("HYBRID EXTRACTION TEST RESULTS")
    print("="*70)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Input: '{test_text}'")
        print("-" * 70)
        
        result = hybrid_extract(test_text)
        
        if result:
            print(f"Found {len(result)} item(s):")
            for item in result:
                print(f"  ✓ {item['quantity']} {item['unit']} of {item['ingredient']}")
        else:
            print("  ✗ No items extracted")
    
    print("\n" + "="*70)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "I had 2 slices of whole wheat bread and 3 eggs",
        "200g grilled chicken, 1 cup brown rice, and steamed broccoli",
        "apple, banana, and orange for breakfast",
        "Pizza 2 slices with a small salad",
        "1 1/2 cups of oatmeal with half banana",
        "Salmon fillet 150g, quinoa 1 cup, and asparagus",
        "chicken breast, rice, and vegetables",
        "3 scrambled eggs, 2 slices toast, glass of milk",
        "Grilled chicken sandwich with fries",
        "Bowl of pasta with meatballs"
    ]
    
    test_hybrid_extract(test_cases)