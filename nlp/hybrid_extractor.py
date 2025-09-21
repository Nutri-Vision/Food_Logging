from .rules import rule_based_extraction
from .spacy_model import load_trained_model
from difflib import SequenceMatcher
import re  

nlp_model = load_trained_model()  

def spacy_extract(nlp, text):
    try:
        doc = nlp(text)
        results = []
        
        for ent in doc.ents:
            if ent.label_ == "FOOD":
                food_name = ent.text.lower().strip()
                if food_name and len(food_name) > 1:
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

def merge_extractions(rule_items, spacy_foods):
    """Merge rule-based and spaCy extractions intelligently"""
    if not spacy_foods:
        return rule_items
    
    # Clean spaCy foods - remove non-food items and numbers
    clean_spacy_foods = []
    for food in spacy_foods:
        food = food.strip().lower()
        # Skip if it's just a number, unit, or common non-food word
        if (re.match(r'^\d+.*', food) or 
            food in ['cups', 'slices', 'glass', 'bowl', 'tbsp', 'tsp'] or
            len(food) < 3):
            continue
        clean_spacy_foods.append(food)
    
    merged_items = []
    used_spacy_foods = set()
    
    for rule_item in rule_items:
        rule_ingredient = rule_item["ingredient"].lower()
        best_match = None
        best_score = 0.0
        
        # Find the best spaCy match for this rule-based item
        for spacy_food in clean_spacy_foods:
            if spacy_food in used_spacy_foods:
                continue
                
            # Check for exact or substring matches
            if spacy_food == rule_ingredient:
                score = 1.0  # Perfect match
            elif spacy_food in rule_ingredient:
                score = 0.9  # SpaCy food is contained in rule ingredient
            elif rule_ingredient in spacy_food:
                score = 0.8  # Rule ingredient is contained in spaCy food
            else:
                # Use sequence matching for similarity
                score = SequenceMatcher(None, rule_ingredient, spacy_food).ratio()
            
            if score > best_score and score > 0.7:  # Higher threshold
                best_match = spacy_food
                best_score = score
        
        # Use spaCy match if found and it's a good match
        if best_match and best_score > 0.7:
            final_ingredient = best_match
            used_spacy_foods.add(best_match)
        else:
            final_ingredient = rule_ingredient
        
        merged_items.append({
            "ingredient": final_ingredient,
            "quantity": rule_item["quantity"],
            "unit": rule_item["unit"]
        })
    
    # Add any high-quality spaCy foods that weren't matched
    for spacy_food in clean_spacy_foods:
        if spacy_food not in used_spacy_foods:
            # Only add if it's clearly a food item
            food_keywords = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'rice', 'bread', 'pasta', 
                           'apple', 'banana', 'orange', 'milk', 'cheese', 'egg', 'potato', 'tomato', 
                           'broccoli', 'spinach', 'carrot', 'yogurt', 'oats', 'quinoa']
            
            is_clear_food = any(keyword in spacy_food for keyword in food_keywords)
            
            if is_clear_food or len(spacy_food) >= 4:  # Longer words more likely to be real foods
                merged_items.append({
                    "ingredient": spacy_food,
                    "quantity": 1.0,
                    "unit": "serving"
                })
    
    return merged_items

def consolidate_items(items):
    """Consolidate duplicate ingredients by summing quantities"""
    if not items:
        return []
    
    consolidated = {}
    
    for item in items:
        ingredient = item["ingredient"].lower().strip()
        quantity = item.get("quantity", 1.0)
        unit = item.get("unit", "serving")
        
        if ingredient in consolidated:
            # If same unit, add quantities; otherwise, keep separate entries
            existing_unit = consolidated[ingredient]["unit"]
            if existing_unit == unit:
                consolidated[ingredient]["quantity"] += quantity
            else:
                # Create a new key for different units
                key = f"{ingredient}_{unit}"
                consolidated[key] = {
                    "ingredient": ingredient,
                    "quantity": quantity,
                    "unit": unit
                }
        else:
            consolidated[ingredient] = {
                "ingredient": ingredient,
                "quantity": round(quantity, 2),
                "unit": unit
            }
    
    # Convert back to list and clean up keys that were modified for different units
    result = []
    for item in consolidated.values():
        result.append(item)
    
    return result

def hybrid_extract(text):
    """
    Main extraction function that combines rule-based and spaCy approaches
    """
    if not text or not text.strip():
        return []
    
    # Step 1: Always run rule-based extraction (provides quantities and units)
    rule_items = rule_based_extraction(text)
    
    # Step 2: If spaCy model is available, use it to enhance ingredient names
    spacy_foods = []
    if nlp_model:
        try:
            spacy_foods = spacy_extract(nlp_model, text)
            print(f"SpaCy extracted: {spacy_foods}")  # Debug output
        except Exception as e:
            print(f"SpaCy model error: {e}")
            spacy_foods = []
    
    # Step 3: Merge the extractions
    if spacy_foods:
        merged_items = merge_extractions(rule_items, spacy_foods)
    else:
        merged_items = rule_items
    
    # Step 4: Consolidate duplicate ingredients
    final_items = consolidate_items(merged_items)
    
    # Debug output
    print(f"Rule-based extracted: {rule_items}")
    print(f"Final items: {final_items}")
    
    return final_items

# Additional utility function for testing
def test_hybrid_extract(test_cases):
    """Test function for hybrid extraction"""
    print("=== Hybrid Extraction Test Results ===")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{test_text}'")
        result = hybrid_extract(test_text)
        
        if result:
            for item in result:
                print(f"  - {item['quantity']} {item['unit']} of {item['ingredient']}")
        else:
            print("  - No items extracted")
    
    print("\n" + "="*50)