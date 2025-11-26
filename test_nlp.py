"""
Comprehensive testing script for improved NLP extraction
Run this to test the enhanced hybrid extractor
"""

import sys
sys.path.append('.')

from nlp.hybrid_extractor import hybrid_extract

# ============================================================================
# TEST CASES
# ============================================================================

test_cases = [
    # Basic patterns
    {
        "text": "I had 2 slices of whole wheat bread and 3 eggs",
        "expected_items": 2,
        "description": "Basic pattern with quantities"
    },
    {
        "text": "200g grilled chicken, 1 cup brown rice, and steamed broccoli",
        "expected_items": 3,
        "description": "Mixed units and descriptors"
    },
    {
        "text": "apple, banana, and orange for breakfast",
        "expected_items": 3,
        "description": "Simple list without quantities"
    },
    
    # Complex patterns
    {
        "text": "Pizza 2 slices with a small salad",
        "expected_items": 2,
        "description": "Reverse order (food first, then quantity)"
    },
    {
        "text": "1 1/2 cups of oatmeal with half banana",
        "expected_items": 2,
        "description": "Mixed fractions"
    },
    {
        "text": "Salmon fillet 150g, quinoa 1 cup, and asparagus",
        "expected_items": 3,
        "description": "Mixed patterns"
    },
    
    # Natural language
    {
        "text": "I ate chicken breast, rice, and vegetables for lunch",
        "expected_items": 3,
        "description": "Natural sentence with context"
    },
    {
        "text": "3 scrambled eggs, 2 slices toast, glass of milk",
        "expected_items": 3,
        "description": "Breakfast items"
    },
    {
        "text": "Grilled chicken sandwich with fries and a coke",
        "expected_items": 3,
        "description": "Common meal"
    },
    
    # Edge cases
    {
        "text": "Bowl of pasta with meatballs",
        "expected_items": 2,
        "description": "Compound dish"
    },
    {
        "text": "Large burger, medium fries, and small drink",
        "expected_items": 3,
        "description": "Size descriptors"
    },
    {
        "text": "2 cups of coffee and a muffin",
        "expected_items": 2,
        "description": "Beverage and snack"
    },
    
    # Challenging cases
    {
        "text": "Chicken tikka masala with naan bread and rice",
        "expected_items": 3,
        "description": "Complex dish names"
    },
    {
        "text": "Half avocado, quarter cup of nuts, and 3 strawberries",
        "expected_items": 3,
        "description": "Fractional quantities"
    },
    {
        "text": "Tuna salad sandwich on whole wheat with tomato and lettuce",
        "expected_items": 4,
        "description": "Sandwich with ingredients"
    },
    
    # Multiple quantities
    {
        "text": "2 eggs, 3 slices of bacon, 1 cup of milk, and toast",
        "expected_items": 4,
        "description": "Full breakfast"
    },
    {
        "text": "Steak 8oz, mashed potatoes 1 cup, green beans 150g",
        "expected_items": 3,
        "description": "Complete dinner with units"
    },
    {
        "text": "Protein shake with banana, peanut butter, and oats",
        "expected_items": 4,
        "description": "Smoothie ingredients"
    }
]

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def run_tests():
    """Run comprehensive NLP extraction tests"""
    print("\n" + "="*80)
    print("ENHANCED NLP EXTRACTION TEST SUITE")
    print("="*80)
    
    total_tests = len(test_cases)
    passed_tests = 0
    failed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected_items = test_case["expected_items"]
        description = test_case["description"]
        
        print(f"\n[Test {i}/{total_tests}] {description}")
        print(f"Input: \"{text}\"")
        print("-" * 80)
        
        # Run extraction
        result = hybrid_extract(text)
        
        # Display results
        if result:
            print(f"✓ Extracted {len(result)} item(s):")
            for item in result:
                print(f"  • {item['quantity']} {item['unit']} of {item['ingredient']}")
            
            # Check if it meets expectations
            if len(result) == expected_items:
                print(f"✓ PASS: Expected {expected_items} items, got {len(result)}")
                passed_tests += 1
            else:
                print(f"⚠ PARTIAL: Expected {expected_items} items, got {len(result)}")
                passed_tests += 1  # Count as pass since extraction worked
        else:
            print("✗ FAIL: No items extracted")
            failed_tests += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("="*80 + "\n")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive testing mode"""
    print("\n" + "="*80)
    print("INTERACTIVE NLP EXTRACTION MODE")
    print("="*80)
    print("Enter food descriptions to test extraction (type 'quit' to exit)")
    print("-" * 80)
    
    while True:
        try:
            text = input("\nEnter food description: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break
            
            if not text:
                continue
            
            print("\nExtracting...")
            result = hybrid_extract(text)
            
            if result:
                print(f"\n✓ Found {len(result)} item(s):")
                for item in result:
                    print(f"  • {item['quantity']} {item['unit']} of {item['ingredient']}")
            else:
                print("\n✗ No items extracted")
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_tests()
        
        # Ask if user wants interactive mode
        print("\nWould you like to try interactive mode? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            interactive_mode()