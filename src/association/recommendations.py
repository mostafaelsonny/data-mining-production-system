"""
src/association/recommendations.py
==================================
Product Recommendation Engine based on Association Rules
Owner: Marwa
Contract Validator: src.contracts.validate_recommendation_output
"""

import os
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import List, Dict, Any
from src.contracts import validate_recommendation_input, validate_recommendation_output

RULES_PATH = os.path.join(os.path.dirname(__file__), '../../models/association/association_rules.csv')

def recommend_products(selected_items: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Public recommendation function satisfying Contract #3.
    Receives list of selected products, queries rules, eliminates items already in cart,
    and returns ranked recommendations.
    """
    # 1. Enforce Input Contract
    cleaned_items = validate_recommendation_input(selected_items)
    if not cleaned_items:
        return []
        
    selected_set = set(cleaned_items)
    recommendations = {}
    
    # 2. Check if rules file exists
    if os.path.exists(RULES_PATH):
        rules_df = pd.read_csv(RULES_PATH)
        for _, row in rules_df.iterrows():
            antecedents = set([item.strip().upper() for item in str(row['antecedents_str']).split(',')])
            consequents = set([item.strip().upper() for item in str(row['consequents_str']).split(',')])
            
            if antecedents.issubset(selected_set):
                for candidate in consequents:
                    if candidate not in selected_set:
                        lift = float(row.get('lift', 1.0))
                        confidence = float(row.get('confidence', 0.5))
                        support = float(row.get('support', 0.05))
                        
                        if candidate not in recommendations or lift > recommendations[candidate]['lift']:
                            recommendations[candidate] = {
                                'recommended_item': candidate,
                                'confidence': round(confidence, 3),
                                'lift': round(lift, 2),
                                'support': round(support, 3),
                                'based_on_rule': f"{row['antecedents_str']} -> {candidate}"
                            }
    else:
        # Graceful development fallback mock for Day 1
        catalog_mock = {
            'WHITE HANGING HEART T-LIGHT HOLDER': 'RED HANGING HEART T-LIGHT HOLDER',
            'HAND WARMER UNION JACK': 'HAND WARMER RED POLKA DOT',
            'KNITTED UNION FLAG HOT WATER BOTTLE': 'RED WOOLLY HOTTIE WHITE HEART.'
        }
        for item in cleaned_items:
            match = catalog_mock.get(item)
            if match and match not in selected_set:
                recommendations[match] = {
                    'recommended_item': match,
                    'confidence': 0.85,
                    'lift': 4.20,
                    'support': 0.05,
                    'based_on_rule': f"{item} -> {match} (Mock Precedent)"
                }
                
    results = list(recommendations.values())
    results.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
    final_output = results[:top_n]
    
    # 3. Enforce Output Contract
    validate_recommendation_output(final_output)
    return final_output
