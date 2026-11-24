"""
src/association/model.py
========================
Frequent Itemset Mining (Apriori vs FP-Growth) & Rule Generation
Owner: Marwa
"""

import time
import pandas as pd
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
except ImportError:
    apriori = fpgrowth = association_rules = None

def mine_association_rules(basket: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.2):
    """
    Mines frequent patterns and generates association rules.
    """
    if fpgrowth is None:
        raise ImportError("mlxtend is required for rule mining.")
        
    start_time = time.time()
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    mining_time = time.time() - start_time
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules = rules.sort_values(by='lift', ascending=False)
    
    return rules, mining_time
