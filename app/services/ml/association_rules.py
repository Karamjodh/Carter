import pandas as pd
from dataclasses import dataclass
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

@dataclass
class AssociationResult:
    """
    Everything the association rules stage produces.
    """
    rules : list
    total_found : int

def run_association_rules(
        df_basket : pd.DataFrame,
        min_support : float = 0.02,
        min_confidence : float = 0.3,
        top_n : int = 20,
) -> AssociationResult:
    """
    Mines association rules from transaction basket data.

    Args:
        df_basket:      basket table from preprocessing
                        must have an 'items' column of type List[str]
        min_support:    minimum frequency for an itemset to be considered
                        0.02 = must appear in at least 2% of transactions
        min_confidence: minimum confidence for a rule to be kept
                        0.3 = antecedent buyers must buy consequent 30%+ of the time
        top_n:          how many top rules to return (sorted by lift)

    Returns:
        AssociationResult with rules list and total count
    """
    if df_basket.empty or "items" not in df_basket.columns:
        return AssociationResult(rules = [], total_found = 0)
    transactions = df_basket["items"].tolist()
    # Remove transactions with only one item
    # — single items can't form association rules
    transactions = [t for t in transactions if len(t) > 1]
    if len(transactions) < 10:
        return AssociationResult(rules = [], total_found = 0)
    # ──  Encode transactions into binary matrix ──────────────────────────
    # FP-Growth needs data in this format:
    #
    #           Laptop  Mouse  Keyboard  Phone
    # order_1     True   True     True  False
    # order_2    False  False    False   True
    # order_3     True   True    False  False
    #
    # TransactionEncoder does this conversion automatically    
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns = te.columns_)
    # ──  Find frequent itemsets ──────────────────────────────────────────
    # An itemset is "frequent" if it appears in at least min_support
    # fraction of all transactions
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support = min_support,
        use_colnames = True,
    )
    if frequent_itemsets.empty:
        frequent_itemsets = fpgrowth(
            df_encoded, min_support = min_support / 2, use_colnames = True)
    if frequent_itemsets.empty:
        return AssociationResult(rules = [], total_found = 0)
    rules_df = association_rules(frequent_itemsets, metric = "confidence", min_threshold = min_confidence,)
    if rules_df.empty :
        return AssociationResult(rules = [], total_found = 0)
    rules_df = rules_df.sort_values("lift",ascending = False)
    total_found = len(rules_df)
    rules_df = rules_df.head(top_n)
    rules_list = [{
        "antecedents" : sorted(list(row["antecedents"])),
        "consequents" : sorted(list(row["consequents"])),
        "support" : round(float(row["support"]),4),
        "confidence" : round(float(row["confidence"]),4),
        "lift" : round(float(row["lift"]),4)
    }
    for _, row in rules_df.iterrows()]
    return AssociationResult(rules = rules_list, total_found = total_found,)