def build_analysis_prompt(data : dict, focus : str = "general") -> str:
    """
    Builds a structured prompt from ML pipeline results.

    Args:
        data:  dictionary containing all ML outputs — segments,
               rules, forecasts, summary stats
        focus: what to focus on — "general", "retention",
               "upsell", "acquisition", "seasonal"

    Returns:
        A fully formatted prompt string ready to send to any LLM
    """
    ### Focus instructions section
    focus_instruction = {
        "general" : "Provide a balanced set of recommendations covering the biggest opportunities.",
        "retention" : "Focus specifically on reducing churn and re-engaging customers who havent bought recently.",
        "upsell" : "Focus on increasing average order value and cross-selling to existing customers.",
        "acquistion" : "Focus on attracting new customers and similar to the top performing segments.",
        "seasonal" : "Focus on capitalising on seasonal trends and visible in the data.",
    }
    focus_text = focus_instruction.get(focus, focus_instruction["general"])
    ### Datset overview section
    summary = data.get("summary",{})
    overview_section = f"""
    ## Dataset Overview
    - Total customers: {summary.get("total_customers", "N/A")}
    - Total transactions: {summary.get("total_transactions", "N/A")}
    - Total revenue: ${summary.get("total_revenue", 0):,.2f}
    - Average order value: ${summary.get("avg_order_value", 0):,.2f}
    - Date range: {summary.get("date_start", "N/A")} to {summary.get("date_end", "N/A")}
    """
    ### Customer segments section
    segments = data.get("segments", [])
    if segments:
        segments_section = "\n## Customer Segments\n"
        for seg in segments:
            segments_section += (
                f"- **{seg['label']}** — "
                f"{seg['pct_of_customers']}% of customers ({seg['size']} people), "
                f"avg spend ${seg['avg_monetary']:,.2f}, "
                f"last purchased {seg['avg_recency_days']} days ago on average, "
                f"buys {seg['avg_frequency']:.1f}x per period\n"
            )
    else:
        segments_section = "\n## Customer Segments\nNo segmentation data available.\n"

    ### Association rules section
    rules  = data.get("association_rules", [])
    if rules:
        rules_section = "\n## Purchase Patterns (What Customers buy together)\n"
        for rule in rules[:6]:
            antecedents = " + ".join(rule["antecedents"])
            consequents = " + ".join(rule["consequents"])
            confidence = rule["confidence"] * 100
            lift = rule["lift"]
            rules_section += (
                f"- Customers who buy **{antecedents}** "
                f"also buy **{consequents}** "
                f"({confidence :.0f}% of the time, "
                f"{lift:.1f}x more likely than random)\n"
            )
    else:
        rules_section = "\n## Purchase Patterens\nNo pattern data available.\n"
            
    ### Forecast section
    forecasts = data.get("forecasts", [])
    if forecasts:
        forecast_section = "\n## Revenue Trends & Forecasts\n"
        for category, info in list(forecasts.items())[:5]:
            for category, info in list(forecasts.items())[:5]:
                trend = info.get("trend_pct", 0)
                direction = "up" if trend > 0 else "down"
                forecast_section += (
                    f"- **{category}** : {direction} {abs(trend):.1f}% "
                    f"vs previous period\n"
                )
    else:
        forecast_section = "\n## Revenue Trends\nNo forecast data available.\n"
    
    ### Final prompt assembly
    prompt = f"""You are a senior marketing strategist with expertise in customer analytics and data-driven campaign planning.

Your audience is a marketing manager — they understand business but are not technical. Avoid terms like "silhouette score" or "lift value". Instead explain what the numbers mean for the business.

{focus_text}

---
{overview_section}
{segments_section}
{rules_section}
{forecast_section}
---

Based on the data above, provide a strategic marketing report with:

1. **Executive Summary** — 2-3 sentences summarising the biggest insight from the data
2. **Top 3 Recommendations** — each with a specific action, which customer segment to target, and expected impact
3. **One Risk or Caveat** — something the marketing team should watch out for
4. **Suggested A/B Test** — one concrete experiment to validate your top recommendation

Be specific. Reference the actual numbers from the data. Keep language clear and actionable.
"""
    return prompt
