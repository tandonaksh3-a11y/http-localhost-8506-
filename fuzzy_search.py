"""
AKRE TERMINAL — Fuzzy Search Engine
Converts user input (company name, alias, partial name, ticker) to NSE/BSE symbol.
Provides dropdown with ranked suggestions.
"""
import streamlit as st
import difflib
import re
import requests

# ─── Common Aliases & Name Mappings ──────────────────────────────────────────
ALIASES = {
    # Common abbreviations → NSE ticker
    "RELIANCE": "RELIANCE", "RIL": "RELIANCE", "JIOS": "RELIANCE",
    "TCS": "TCS", "TATA CONSULTANCY": "TCS",
    "INFY": "INFY", "INFOSYS": "INFY",
    "HDFCBANK": "HDFCBANK", "HDFC BANK": "HDFCBANK", "HDFC": "HDFCBANK",
    "ICICIBANK": "ICICIBANK", "ICICI BANK": "ICICIBANK", "ICICI": "ICICIBANK",
    "SBIN": "SBIN", "SBI": "SBIN", "STATE BANK": "SBIN",
    "KOTAKBANK": "KOTAKBANK", "KOTAK": "KOTAKBANK", "KOTAK MAHINDRA": "KOTAKBANK",
    "WIPRO": "WIPRO",
    "HCLTECH": "HCLTECH", "HCL": "HCLTECH", "HCL TECH": "HCLTECH",
    "TATAMOTORS": "TATAMOTORS", "TATA MOTORS": "TATAMOTORS",
    "MARUTI": "MARUTI", "MARUTI SUZUKI": "MARUTI",
    "SUNPHARMA": "SUNPHARMA", "SUN PHARMA": "SUNPHARMA", "SUN": "SUNPHARMA",
    "BHARTIARTL": "BHARTIARTL", "AIRTEL": "BHARTIARTL", "BHARTI AIRTEL": "BHARTIARTL",
    "TATASTEEL": "TATASTEEL", "TATA STEEL": "TATASTEEL",
    "LT": "LT", "LARSEN": "LT", "L&T": "LT", "LARSEN AND TOUBRO": "LT",
    "HINDUNILVR": "HINDUNILVR", "HUL": "HINDUNILVR", "HINDUSTAN UNILEVER": "HINDUNILVR",
    "ITC": "ITC",
    "AXISBANK": "AXISBANK", "AXIS BANK": "AXISBANK", "AXIS": "AXISBANK",
    "BAJFINANCE": "BAJFINANCE", "BAJAJ FINANCE": "BAJFINANCE",
    "ASIANPAINT": "ASIANPAINT", "ASIAN PAINTS": "ASIANPAINT",
    "TECHM": "TECHM", "TECH MAHINDRA": "TECHM",
    "NTPC": "NTPC",
    "POWERGRID": "POWERGRID", "POWER GRID": "POWERGRID",
    "ULTRACEMCO": "ULTRACEMCO", "ULTRATECH": "ULTRACEMCO", "ULTRATECH CEMENT": "ULTRACEMCO",
    "COALINDIA": "COALINDIA", "COAL INDIA": "COALINDIA",
    "NESTLEIND": "NESTLEIND", "NESTLE": "NESTLEIND", "NESTLE INDIA": "NESTLEIND",
    "DRREDDY": "DRREDDY", "DR REDDY": "DRREDDY", "DR REDDYS": "DRREDDY",
    "CIPLA": "CIPLA",
    "ONGC": "ONGC",
    "BPCL": "BPCL",
    "IOC": "IOC", "INDIAN OIL": "IOC",
    "GAIL": "GAIL",
    "M&M": "M&M", "MAHINDRA": "M&M",
    "EICHERMOT": "EICHERMOT", "EICHER": "EICHERMOT", "ROYAL ENFIELD": "EICHERMOT",
    "HEROMOTOCO": "HEROMOTOCO", "HERO": "HEROMOTOCO", "HERO MOTO": "HEROMOTOCO",
    "BAJAJ-AUTO": "BAJAJ-AUTO", "BAJAJ AUTO": "BAJAJ-AUTO",
    "DIVISLAB": "DIVISLAB", "DIVIS LAB": "DIVISLAB",
    "APOLLOHOSP": "APOLLOHOSP", "APOLLO": "APOLLOHOSP", "APOLLO HOSPITALS": "APOLLOHOSP",
    "ADANIGREEN": "ADANIGREEN", "ADANI GREEN": "ADANIGREEN",
    "ADANIENT": "ADANIENT", "ADANI": "ADANIENT", "ADANI ENTERPRISES": "ADANIENT",
    "ADANIPORTS": "ADANIPORTS", "ADANI PORTS": "ADANIPORTS",
    "TATAPOWER": "TATAPOWER", "TATA POWER": "TATAPOWER",
    "DLF": "DLF",
    "VEDL": "VEDL", "VEDANTA": "VEDL",
    "JSWSTEEL": "JSWSTEEL", "JSW STEEL": "JSWSTEEL", "JSW": "JSWSTEEL",
    "HINDALCO": "HINDALCO",
    "INDUSINDBK": "INDUSINDBK", "INDUSIND": "INDUSINDBK", "INDUSIND BANK": "INDUSINDBK",
    "BRITANNIA": "BRITANNIA",
    "DABUR": "DABUR",
    "COLPAL": "COLPAL", "COLGATE": "COLPAL",
    "MARICO": "MARICO",
    "SIEMENS": "SIEMENS",
    "ABB": "ABB",
    "HAVELLS": "HAVELLS",
    "VOLTAS": "VOLTAS",
    "PIDILITIND": "PIDILITIND", "PIDILITE": "PIDILITIND",
    "BIOCON": "BIOCON",
    "LUPIN": "LUPIN",
    "TATACONSUM": "TATACONSUM", "TATA CONSUMER": "TATACONSUM",
    "GODREJCP": "GODREJCP", "GODREJ": "GODREJCP",
    "PNB": "PNB", "PUNJAB NATIONAL": "PNB",
    "BANKBARODA": "BANKBARODA", "BANK OF BARODA": "BANKBARODA", "BOB": "BANKBARODA",
    "CANBK": "CANBK", "CANARA BANK": "CANBK",
    "IDFCFIRSTB": "IDFCFIRSTB", "IDFC FIRST": "IDFCFIRSTB",
    "FEDERALBNK": "FEDERALBNK", "FEDERAL BANK": "FEDERALBNK",
    "BANDHANBNK": "BANDHANBNK", "BANDHAN": "BANDHANBNK",
}

# All known symbols for fuzzy matching
ALL_SYMBOLS = sorted(set(ALIASES.values()))


def fuzzy_match(query: str, max_results: int = 8) -> list:
    """
    Match user input against known tickers & aliases.
    Returns list of dicts: [{symbol, name, confidence, source}, ...]
    """
    if not query or len(query.strip()) < 2:
        return []

    query = query.strip().upper()
    results = []

    # 1. Exact symbol match
    if query in ALL_SYMBOLS:
        results.append({
            "symbol": query,
            "name": _get_company_name(query),
            "confidence": 100,
            "source": "exact_match"
        })
        return results

    # 2. Exact alias match
    if query in ALIASES:
        sym = ALIASES[query]
        results.append({
            "symbol": sym,
            "name": _get_company_name(sym),
            "confidence": 98,
            "source": "alias"
        })
        return results

    # 3. Partial alias match (startswith)
    partial_matches = []
    for alias, sym in ALIASES.items():
        if alias.startswith(query) or query.startswith(alias):
            if sym not in [r["symbol"] for r in partial_matches]:
                partial_matches.append({
                    "symbol": sym,
                    "name": _get_company_name(sym),
                    "confidence": 85,
                    "source": "partial"
                })

    # 4. Fuzzy string matching against all aliases
    all_alias_keys = list(ALIASES.keys())
    close_matches = difflib.get_close_matches(query, all_alias_keys, n=max_results, cutoff=0.5)
    for match in close_matches:
        sym = ALIASES[match]
        if sym not in [r["symbol"] for r in partial_matches] and sym not in [r["symbol"] for r in results]:
            ratio = difflib.SequenceMatcher(None, query, match).ratio()
            partial_matches.append({
                "symbol": sym,
                "name": _get_company_name(sym),
                "confidence": int(ratio * 100),
                "source": "fuzzy"
            })

    # Combine, deduplicate, and sort
    seen = set()
    for r in results + partial_matches:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            results.append(r) if r not in results else None

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # 5. NSE API fallback if no results
    if not results:
        nse_results = _nse_search_fallback(query)
        results.extend(nse_results)

    return results[:max_results]


def _get_company_name(symbol: str) -> str:
    """Get the display name for a symbol from aliases."""
    name_map = {}
    for alias, sym in ALIASES.items():
        if sym == symbol and len(alias) > len(name_map.get(sym, "")):
            # Prefer longer alias names (more descriptive)
            if " " in alias or len(alias) > 6:
                name_map[sym] = alias.title()

    return name_map.get(symbol, symbol)


def _nse_search_fallback(query: str) -> list:
    """Fallback: search NSE API for the query."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        url = f"https://www.nseindia.com/api/search/autocomplete?q={query}"
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for item in data.get("symbols", [])[:5]:
                results.append({
                    "symbol": item.get("symbol", ""),
                    "name": item.get("symbol_info", item.get("symbol", "")),
                    "confidence": 70,
                    "source": "nse_api"
                })
            return results
    except Exception:
        pass
    return []


def render_search_bar():
    """
    Render the AKRE fuzzy search bar with autocomplete dropdown.
    Returns (symbol, go_clicked) tuple.
    """
    col_cmd, col_btn = st.columns([5, 1])

    with col_cmd:
        query = st.text_input(
            "🔍 SEARCH",
            value="",
            placeholder="Search: company name, ticker, or alias (e.g. Reliance, SBI, HDFC Bank)",
            label_visibility="collapsed",
            key="akre_search"
        )

    with col_btn:
        go_clicked = st.button("▶ ANALYZE", type="primary", use_container_width=True)

    # Show dropdown suggestions
    resolved_symbol = query.strip().upper() if query else ""

    if query and len(query.strip()) >= 2 and not go_clicked:
        matches = fuzzy_match(query.strip())
        if matches and len(matches) > 0:
            # Only show dropdown if not an exact match to what's typed
            typed_upper = query.strip().upper()
            if not (len(matches) == 1 and matches[0]["symbol"] == typed_upper and matches[0]["confidence"] == 100):
                st.markdown("""<div style="color:#616161;font-size:9px;letter-spacing:1px;margin:-8px 0 4px 0;">
                SUGGESTIONS</div>""", unsafe_allow_html=True)

                suggestion_cols = st.columns(min(len(matches), 4))
                for i, match in enumerate(matches[:4]):
                    with suggestion_cols[i]:
                        conf_color = "#00E676" if match["confidence"] >= 90 else (
                            "#FFC107" if match["confidence"] >= 70 else "#9E9E9E")
                        if st.button(
                            f"{match['symbol']} ({match['confidence']}%)",
                            key=f"suggestion_{i}",
                            use_container_width=True
                        ):
                            resolved_symbol = match["symbol"]
                            go_clicked = True

        # Auto-resolve if single high-confidence match
        if matches and len(matches) == 1 and matches[0]["confidence"] >= 85:
            resolved_symbol = matches[0]["symbol"]

    # If no .NS or .BO suffix and not a special command, add .NS
    if resolved_symbol and not resolved_symbol.startswith("SECTOR:"):
        if not resolved_symbol.endswith(".NS") and not resolved_symbol.endswith(".BO"):
            resolved_symbol = resolved_symbol + ".NS"

    return resolved_symbol, go_clicked
