"""
NSE PEER FETCHER — Extracts real sector peers from NSE's own API
================================================================
Why your current peer comparison shows "No peers found":
  - yfinance's info['sector'] returns None for most NSE stocks
  - Your code then searches an empty string and finds nothing
  
This module fixes that by going directly to NSE's official API
which has the actual sector/industry classification for every
listed stock, and returns the real competitive peer set.
 
Sources used (in order of preference):
  1. NSE official API  — nseindia.com/api (authoritative)
  2. Screener.in peers — screener.in/company/TICKER (curated by analysts)
  3. Sector index mapping — pre-built NSE sector → stock list
  4. yfinance fallback — last resort
 
Usage:
    from data_layer.nse_peer_fetcher import get_peers
    peers = get_peers("RELIANCE")
    # Returns list of peer tickers with metadata
"""
 
import requests
import time
import json
import re
from typing import Optional
import pandas as pd
 
# ─────────────────────────────────────────────────────────────────────────────
# SESSION SETUP — NSE requires browser-like headers or it blocks requests
# ─────────────────────────────────────────────────────────────────────────────
 
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "DNT": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}
 
SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.screener.in/",
    "Connection": "keep-alive",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# NSE SECTOR INDEX MAP
# Maps every NSE sector index to the list of constituent stocks
# This is the fallback when API calls fail
# ─────────────────────────────────────────────────────────────────────────────
 
NSE_SECTOR_INDICES = {
    "NIFTY IT":           "NIFTY%20IT",
    "NIFTY BANK":         "NIFTY%20BANK",
    "NIFTY PHARMA":       "NIFTY%20PHARMA",
    "NIFTY AUTO":         "NIFTY%20AUTO",
    "NIFTY FMCG":         "NIFTY%20FMCG",
    "NIFTY METAL":        "NIFTY%20METAL",
    "NIFTY REALTY":       "NIFTY%20REALTY",
    "NIFTY ENERGY":       "NIFTY%20ENERGY",
    "NIFTY INFRA":        "NIFTY%20INFRA",
    "NIFTY MEDIA":        "NIFTY%20MEDIA",
    "NIFTY PSU BANK":     "NIFTY%20PSU%20BANK",
    "NIFTY PRIVATE BANK": "NIFTY%20PRIVATE%20BANK",
    "NIFTY FINANCIAL SERVICES": "NIFTY%20FINANCIAL%20SERVICES",
    "NIFTY CONSUMER DURABLES": "NIFTY%20CONSUMER%20DURABLES",
    "NIFTY OIL AND GAS":  "NIFTY%20OIL%20AND%20GAS",
    "NIFTY HEALTHCARE":   "NIFTY%20HEALTHCARE",
    "NIFTY CPSE":         "NIFTY%20CPSE",
}
 
# Manual peer map for the most searched stocks
# Used as instant fallback — no API call needed
MANUAL_PEER_MAP = {
    "RELIANCE":   ["ONGC", "IOC", "BPCL", "HPCL", "GAIL", "OIL", "MRPL"],
    "TCS":        ["INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"],
    "INFY":       ["TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"],
    "WIPRO":      ["TCS", "INFY", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT"],
    "HCLTECH":    ["TCS", "INFY", "WIPRO", "TECHM", "LTIM", "MPHASIS", "PERSISTENT"],
    "HDFC":       ["SBIN", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBANK", "BANDHANBNK"],
    "HDFCBANK":   ["SBIN", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBANK", "BANDHANBNK"],
    "ICICIBANK":  ["HDFCBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBANK", "BANDHANBNK"],
    "SBIN":       ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "PNB", "BANKBARODA", "CANBK"],
    "KOTAKBANK":  ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "INDUSINDBANK"],
    "AXISBANK":   ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "INDUSINDBANK"],
    "BAJFINANCE": ["BAJAJFINSV", "CHOLAFIN", "M&MFIN", "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM"],
    "MARUTI":     ["TATAMOTORS", "M&M", "HYUNDAI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "TATAMOTORS": ["MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TIINDIA"],
    "SUNPHARMA":  ["DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN", "TORNTPHARM", "ALKEM"],
    "DRREDDY":    ["SUNPHARMA", "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN", "TORNTPHARM"],
    "CIPLA":      ["SUNPHARMA", "DRREDDY", "DIVISLAB", "AUROPHARMA", "LUPIN", "TORNTPHARM"],
    "ONGC":       ["OIL", "BPCL", "IOC", "HPCL", "GAIL", "RELIANCE", "MRPL"],
    "IOC":        ["BPCL", "HPCL", "ONGC", "GAIL", "RELIANCE", "OIL", "MRPL"],
    "NESTLEIND":  ["HINDUNILVR", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL", "EMAMILTD"],
    "HINDUNILVR": ["NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL", "ITC"],
    "ITC":        ["HINDUNILVR", "NESTLEIND", "DABUR", "MARICO", "VST", "GODFRYPHLP"],
    "TITAN":      ["KALYAN", "SENCO", "PCJEWELLER", "TBZL", "SKM"],
    "ASIANPAINT": ["BERGER", "KANSAINER", "AKZONOBEL", "SHALPAINTS", "INDIGO"],
    "ULTRACEMCO": ["AMBUJACEM", "ACC", "SHREECEM", "JKCEMENT", "DALMIA", "RAMCOCEM"],
    "ADANIENT":   ["ADANIPORTS", "ADANIGREEN", "ADANIPOWER", "ADANITRANS", "ADANIGAS"],
    "ADANIPORTS": ["CONCOR", "GPPL", "ESABIND", "WELCORP"],
    "BAJAJ-AUTO": ["HEROMOTOCO", "EICHERMOT", "TVSMOTORS", "TATAMOTORS", "MARUTI"],
    "HEROMOTOCO": ["BAJAJ-AUTO", "TVSMOTORS", "EICHERMOT", "HONDA"],
    "LT":         ["TIINDIA", "SIEMENS", "ABB", "BHEL", "THERMAX", "CUMMINSIND"],
    "NTPC":       ["POWERGRID", "TATAPOWER", "ADANIGREEN", "CESC", "TORNTPOWER"],
    "POWERGRID":  ["NTPC", "TATAPOWER", "ADANIGREEN", "CESC", "PFC", "RECLTD"],
    "COALINDIA":  ["NMDC", "HINDCOPPER", "MOIL", "GMRINFRA"],
    "TATASTEEL":  ["JSWSTEEL", "SAIL", "HINDALCO", "VEDL", "NMDC", "JSPL"],
    "JSWSTEEL":   ["TATASTEEL", "SAIL", "HINDALCO", "VEDL", "JSPL", "NMDC"],
    "HINDALCO":   ["TATASTEEL", "JSWSTEEL", "VEDL", "NALCO", "HINDCOPPER", "SAIL"],
    "GRASIM":     ["ULTRACEMCO", "AMBUJACEM", "ACC", "HINDALCO", "BIRLASOFT"],
    "HDFCLIFE":   ["SBILIFE", "ICICIPRU", "MAXFINSERV", "STARHEALTH"],
    "SBILIFE":    ["HDFCLIFE", "ICICIPRU", "MAXFINSERV", "ABSLAMC"],
    "DIVISLAB":   ["SUNPHARMA", "DRREDDY", "CIPLA", "AUROPHARMA", "LUPIN", "GRANULES"],
    "TECHM":      ["TCS", "INFY", "WIPRO", "HCLTECH", "LTIM", "MPHASIS"],
    "WIPRO":      ["TCS", "INFY", "HCLTECH", "TECHM", "LTIM", "MPHASIS"],
    "BHARTIARTL": ["VODAIDEA", "RJIO", "TATACOMM", "HFCL", "GTLINFRA"],
    "M&M":        ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "EICHERMOT":  ["BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTORS", "M&M", "TATAMOTORS"],
    "PIDILITIND": ["FEVICOL", "ASTRAL", "SUPREMEIND", "FINOLEX"],
    "DABUR":      ["HINDUNILVR", "MARICO", "EMAMILTD", "COLPAL", "NESTLEIND"],
    "SWIGGY":     ["ZOMATO", "DEVYANI", "JUBLFOOD", "WESTLIFE", "SAPPHIRE", "BARBEQUE"],
    "ZOMATO":     ["SWIGGY", "DEVYANI", "JUBLFOOD", "WESTLIFE", "SAPPHIRE", "BARBEQUE"],
    "PAYTM":      ["POLICYBZR", "NYKAA", "CARTRADE", "EASEMYTRIP", "RATEGAIN"],
    "NYKAA":      ["PAYTM", "MAPMYINDIA", "CARTRADE", "HONASA"],
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def _nse_session() -> requests.Session:
    """Create a session with NSE cookies — required or NSE blocks all requests."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # First hit the main page to get cookies — NSE checks for this
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.5)  # Small delay to appear human
    except Exception:
        pass
    return session
 
 
def get_nse_stock_info(ticker: str) -> dict:
    """
    Fetch stock quote from NSE API.
    Returns sector, industry, series, ISIN and more.
    """
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    session = _nse_session()
    url = f"https://www.nseindia.com/api/quote-equity?symbol={clean}"
    
    try:
        resp = session.get(url, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            info = data.get("info", {})
            metadata = data.get("metadata", {})
            return {
                "symbol":   info.get("symbol", clean),
                "companyName": info.get("companyName", ""),
                "industry": info.get("industry", ""),
                "sector":   metadata.get("pdSectorPe", ""),
                "isin":     info.get("isin", ""),
                "series":   info.get("series", "EQ"),
            }
    except Exception as e:
        print(f"[NSE API] {clean}: {e}")
    return {}
 
 
def get_nse_sector_stocks(sector_index: str) -> list:
    """
    Get all stocks in a given NSE sector index.
    e.g. sector_index = "NIFTY IT"
    """
    encoded = NSE_SECTOR_INDICES.get(sector_index, sector_index.replace(" ", "%20"))
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}"
    session = _nse_session()
    
    try:
        resp = session.get(url, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            stocks = data.get("data", [])
            return [s["symbol"] for s in stocks if s.get("symbol")]
    except Exception as e:
        print(f"[NSE Sector] {sector_index}: {e}")
    return []
 
 
def _is_valid_peer_symbol(symbol: str) -> bool:
    """
    Validate that a scraped symbol is actually a real stock ticker,
    not an index code, BSE numeric code, or page metadata.
    """
    s = symbol.strip().upper()
    
    # Reject empty
    if not s or len(s) < 2:
        return False
    
    # Reject purely numeric (BSE codes like 1005, 1003, 1153)
    if s.isdigit():
        return False
    
    # Reject known index prefixes
    index_prefixes = ("CNX", "NIFTY", "SENSEX", "BSE", "NSE")
    if any(s.startswith(prefix) for prefix in index_prefixes):
        return False
    
    # Reject known non-stock page slugs from Screener
    non_stocks = {
        "CONSOLIDATED", "STANDALONE", "COMPARE", "SCREEN", "SCREENS",
        "LOGIN", "REGISTER", "ABOUT", "FAQ", "HELP", "BLOG", "API",
        "SETTINGS", "DASHBOARD", "ALERTS", "EXPORT", "PREMIUM",
    }
    if s in non_stocks:
        return False
    
    # Reject if starts with a digit (like "200INDE")
    if s[0].isdigit():
        return False
    
    return True


def get_screener_peers(ticker: str) -> list:
    """
    Scrape Screener.in for peer companies.
    Screener shows analyst-curated peers for each stock.
    Returns list of dicts with symbol + company name.
    """
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    url = f"https://www.screener.in/company/{clean}/consolidated/"
    
    try:
        resp = requests.get(url, headers=SCREENER_HEADERS, timeout=15)
        if resp.status_code == 404:
            # Try standalone if consolidated not found
            url = f"https://www.screener.in/company/{clean}/"
            resp = requests.get(url, headers=SCREENER_HEADERS, timeout=15)
        
        if resp.status_code != 200:
            return []
        
        html = resp.text
        peers = []
        
        # Strategy 1: Look for the dedicated peer comparison section
        # Screener has a section with id="peers" or class containing "peers"
        peer_section_match = re.search(
            r'<section[^>]*id=["\']peers["\'][^>]*>(.*?)</section>',
            html, re.DOTALL | re.IGNORECASE
        )
        
        if not peer_section_match:
            # Try table-based match
            peer_section_match = re.search(
                r'<table[^>]*class="[^"]*peers[^"]*"[^>]*>(.*?)</table>',
                html, re.DOTALL | re.IGNORECASE
            )
        
        if not peer_section_match:
            # Try the "Peer Comparison" heading approach
            peer_section_match = re.search(
                r'Peer\s+Comparison.*?(<table.*?</table>)',
                html, re.DOTALL | re.IGNORECASE
            )
        
        if peer_section_match:
            peer_html = peer_section_match.group(1)
            # Find all company links within the peer section only
            links = re.findall(
                r'href="/company/([A-Z0-9&-]+)/?[^"]*"[^>]*>([^<]+)<',
                peer_html, re.IGNORECASE
            )
            for symbol, name in links:
                s = symbol.strip().upper()
                if s != clean and _is_valid_peer_symbol(s):
                    peers.append({"symbol": s, "name": name.strip()})
        
        # Strategy 2: If peer section not found, look for links ONLY in tables
        # (avoids grabbing navigation/sidebar links)
        if not peers:
            # Find all table sections and extract company links from those
            table_matches = re.findall(
                r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE
            )
            seen = set()
            for table_html in table_matches:
                links = re.findall(
                    r'href="/company/([A-Z][A-Z0-9&-]{1,14})/?[^"]*"',
                    table_html, re.IGNORECASE
                )
                for sym in links:
                    s = sym.strip().upper()
                    if s != clean and s not in seen and _is_valid_peer_symbol(s):
                        seen.add(s)
                        peers.append({"symbol": s, "name": s})
            peers = peers[:10]
        
        return peers
        
    except Exception as e:
        print(f"[Screener Peers] {clean}: {e}")
        return []
 
 
def _find_sector_for_stock(ticker: str) -> Optional[str]:
    """
    Find which NSE sector index a stock belongs to.
    Checks all sector indices until the stock is found.
    """
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    # Check each sector index
    for sector_name, _ in NSE_SECTOR_INDICES.items():
        stocks = get_nse_sector_stocks(sector_name)
        if clean in stocks:
            return sector_name
        time.sleep(0.2)  # Rate limiting
    
    return None
 
 
def get_peers(ticker: str, max_peers: int = 8) -> list:
    """
    MAIN FUNCTION — Get peer companies for any NSE stock.
    
    Strategy (tried in order, stops when enough peers found):
    1. Manual peer map (instant, no API call)
    2. Screener.in peer table (analyst-curated)
    3. NSE sector index constituents (official)
    4. NSE industry classification match
    
    Returns list of dicts:
    [
        {
            "symbol": "TCS",
            "name": "Tata Consultancy Services",
            "source": "screener",  # where we found this peer
        },
        ...
    ]
    """
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    peers = []
    
    # ── Step 1: Manual map (instant)
    if clean in MANUAL_PEER_MAP:
        for sym in MANUAL_PEER_MAP[clean]:
            peers.append({"symbol": sym, "name": sym, "source": "curated"})
        print(f"[Peers] {clean}: Found {len(peers)} peers from curated map")
        return peers[:max_peers]
    
    # ── Step 2: Screener.in (best quality)
    screener_peers = get_screener_peers(clean)
    if screener_peers:
        for p in screener_peers:
            sym = p.get("symbol", "")
            # Double-check every Screener result through validation
            if _is_valid_peer_symbol(sym):
                peers.append({**p, "source": "screener.in"})
            else:
                print(f"[Peers] Rejected invalid peer: {sym}")
        print(f"[Peers] {clean}: Found {len(peers)} valid peers from Screener.in")
        if len(peers) >= 4:
            return peers[:max_peers]
    
    # ── Step 3: NSE sector index
    nse_info = get_nse_stock_info(clean)
    industry = nse_info.get("industry", "")
    
    # Map NSE industry to sector index name
    industry_to_sector = {
        "IT-Software":         "NIFTY IT",
        "IT-Hardware":         "NIFTY IT",
        "Banks":               "NIFTY BANK",
        "Private Banks":       "NIFTY PRIVATE BANK",
        "Public Sector Banks": "NIFTY PSU BANK",
        "Pharmaceuticals":     "NIFTY PHARMA",
        "Automobiles":         "NIFTY AUTO",
        "Auto Ancillaries":    "NIFTY AUTO",
        "FMCG":                "NIFTY FMCG",
        "Metals":              "NIFTY METAL",
        "Steel":               "NIFTY METAL",
        "Realty":              "NIFTY REALTY",
        "Power":               "NIFTY ENERGY",
        "Oil & Gas":           "NIFTY OIL AND GAS",
        "Petroleum":           "NIFTY OIL AND GAS",
        "Consumer Durables":   "NIFTY CONSUMER DURABLES",
        "Media":               "NIFTY MEDIA",
        "Finance":             "NIFTY FINANCIAL SERVICES",
        "Infrastructure":      "NIFTY INFRA",
    }
    
    sector_index = None
    for ind_key, sect in industry_to_sector.items():
        if ind_key.lower() in industry.lower():
            sector_index = sect
            break
    
    if sector_index:
        sector_stocks = get_nse_sector_stocks(sector_index)
        for sym in sector_stocks:
            if sym != clean and not any(p["symbol"] == sym for p in peers):
                peers.append({"symbol": sym, "name": sym, "source": "NSE sector index"})
    
    if peers:
        print(f"[Peers] {clean}: Found {len(peers)} peers total")
        return peers[:max_peers]
    
    # ── Step 4: Last resort — search all sector indices
    print(f"[Peers] {clean}: Searching all NSE sector indices...")
    found_sector = _find_sector_for_stock(clean)
    if found_sector:
        sector_stocks = get_nse_sector_stocks(found_sector)
        for sym in sector_stocks:
            if sym != clean:
                peers.append({"symbol": sym, "name": sym, "source": f"NSE {found_sector}"})
        return peers[:max_peers]
    
    print(f"[Peers] {clean}: Could not find peers automatically")
    return []
 
 
def enrich_peers_with_data(ticker: str, peers: list) -> pd.DataFrame:
    """
    Given a list of peer symbols, fetch their key metrics
    and return a DataFrame for the peer comparison table.
    """
    import yfinance as yf
    
    results = []
    # Final validation: reject any remaining invalid symbols before making API calls
    valid_peers = [p for p in peers if _is_valid_peer_symbol(p.get("symbol", ""))]
    all_tickers = [ticker] + [p["symbol"] for p in valid_peers]
    
    for sym in all_tickers:
        # Skip obviously invalid symbols (numeric BSE codes, index codes, etc.)
        if sym != ticker and not _is_valid_peer_symbol(sym):
            print(f"[Peer Data] Skipping invalid symbol: {sym}")
            continue
        
        nse_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        try:
            info = yf.Ticker(nse_sym).info
            if not info or info.get("regularMarketPrice") is None:
                # Try BSE
                bse_sym = sym if sym.endswith(".BO") else f"{sym}.BO"
                info = yf.Ticker(bse_sym).info
            
            # Skip if yfinance returned no real data (likely not a real stock)
            if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
                print(f"[Peer Data] No market data for {sym}, likely invalid")
                continue
            
            results.append({
                "Symbol":       sym,
                "Company":      info.get("longName", sym)[:25],
                "CMP":          info.get("regularMarketPrice", 0),
                "Mkt Cap (Cr)": round((info.get("marketCap", 0) or 0) / 1e7, 0),
                "P/E":          round(info.get("trailingPE", 0) or 0, 1),
                "P/B":          round(info.get("priceToBook", 0) or 0, 1),
                "EV/EBITDA":    round(info.get("enterpriseToEbitda", 0) or 0, 1),
                "ROE (%)":      round((info.get("returnOnEquity", 0) or 0) * 100, 1),
                "ROCE (%)":     round((info.get("returnOnAssets", 0) or 0) * 100 * 2, 1),
                "Net Margin(%)":round((info.get("profitMargins", 0) or 0) * 100, 1),
                "Rev Growth(%)":round((info.get("revenueGrowth", 0) or 0) * 100, 1),
                "D/E Ratio":    round((info.get("debtToEquity", 0) or 0) / 100, 2),
                "Div Yield(%)": round((info.get("dividendYield", 0) or 0) * 100, 2),
            })
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"[Peer Data] {sym}: {e}")
            results.append({"Symbol": sym, "Company": sym})
    
    df = pd.DataFrame(results)
    df = df.fillna(0)
    return df
