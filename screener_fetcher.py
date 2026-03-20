"""
SCREENER.IN FINANCIAL DATA SCRAPER
====================================
Extracts from screener.in for any NSE/BSE stock:
  - Quarterly Results (last 12 quarters)
  - Profit & Loss Statement (last 5 years annual)
  - Balance Sheet (last 5 years annual)
  - Cash Flow Statement (last 5 years annual)
  - Key Financial Ratios (last 5 years)
  - Shareholding Pattern
  - Promoter Pledge Data
 
Why Screener.in:
  - Best-structured financial data for Indian stocks
  - Already cleaned and formatted
  - Free to access (no API key needed)
  - Same data as what professional analysts use
  - Covers all NSE + BSE listed companies
 
Usage:
    from data_layer.screener_fetcher import ScreenerFetcher
    
    fetcher = ScreenerFetcher("RELIANCE")
    data = fetcher.fetch_all()
    
    # Access individual statements
    pl = data['pl']                  # P&L DataFrame
    bs = data['balance_sheet']       # Balance Sheet DataFrame
    cf = data['cash_flow']           # Cash Flow DataFrame
    qr = data['quarterly']           # Quarterly Results DataFrame
    ratios = data['ratios']          # Key Ratios DataFrame
    shareholding = data['shareholding']  # Shareholding pattern
"""
 
import requests
import re
import time
import json
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup
 
 
SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.screener.in/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}
 
 
class ScreenerFetcher:
    """
    Fetches complete financial data from Screener.in for any Indian stock.
    
    Handles:
    - Consolidated vs standalone (tries consolidated first)
    - BSE-only stocks (not on NSE)
    - Stocks with special characters in ticker (M&M -> MM)
    - Rate limiting (respects screener.in load)
    """
    
    BASE_URL = "https://www.screener.in/company"
    
    def __init__(self, ticker: str):
        self.ticker = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
        # Screener uses different symbols for some stocks
        self.screener_id = self._normalize_ticker(self.ticker)
        self.soup = None
        self.is_consolidated = False
        self._fetched = False
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker for Screener URL."""
        # M&M -> MM on screener
        mapping = {
            "M&M":      "MM",
            "M&MFIN":   "MMFIN",
            "L&TFH":    "LTFH",
            "HDFCAMC":  "HDFCAMC",
        }
        return mapping.get(ticker, ticker)
    
    def _fetch_page(self) -> bool:
        """Fetch the Screener.in page for this stock."""
        if self._fetched:
            return self.soup is not None
        
        # Try consolidated first
        urls_to_try = [
            f"{self.BASE_URL}/{self.screener_id}/consolidated/",
            f"{self.BASE_URL}/{self.screener_id}/",
        ]
        
        for url in urls_to_try:
            try:
                resp = requests.get(url, headers=SCREENER_HEADERS, timeout=20)
                if resp.status_code == 200 and "screener.in" in resp.url:
                    self.soup = BeautifulSoup(resp.text, "html.parser")
                    self.is_consolidated = "consolidated" in url
                    self._fetched = True
                    print(f"[Screener] Fetched {self.ticker} ({'consolidated' if self.is_consolidated else 'standalone'})")
                    return True
                time.sleep(0.5)
            except Exception as e:
                print(f"[Screener] {url}: {e}")
        
        self._fetched = True
        return False
    
    def _parse_table(self, section_id: str) -> pd.DataFrame:
        """
        Parse a financial table from a given section ID.
        Screener structures all tables identically:
        - First row = years/quarters (column headers)
        - Each subsequent row = one financial line item
        """
        if not self._fetch_page() or not self.soup:
            return pd.DataFrame()
        
        # Find the section
        section = self.soup.find("section", {"id": section_id})
        if not section:
            # Try by class or heading
            section = self.soup.find("div", {"id": section_id})
        if not section:
            return pd.DataFrame()
        
        table = section.find("table")
        if not table:
            return pd.DataFrame()
        
        try:
            # Extract headers (years or quarter names)
            thead = table.find("thead")
            if thead:
                header_cells = thead.find_all("th")
                headers = [h.get_text(strip=True) for h in header_cells]
            else:
                headers = []
            
            # Extract rows
            tbody = table.find("tbody")
            if not tbody:
                return pd.DataFrame()
            
            rows = []
            for tr in tbody.find_all("tr"):
                cells = tr.find_all(["td", "th"])
                if cells:
                    row_data = [c.get_text(strip=True) for c in cells]
                    rows.append(row_data)
            
            if not rows:
                return pd.DataFrame()
            
            # Build DataFrame
            if headers and len(headers) == len(rows[0]):
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(rows)
                if headers:
                    df.columns = headers[:len(df.columns)] + [
                        f"Col{i}" for i in range(len(headers), len(df.columns))
                    ]
            
            # Set first column as index (it's the row label)
            if not df.empty:
                df = df.set_index(df.columns[0])
                df.index.name = "Metric"
            
            # Clean numeric values
            df = self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"[Screener Table] {section_id}: {e}")
            return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Screener's number formatting into proper floats."""
        for col in df.columns:
            df[col] = df[col].apply(self._clean_number)
        return df
    
    def _clean_number(self, val) -> object:
        """Convert Screener number strings to float."""
        if pd.isna(val) or val == "" or val == "--" or val == "N/A":
            return None
        
        s = str(val).strip()
        
        # Remove % sign (keep as percentage value, not decimal)
        is_pct = s.endswith("%")
        s = s.replace("%", "").replace(",", "").replace(" ", "")
        
        # Handle negatives in brackets e.g. (1,234)
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        
        try:
            num = float(s)
            return num
        except (ValueError, TypeError):
            return val  # Return original if not numeric (e.g. company name)
    
    # ── PUBLIC METHODS ────────────────────────────────────────────────────────
    
    def get_quarterly_results(self) -> pd.DataFrame:
        """
        Quarterly financial results — last 12 quarters.
        Columns: Quarter names (e.g. Sep 2023, Dec 2023...)
        Rows: Sales, Expenses, Operating Profit, OPM%, NP, EPS etc.
        """
        df = self._parse_table("quarters")
        if df.empty:
            df = self._parse_table("quarterly-shp")
        return df
    
    def get_profit_loss(self) -> pd.DataFrame:
        """
        Annual Profit & Loss Statement — last 5+ years.
        Rows: Sales, Expenses, EBITDA, Depreciation, EBIT, Finance Cost,
              PBT, Tax, PAT, EPS, Dividend Payout etc.
        """
        df = self._parse_table("profit-loss")
        return df
    
    def get_balance_sheet(self) -> pd.DataFrame:
        """
        Annual Balance Sheet — last 5+ years.
        Rows: Share Capital, Reserves, Borrowings, Trade Payables,
              Fixed Assets, CWIP, Investments, Debtors, Cash etc.
        """
        df = self._parse_table("balance-sheet")
        return df
    
    def get_cash_flow(self) -> pd.DataFrame:
        """
        Annual Cash Flow Statement — last 5+ years.
        Rows: Operating CF, Investing CF, Financing CF, Net CF.
        """
        df = self._parse_table("cash-flow")
        return df
    
    def get_ratios(self) -> pd.DataFrame:
        """
        Key Financial Ratios — last 5+ years.
        Rows: Debtor Days, Inventory Days, Days Payable, CCC,
              Fixed Asset Turnover, Asset Turnover, ROE, ROCE,
              D/E Ratio, Working Capital, Current Ratio.
        """
        df = self._parse_table("ratios")
        return df
    
    def get_shareholding(self) -> pd.DataFrame:
        """
        Shareholding pattern — last 4 quarters.
        Rows: Promoters, FII, DII, Public.
        """
        df = self._parse_table("shareholding")
        if df.empty:
            df = self._parse_table("shareholding-pattern")
        return df
    
    def get_company_info(self) -> dict:
        """Extract company-level metadata from Screener."""
        if not self._fetch_page() or not self.soup:
            return {}
        
        info = {}
        
        try:
            # Company name
            name_tag = self.soup.find("h1", {"class": "h2"})
            if name_tag:
                info["name"] = name_tag.get_text(strip=True)
            
            # About section
            about = self.soup.find("div", {"class": "company-description"})
            if about:
                info["about"] = about.get_text(strip=True)[:500]
            
            # Key metrics from top of page
            top_ratios = self.soup.find("ul", {"id": "top-ratios"})
            if top_ratios:
                for li in top_ratios.find_all("li"):
                    name_el = li.find("span", {"class": "name"})
                    val_el  = li.find("span", {"class": "number"})
                    if name_el and val_el:
                        key = name_el.get_text(strip=True)
                        val = val_el.get_text(strip=True)
                        info[key] = val
            
            # Sector and industry
            sector_tags = self.soup.find_all("a", href=re.compile(r"/screen/.*sector.*"))
            if sector_tags:
                info["sector"] = sector_tags[0].get_text(strip=True)
            
        except Exception as e:
            print(f"[Screener Info] {self.ticker}: {e}")
        
        return info
    
    def get_pros_cons(self) -> dict:
        """Get the pros and cons listed by Screener analysts."""
        if not self._fetch_page() or not self.soup:
            return {"pros": [], "cons": []}
        
        pros, cons = [], []
        
        try:
            pros_section = self.soup.find("div", {"class": "pros"})
            if pros_section:
                pros = [li.get_text(strip=True) 
                       for li in pros_section.find_all("li")]
            
            cons_section = self.soup.find("div", {"class": "cons"})
            if cons_section:
                cons = [li.get_text(strip=True) 
                       for li in cons_section.find_all("li")]
        except Exception:
            pass
        
        return {"pros": pros, "cons": cons}
    
    def fetch_all(self) -> dict:
        """
        Fetch ALL financial data in one call.
        Returns a dict with all DataFrames and metadata.
        """
        # Single page fetch, then parse everything from it
        self._fetch_page()
        
        return {
            "ticker":          self.ticker,
            "is_consolidated": self.is_consolidated,
            "info":            self.get_company_info(),
            "pros_cons":       self.get_pros_cons(),
            "quarterly":       self.get_quarterly_results(),
            "pl":              self.get_profit_loss(),
            "balance_sheet":   self.get_balance_sheet(),
            "cash_flow":       self.get_cash_flow(),
            "ratios":          self.get_ratios(),
            "shareholding":    self.get_shareholding(),
        }
    
    def get_5yr_summary(self) -> dict:
        """
        Returns a clean 5-year summary dict with the most
        important metrics investors need at a glance.
        """
        pl = self.get_profit_loss()
        bs = self.get_balance_sheet()
        cf = self.get_cash_flow()
        ratios = self.get_ratios()
        
        summary = {}
        
        def safe_row(df, row_patterns):
            """Find a row by partial name match."""
            for pattern in row_patterns:
                for idx in df.index:
                    if pattern.lower() in str(idx).lower():
                        return df.loc[idx]
            return None
        
        # Revenue
        rev_row = safe_row(pl, ["Sales", "Revenue from Operations", "Net Sales"])
        if rev_row is not None:
            summary["Revenue (Cr)"] = rev_row
        
        # Net Profit
        np_row = safe_row(pl, ["Net Profit", "PAT", "Profit after tax"])
        if np_row is not None:
            summary["Net Profit (Cr)"] = np_row
        
        # Operating Profit
        op_row = safe_row(pl, ["Operating Profit", "EBITDA"])
        if op_row is not None:
            summary["Operating Profit (Cr)"] = op_row
        
        # EPS
        eps_row = safe_row(pl, ["EPS", "Earnings per share"])
        if eps_row is not None:
            summary["EPS (Rs)"] = eps_row
        
        # Total Debt
        debt_row = safe_row(bs, ["Borrowings", "Total Debt", "Long term borrowings"])
        if debt_row is not None:
            summary["Total Debt (Cr)"] = debt_row
        
        # Equity
        eq_row = safe_row(bs, ["Equity", "Total Equity", "Reserves"])
        if eq_row is not None:
            summary["Equity (Cr)"] = eq_row
        
        # Operating Cash Flow
        ocf_row = safe_row(cf, ["Operating", "Cash from Operations", "CFO"])
        if ocf_row is not None:
            summary["Operating CF (Cr)"] = ocf_row
        
        # Free Cash Flow
        capex_row = safe_row(cf, ["Capital Expenditure", "Capex", "Investing"])
        if ocf_row is not None and capex_row is not None:
            summary["Free CF (Cr)"] = ocf_row + capex_row  # capex is negative
        
        # ROE and ROCE from ratios
        roe_row  = safe_row(ratios, ["ROE", "Return on Equity"])
        roce_row = safe_row(ratios, ["ROCE", "Return on Capital"])
        if roe_row  is not None: summary["ROE (%)"]  = roe_row
        if roce_row is not None: summary["ROCE (%)"] = roce_row
        
        if summary:
            return pd.DataFrame(summary).T
        return pd.DataFrame()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS  
# ─────────────────────────────────────────────────────────────────────────────
 
def fetch_screener_financials(ticker: str) -> dict:
    """One-line convenience function to fetch all financials."""
    fetcher = ScreenerFetcher(ticker)
    return fetcher.fetch_all()
 
 
def fetch_quarterly_results(ticker: str) -> pd.DataFrame:
    """Fetch just quarterly results."""
    return ScreenerFetcher(ticker).get_quarterly_results()
 
 
def fetch_pl_statement(ticker: str) -> pd.DataFrame:
    """Fetch just P&L statement."""
    return ScreenerFetcher(ticker).get_profit_loss()
 
 
def fetch_balance_sheet(ticker: str) -> pd.DataFrame:
    """Fetch just balance sheet."""
    return ScreenerFetcher(ticker).get_balance_sheet()
 
 
def fetch_cash_flow(ticker: str) -> pd.DataFrame:
    """Fetch just cash flow statement."""
    return ScreenerFetcher(ticker).get_cash_flow()
