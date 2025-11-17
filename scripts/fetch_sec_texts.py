#!/usr/bin/env python3
"""
Fetch 2024 Form 10-K filings for C, GS, JPM from SEC EDGAR and save plain text.

Outputs:
  data/sec/citi_2024_10k.txt
  data/sec/gs_2024_10k.txt
  data/sec/jpm_2024_10k.txt
  data/sec/filing_meta.jsonl   # one JSON line per file with url, accession, cik, ticker
"""

import os, re, json, shutil, pathlib, time
from datetime import datetime
from bs4 import BeautifulSoup
from html2text import HTML2Text
from sec_edgar_downloader import Downloader

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sec"
TMP_DIR = ROOT / "data" / "sec_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.environ.get("SEC_USER_AGENT", "example@example.com")

# NOTE: SEC requires a valid User-Agent header with your email/company info.
# Set this environment variable before running: export SEC_USER_AGENT="YourName YourEmail@company.com"

# Ticker -> friendly name + target output filename stem
TARGETS = {
    "C":   ("citi_2024_10k",  "Citigroup"),
    "GS":  ("gs_2024_10k",    "Goldman Sachs"),
    "JPM": ("jpm_2024_10k",   "JPMorgan Chase"),
}

# We want the most recent 10-K filings (likely 2024 annual reports filed in early 2025)
# But to be safe, we'll accept filings from 2024 or 2025
VALID_FILED_YEARS = [2024, 2025]
FORM_TYPE  = "10-K"

def html_to_text(html_path: pathlib.Path) -> str:
    html = html_path.read_text(errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Drop heavy nav/ix elements commonly present in EDGAR Inline XBRL
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    # Some EDGAR docs have <ix:*> tags; unwrap them to keep visible text
    for el in soup.find_all():
        if el.name and ":" in el.name:
            el.name = el.name.split(":")[-1]

    h = HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # no wrapping
    text = h.handle(str(soup))

    # Normalize whitespace, collapse multiple blank lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def parse_submission_metadata(submission_path: pathlib.Path) -> dict:
    """
    Parse metadata from full-submission.txt header
    """
    content = submission_path.read_text(errors="ignore")
    lines = content.split('\n')

    metadata = {}
    for line in lines[:100]:  # Header is at the top
        line = line.strip()
        if line.startswith('ACCESSION NUMBER:'):
            metadata['accession_number'] = line.split(':', 1)[1].strip()
        elif line.startswith('CONFORMED SUBMISSION TYPE:'):
            metadata['form_type'] = line.split(':', 1)[1].strip()
        elif line.startswith('FILED AS OF DATE:'):
            filed_date = line.split(':', 1)[1].strip()
            # Format as YYYY-MM-DD
            if len(filed_date) == 8:
                metadata['filing_date'] = f"{filed_date[:4]}-{filed_date[4:6]}-{filed_date[6:8]}"
        elif line.startswith('CENTRAL INDEX KEY:') and 'cik' not in metadata:
            metadata['cik'] = line.split(':', 1)[1].strip()
        elif line.startswith('COMPANY CONFORMED NAME:') and 'company_name' not in metadata:
            metadata['company_name'] = line.split(':', 1)[1].strip()

    return metadata

def extract_10k_text(submission_path: pathlib.Path) -> str:
    """
    Extract the main 10-K document text from full-submission.txt
    """
    content = submission_path.read_text(errors="ignore")

    # Find the main 10-K document (usually the largest HTML section)
    # Look for <DOCUMENT> tags
    import re

    # Split by document sections
    doc_pattern = r'<DOCUMENT>\s*<TYPE>([^<\n]+)\s*<SEQUENCE>(\d+)'
    documents = []

    sections = re.split(r'<DOCUMENT>', content)
    for section in sections[1:]:  # Skip header
        type_match = re.search(r'<TYPE>([^<\n]+)', section)
        if type_match:
            doc_type = type_match.group(1).strip()
            # Look for the main 10-K document
            if doc_type in ['10-K', '10-K/A'] or 'htm' in doc_type.lower():
                documents.append((doc_type, section))

    if not documents:
        # Fallback: return the whole content after header
        header_end = content.find('</SEC-HEADER>')
        if header_end > 0:
            return content[header_end + len('</SEC-HEADER>'):]
        return content

    # Pick the largest document (likely the main 10-K)
    main_doc = max(documents, key=lambda x: len(x[1]))
    doc_content = main_doc[1]

    # Extract HTML content between <html> or <HTML> tags
    html_match = re.search(r'<html[^>]*>(.*?)</html>', doc_content, re.DOTALL | re.IGNORECASE)
    if not html_match:
        html_match = re.search(r'<HTML[^>]*>(.*?)</HTML>', doc_content, re.DOTALL | re.IGNORECASE)

    if html_match:
        html_content = html_match.group(1)
        # Convert HTML to text
        soup = BeautifulSoup(f"<html>{html_content}</html>", "lxml")

        # Drop heavy nav/ix elements and XBRL tags
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        # Handle XBRL and other namespaced tags - unwrap them but keep content
        for el in soup.find_all():
            if el.name and ":" in el.name:
                el.unwrap()

        h = HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        text = h.handle(str(soup))

        # Normalize whitespace
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # If no HTML found, try to extract text after </DESCRIPTION>
    desc_end = doc_content.find('</DESCRIPTION>')
    if desc_end > 0:
        text_start = desc_end + len('</DESCRIPTION>')
        raw_text = doc_content[text_start:].strip()
        # Basic cleanup for non-HTML content
        lines = raw_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and not line.endswith('>'):
                clean_lines.append(line)
        return '\n'.join(clean_lines) if clean_lines else raw_text

    # Fallback: return raw content
    return doc_content

def main():
    # Initialize downloader with the correct API
    # Extract email from USER_AGENT string (e.g., "Name (email@domain.com)" -> "email@domain.com")
    import re
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', USER_AGENT)
    email = email_match.group() if email_match else USER_AGENT

    dl = Downloader("temp_downloads", email_address=email)

    meta_path = OUT_DIR / "filing_meta.jsonl"
    if meta_path.exists():
        backup = OUT_DIR / f"filing_meta_{int(time.time())}.jsonl.bak"
        shutil.copy2(meta_path, backup)

    for ticker, (stem, friendly) in TARGETS.items():
        print(f"[+] {friendly} ({ticker}) — fetching most recent {FORM_TYPE}")

        # sec-edgar-downloader doesn't filter by "filed year" directly,
        # so we download a small number and filter ourselves.
        # Grab up to 5 most recent 10-Ks, then pick the most recent one filed in valid years.
        try:
            # Try the newer API first
            num_downloaded = dl.get(FORM_TYPE, ticker, limit=5)
        except TypeError:
            # Fall back to older API
            num_downloaded = dl.get(FORM_TYPE, ticker, amount=5)

        # The downloader saves files to sec-edgar-filings/{ticker}/{form_type}/
        filing_dir = pathlib.Path("sec-edgar-filings") / ticker / FORM_TYPE

        if not filing_dir.exists():
            print(f"    [error] No filings downloaded for {ticker}. Skipping.")
            continue

        print(f"    [info] Downloaded {num_downloaded} filings to {filing_dir}")

        # Find full-submission.txt files and parse their metadata
        candidates = []
        for submission_file in filing_dir.rglob("full-submission.txt"):
            metadata = parse_submission_metadata(submission_file)
            filed_at = metadata.get("filing_date")
            try:
                filed_year = int(filed_at.split("-")[0]) if filed_at else None
            except Exception:
                filed_year = None

            if filed_year and filed_year in VALID_FILED_YEARS:
                candidates.append((submission_file, metadata))

        if not candidates:
            print(f"    [warn] No {FORM_TYPE} filed in {VALID_FILED_YEARS} found for {ticker}. Keeping the most recent instead.")
            # fall back to newest submission file
            submissions = sorted(filing_dir.rglob("full-submission.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not submissions:
                print("    [error] No full-submission.txt found. Skipping.")
                continue
            submission_file = submissions[0]
            metadata = parse_submission_metadata(submission_file)
        else:
            submission_file, metadata = sorted(candidates, key=lambda x: x[1].get("filing_date",""), reverse=True)[0]

        text = extract_10k_text(submission_file)

        # Write clean text
        out_txt = OUT_DIR / f"{stem}.txt"
        header = []
        header.append(f"__SOURCE_URL__: https://www.sec.gov/Archives/edgar/data/{metadata.get('cik','')}/{metadata.get('accession_number','').replace('-','')}/{metadata.get('accession_number','')}.txt")
        header.append(f"__ACCESSION__: {metadata.get('accession_number','')}")
        header.append(f"__CIK__: {metadata.get('cik','')}")
        header.append(f"__TICKER__: {ticker}")
        header.append(f"__FILING_DATE__: {metadata.get('filing_date','')}")
        header.append(f"__FORM__: {metadata.get('form_type', FORM_TYPE)}")
        header.append(f"__COMPANY__: {metadata.get('company_name','')}")
        out_txt.write_text("\n".join(header) + "\n\n" + text)
        print(f"    [ok] Wrote {out_txt.relative_to(ROOT)}  ({len(text):,} chars)")

        # Write meta
        with open(meta_path, "a") as f:
            rec = {
                "ticker": ticker,
                "name": friendly,
                "form": metadata.get("form_type", FORM_TYPE),
                "accession": metadata.get("accession_number",""),
                "filing_date": metadata.get("filing_date",""),
                "cik": metadata.get("cik",""),
                "company_name": metadata.get("company_name",""),
                "document_url": f"https://www.sec.gov/Archives/edgar/data/{metadata.get('cik','')}/{metadata.get('accession_number','').replace('-','')}/{metadata.get('accession_number','')}.txt",
                "saved_as": str(out_txt.relative_to(ROOT)),
            }
            f.write(json.dumps(rec) + "\n")

    # Cleanup temp cache (optional; comment out if you want to inspect)
    # shutil.rmtree("sec-edgar-filings", ignore_errors=True)
    print("[done] All filings processed.")

if __name__ == "__main__":
    main()
