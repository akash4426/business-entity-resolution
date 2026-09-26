import re
import unicodedata
from unidecode import unidecode
import pandas as pd

# --- Name Normalization ---

LEGAL_SUFFIXES_RE = re.compile(
    r'\b(inc|incorporated|corp|corporation|ltd|limited|pvt|private|llc|llp|plc|gmbh|sarl|sas|sa|co|company|assoc|associates|partners|group|holdings|enterprises|services|solutions|technologies|technology|industries|consultants|international|dba|d/b/a)\b\.?',
    flags=re.IGNORECASE
)

def normalize_name(name: str) -> str:
    if pd.isna(name) or name is None or str(name).strip() == '':
        return ''
    
    # 1. Unicode & Transliteration (Devanagari -> Latin etc.)
    name = str(name)
    name = unidecode(name)
    
    # 2. Lowercase
    name = name.lower()
    
    # 3. & / and normalization
    name = name.replace('&', ' and ')
    
    # 4. Remove special leading patterns like <<, --
    name = re.sub(r'^[^a-z0-9]+', '', name)
    
    # 5. Domain suffix handling (e.g. .com, .net) - replace dot with space
    name = re.sub(r'\.(com|net|org|io|co|in)\b', r' \1', name)
    
    # 6. Punctuation normalization - replace with space
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # 7. Legal suffix removal / standardization
    # Since we want to compare names without legal suffix noise, it's often better to remove them completely
    # or map them to standard tokens. We'll remove them for the base normalized name.
    name = LEGAL_SUFFIXES_RE.sub(' ', name)
    
    # 8. Whitespace normalization
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def tokenize_name(name: str) -> list:
    norm = normalize_name(name)
    return norm.split() if norm else []


# --- Address Normalization ---

ADDRESS_ABBREV_MAP = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bp\s*o\s*box\b': 'pobox'
}

def normalize_address(address: str) -> str:
    if pd.isna(address) or address is None or str(address).strip() == '':
        return ''
        
    address = str(address)
    address = unidecode(address).lower()
    
    # Punctuation to space
    address = re.sub(r'[^\w\s]', ' ', address)
    
    # Abbreviations
    for pat, repl in ADDRESS_ABBREV_MAP.items():
        address = re.sub(pat, repl, address)
        
    # Remove ordinal indicators (1st -> 1, 2nd -> 2)
    address = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', address)
    
    # Whitespace normalization
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address

def extract_numbers(address: str) -> str:
    """Extract all digit sequences from address, useful for building number match."""
    if not address:
        return ''
    numbers = re.findall(r'\d+', address)
    return ' '.join(numbers)

def tokenize_address(address: str) -> list:
    norm = normalize_address(address)
    return norm.split() if norm else []

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with 'business_name' and 'business_address' and adds normalized columns.
    Uses polars for speed if possible, but pandas `apply` is okay if string operations are fast enough.
    For 5M rows, pandas apply might take a couple of minutes. 
    """
    if 'business_name' in df.columns:
        df['name_norm'] = df['business_name'].fillna('').apply(normalize_name)
    
    if 'business_address' in df.columns:
        df['address_norm'] = df['business_address'].fillna('').apply(normalize_address)
        df['address_numbers'] = df['address_norm'].apply(extract_numbers)
        
    return df
