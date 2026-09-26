import re
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz

LEGAL_SUFFIXES_RE = re.compile(
    r'\b(inc|incorporated|corp|corporation|ltd|limited|pvt|private|llc|llp|plc|gmbh|sarl|sas|sa|co|company|assoc|associates|partners|group|holdings|enterprises|services|solutions|technologies|technology|industries|consultants|international|dba|d/b/a)\b\.?',
    flags=re.IGNORECASE
)

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

def normalize_name_v2(name: str):
    if pd.isna(name) or name is None or str(name).strip() == '':
        return {
            'name_raw': '',
            'name_unicode': '',
            'name_lower': '',
            'name_punct': '',
            'name_legal': '',
            'name_tokens': [],
            'name_sorted': ''
        }
    
    name_str = str(name)
    name_uni = unidecode(name_str)
    name_lower = name_uni.lower()
    
    # Punctuation to space
    name_punct = re.sub(r'[^\w\s]', ' ', name_lower)
    name_punct = re.sub(r'\s+', ' ', name_punct).strip()
    
    # Legal suffix removal
    name_legal = LEGAL_SUFFIXES_RE.sub(' ', name_punct)
    name_legal = re.sub(r'\s+', ' ', name_legal).strip()
    
    tokens = name_legal.split()
    name_sorted = ' '.join(sorted(tokens))
    
    return {
        'name_raw': name_str,
        'name_unicode': name_uni,
        'name_lower': name_lower,
        'name_punct': name_punct,
        'name_legal': name_legal,
        'name_tokens': tokens,
        'name_sorted': name_sorted
    }

def normalize_address_v2(address: str):
    if pd.isna(address) or address is None or str(address).strip() == '':
        return {
            'address_raw': '',
            'address_lower': '',
            'address_punct': '',
            'address_norm': '',
            'address_numbers': '',
            'address_tokens': [],
            'postal_code': ''
        }
        
    addr_str = str(address)
    addr_lower = unidecode(addr_str).lower()
    
    addr_punct = re.sub(r'[^\w\s]', ' ', addr_lower)
    addr_punct = re.sub(r'\s+', ' ', addr_punct).strip()
    
    addr_norm = addr_punct
    for pat, repl in ADDRESS_ABBREV_MAP.items():
        addr_norm = re.sub(pat, repl, addr_norm)
        
    addr_norm = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', addr_norm)
    addr_norm = re.sub(r'\s+', ' ', addr_norm).strip()
    
    numbers = re.findall(r'\d+', addr_norm)
    address_numbers = ' '.join(numbers)
    
    tokens = addr_norm.split()
    
    # Simple postal code extraction (5 or 6 digits)
    postal_code = ''
    postal_match = re.search(r'\b\d{5,6}\b', addr_norm)
    if postal_match:
        postal_code = postal_match.group(0)
    
    return {
        'address_raw': addr_str,
        'address_lower': addr_lower,
        'address_punct': addr_punct,
        'address_norm': addr_norm,
        'address_numbers': address_numbers,
        'address_tokens': tokens,
        'postal_code': postal_code
    }

def process_batch(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized execution over rows using apply (returns series of dicts)
    # It's faster to do this via map in pure python list comprehension
    
    if 'business_name' in df.columns:
        names = df['business_name'].tolist()
        res_names = [normalize_name_v2(n) for n in names]
        for k in res_names[0].keys():
            df[k] = [r[k] for r in res_names]
            
    if 'business_address' in df.columns:
        addrs = df['business_address'].tolist()
        res_addrs = [normalize_address_v2(a) for a in addrs]
        for k in res_addrs[0].keys():
            df[k] = [r[k] for r in res_addrs]
            
    if 'country' in df.columns:
        # Normalize country to handle missing/uppercase/whitespace
        df['country_norm'] = df['country'].fillna('').astype(str).str.lower().str.strip()
            
    return df
