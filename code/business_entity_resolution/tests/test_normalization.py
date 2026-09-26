import pytest
import pandas as pd
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers

def test_normalize_name():
    assert normalize_name("Acme Corp.") == "acme"
    assert normalize_name("Acme Corporation") == "acme"
    assert normalize_name("Smith & Wesson") == "smith and wesson"
    assert normalize_name(" Café Mõcha ") == "cafe mocha" # Unicode transliteration
    assert normalize_name("Google.com") == "google com"
    assert normalize_name("--Invalid Name") == "invalid name"
    assert normalize_name(None) == ""
    assert normalize_name(float('nan')) == ""

def test_normalize_address():
    assert normalize_address("123 Main St.") == "123 main street"
    assert normalize_address("456 1st Ave") == "456 1 avenue"
    assert normalize_address("P.O. BOX 123") == "pobox 123"
    assert normalize_address("Apartment 4B, Building C") == "apartment 4b building c"
    assert normalize_address(None) == ""

def test_extract_numbers():
    assert extract_numbers("123 main street apt 4") == "123 4"
    assert extract_numbers("no numbers here") == ""
    assert extract_numbers("456 1 avenue") == "456 1"
