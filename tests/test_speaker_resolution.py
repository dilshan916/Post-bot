import pytest
from src.utils import resolve_speaker_category

def test_resolve_speaker_category_standard():
    assert resolve_speaker_category("MALE") == "MALE"
    assert resolve_speaker_category("FEMALE") == "FEMALE"
    assert resolve_speaker_category("OLD_MALE") == "OLD_MALE"
    assert resolve_speaker_category("OLD_FEMALE") == "OLD_FEMALE"
    assert resolve_speaker_category("CHILD_MALE") == "CHILD_MALE"
    assert resolve_speaker_category("CHILD_FEMALE") == "CHILD_FEMALE"
    assert resolve_speaker_category("CHIBI_MALE") == "CHILD_MALE"
    assert resolve_speaker_category("CHIBI_FEMALE") == "CHILD_FEMALE"

def test_resolve_speaker_category_special_roles():
    assert resolve_speaker_category("INTRO") == "INTRO"
    assert resolve_speaker_category("OUTRO") == "OUTRO"
    assert resolve_speaker_category("NARRATOR") == "NARRATOR"

def test_resolve_speaker_category_custom():
    # Parents / Elderly
    assert resolve_speaker_category("Mom") == "OLD_FEMALE"
    assert resolve_speaker_category("Mother") == "OLD_FEMALE"
    assert resolve_speaker_category("Dad") == "OLD_MALE"
    assert resolve_speaker_category("Father") == "OLD_MALE"
    assert resolve_speaker_category("Grandpa") == "OLD_MALE"
    assert resolve_speaker_category("Grandmother") == "OLD_FEMALE"
    assert resolve_speaker_category("Stepdad") == "OLD_MALE"
    assert resolve_speaker_category("Stepmom") == "OLD_FEMALE"
    assert resolve_speaker_category("MIL") == "OLD_FEMALE"
    assert resolve_speaker_category("FIL") == "OLD_MALE"
    
    # Children
    assert resolve_speaker_category("Little Boy") == "CHILD_MALE"
    assert resolve_speaker_category("Toddler Girl") == "CHILD_FEMALE"
    assert resolve_speaker_category("Kid") == "CHILD_FEMALE"  # defaults to female if no male indicator
    assert resolve_speaker_category("Male Kid") == "CHILD_MALE"
    assert resolve_speaker_category("Baby boy") == "CHILD_MALE"
    
    # Standard adults
    assert resolve_speaker_category("Husband") == "MALE"
    assert resolve_speaker_category("Wife") == "FEMALE"
    assert resolve_speaker_category("Boyfriend") == "MALE"
    assert resolve_speaker_category("Girlfriend") == "FEMALE"
    assert resolve_speaker_category("Brother") == "MALE"
    assert resolve_speaker_category("Sister") == "FEMALE"
    assert resolve_speaker_category("Dave") == "FEMALE"  # default fallback
    assert resolve_speaker_category("Alex") == "FEMALE"  # default fallback
    
    # User numbers
    assert resolve_speaker_category("USER_1") == "MALE"  # 1 is odd -> male
    assert resolve_speaker_category("USER_2") == "FEMALE"  # 2 is even -> female
    assert resolve_speaker_category("USER_3") == "MALE"
