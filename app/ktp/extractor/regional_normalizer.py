import re
import difflib
from difflib import SequenceMatcher
from typing import Optional

# List of known regional villages/districts for typo correction (e.g., Bandung / West Java regencies)
KNOWN_VILLAGES = [
    "SUKAMULYA", "BOJONGSALAM", "LENGKONG", "RANCAEKEK WETAN", "RANCAEKEK KULON",
    "RANCABEUREUM", "BOJONGMANGGU", "CILAMPENI", "SAYATI", "MARGAHAUYU",
    "DAYEUHKOLOT", "BALEENDAH", "CIPARAY", "MAJALAYA", "PENGALENGAN"
]

KNOWN_KECAMATAN = [
    "RANCAEKEK", "KATAPANG", "MARGAHAYU", "DAYEUHKOLOT", "BALEENDAH",
    "BOJONGSOANG", "CIPARAY", "MAJALAYA", "PASIRJAMBU", "CIWIDEY", "CIMAUNG"
]

def normalize_regional_text(text: Optional[str], field_type: str = "village") -> Optional[str]:
    """
    Fuzzy repair for kelurahan/desa, kecamatan, and address fragments based on regional gazetteers.
    Rejects false misread headers like 'LENIS' (from JENIS KELAMIN).
    """
    if not text or not text.strip():
        return None

    cleaned = text.upper().strip()
    cleaned = re.sub(r'[\s:._=\-]+', ' ', cleaned).strip()

    # Reject false misread headers
    if cleaned in ["LENIS", "FANE KELAMIN", "KELAMIN", "TAKILAKI", "PEREMPUAN"]:
        return None

    # Generic fuzzy match against gazetteers
    target_dict = KNOWN_VILLAGES if field_type in ["village", "kelurahan_desa"] else (KNOWN_KECAMATAN if field_type == "kecamatan" else KNOWN_VILLAGES + KNOWN_KECAMATAN)
    
    if cleaned in target_dict:
        return cleaned

    filtered_cand = [w for w in target_dict if abs(len(w) - len(cleaned)) <= 3]
    if filtered_cand:
        matches = difflib.get_close_matches(cleaned, filtered_cand, n=1, cutoff=0.75)
        if matches:
            return matches[0]

    return cleaned
