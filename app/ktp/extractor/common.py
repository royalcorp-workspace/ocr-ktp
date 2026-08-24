import difflib
import re
from typing import Dict, List, Optional

LABELS_PATTERN = {
    "NIK": r'\b(N[\s\._]*[1Il|!i][\s\._]*[Kk]|NIK|N1K|NlK|kK|IK)\b',
    "NAMA": r'\b(N[\s\._]*[A4@a4][\s\._]*[MNmn][\s\._]*[A4@a4]|NAMA|AMA|N4MA|N4NN|NAM4)\b',
    "TEMPAT_TGL_LAHIR": r'\b(T[Ee3aA\[\]\s]*[MSms]?[Pp]?[Aa4@\._]*[Tt]?[/\.,\s]*T[Aa4@nN]*[GGg0]*[LLl1]*[/\.,\s]*L[Aa4@][HHh][Ii1|][RRr]|TEMPAT[/\.,\s]*TGL[/\.,\s]*LAHIR|TEMPAT[/\.,\s]*TANGGAL[/\.,\s]*LAHIR|TEMPAT[/\.,\s]*LAHIR|TEMPAITGILAHIR|TEMPAITGL|TEMPAITGI|TEMPAIT|TEMPATTGL|TEMPAT\s*TG|TESPAT[/\.,\s]*TGI|TGL[/\.,\s]*LAHIR|TTL|empat[/\.,\s]*T|empatt[/\.,\s]*T|empat[/\.,\s]*Latrir|empatt[/\.,\s]*Latrir|tempat[/\.,\s]*Latrir|at/Tgl|t/Tgl|Lah|Lahir)\b',
    "JENIS_KELAMIN": r'\b(J[Ee1i3][Nn1i][Ii1|sS5][\s\._]*K[Ee3][Ll1|][Aa4@][MmNn][Ii1|][Nn]|JENIS[\s\.]*KELAMIN|JENIS[\s\.]*KELAM1N|JENIS[\s\.]*KELAMN|Jora|Joris|Jera|Kelamin|Kelain|Kalinin|JENIS|Jnis|celamin|elamin)\b',
    "GOL_DARAH": r'\b(G[Oo0][Ll1|][\.\s]*D[Aa4@][Rr][Aa4@][Hh]|GOL[\.\s]*DARAH|GOLDARAH)\b',
    "ALAMAT": r'\b([Aa4@][Ll1|][Aa4@][MmNn][Aa4@][Tt1i|]|ALAMAT|ALAMA|ALMA|ALAMAL|ALAMAI|ALAMTI|ALMT|ALAMT|LAMAT|AMAT)\b',
    "RT_RW": r'\b(R[Tt1i|][\s\/\.-]*R[WwVv]|RT[\s\/\.-]*RW|RTIRW|RTRW|RIRW|ATAW|RT/AW|T/RW|T/AW|RT/RAW|RT/RN)\b',
    "KEL_DESA": r'\b(K[ELel1Ick]{1,3}[urahan\._/\s]*D[ESAesa3]{2,3}s?[aei]?|KEL[/\.\s]*DESA|KELLDESA|KELDESA|KELURAHAN[/\.\s]*DESA|DESA[/\.\s]*KELURAHAN|KELURAHAN|DESA|el/Desa|l/Desa)\b',
    "KECAMATAN": r'\b(K[Ee3aA]C[Aa4@]M[Aa4@]T[Aa4@]N|KECAMATAN|Kacamatan|ecamatan|camatan)\b',
    "AGAMA": r'\b([Aa4@]G[Aa4@]M[Aa4@]|AGAMA|Agama|gama)\b',
    "STATUS_PERKAWINAN": r'\b(S[Tt1i][Aa4@]T[Uu][Ss5][\s\.]*P[Ee3][Rr][Kk][Aa4@][WwVv][Ii1|][Nn][Aa4@][Nn]|STATUS[\s\.]*PERKAWINAN|STATUS[\s\.]*PERKAW1NAN|STATUS[\s\.]*PERKABINAN|STATUS[\s\.]*KAWIN)\b',
    "PEKERJAAN": r'\b(P[Ee3][Kk][Ee3][Rr][Jj][Aa4@][Aa4@][Nn]|PEKERJAAN|Pekerjaar|ekerjaan)\b',
    "KEWARGANEGARAAN": r'\b(K[Ee3][WwVv][Aa4@][Rr]G[Aa4@]N[Ee3]G[Aa4@][Rr][Aa4@][Nn]|KEWARGANEGARAAN|KEWARGANEGARAN|KWN|inegaraan)\b',
    "BERLAKU_HINGGA": r'\b(B[Ee3][Rr][Ll1|][Aa4@][Kk][Uu][\s\.]*H[Ii1|][Nn]G[Gg][Aa4@]|BERLAKU[\s\.]*HINGGA|BERLAKU[\s\.]*HINGA|BERLAKU[\s\.]*S\/D|Berlaku Hinggn)\b',
}

DIGIT_MAP = {
    'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'U': '0', 'c': '0',
    'I': '1', 'l': '1', '|': '1', '!': '1', 'L': '1',
    'Z': '2', 'z': '2', 'C': '2',
    'E': '3', 'e': '3',
    'A': '4', 'Y': '4', 'y': '4', 'q': '4',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7', 't': '7',
    'B': '8',
    'q': '9', 'g': '9', 'y': '9'
}

INDONESIAN_CITIES = {
    "BANDUNG", "INDRAMAYU", "SUMEDANG", "JAKARTA", "JAKARTA PUSAT", "JAKARTA SELATAN",
    "JAKARTA BARAT", "JAKARTA TIMUR", "JAKARTA UTARA", "SURABAYA", "SEMARANG", "MEDAN",
    "BOGOR", "BEKASI", "TANGERANG", "TANGERANG SELATAN", "DEPOK", "CIANJUR", "GARUT",
    "TASIKMALAYA", "CIREBON", "SUKABUMI", "PURWAKARTA", "SUBANG", "MAJALENGKA", "KUNINGAN",
    "BREBES", "TEGAL", "BANYUMAS", "CILACAP", "YOGYAKARTA", "SOLO", "SURAKARTA", "MALANG",
    "BALI", "DENPASAR", "PADANG", "PALEMBANG", "LAMPUNG", "BANTEN", "SERANG", "CILEGON",
    "PONTIANAK", "BANJARMASIN", "BALIKPAPAN", "SAMARINDA", "MAKASSAR", "MANADO", "AMBON",
    "JAYAPURA", "MATARAM", "KUPANG", "PEKANBARU", "BATAM", "JAMBI", "BENGKULU", "BANDA ACEH",
    "LHOKSEUMAWE", "BINJAI", "DUMAI", "PAYAKUMBUH", "BUKITTINGGI", "PALU", "GORONTALO",
    "KENDARI", "PAREPARE", "TARAKAN", "BONTANG", "KOTA-SUBANG"
}

KNOWN_KECAMATAN_KELURAHAN = INDONESIAN_CITIES

BOUNDARY_KEYWORDS = [
    r'\bTEMPAT\b', r'\bLAHIR\b', r'\bALAMAT\b', r'\bRT\b', r'\bRW\b',
    r'\bKEL\b', r'\bDESA\b', r'\bKECAMATAN\b', r'\bAGAMA\b', r'\bSTATUS\b',
    r'\bKAWIN\b', r'\bPEKERJAAN\b', r'\bKEWARGANEGARAAN\b', r'\bBERLAKU\b',
    r'\b\d{2}[./-]\d{2}[./-]\d{4}\b'
]

_BLOCK_STOPPERS = [
    r'\bKECAMATAN\b', r'\bAGAMA\b', r'\bSTATUS\b', r'\bPERKAWINAN\b',
    r'\bPEKERJAAN\b', r'\bKEWARGANEGARAAN\b', r'\bBERLAKU\b', r'\bHINGGA\b',
    r'\bKEL\b', r'\bDESA\b', r'\bRT\b', r'\bRW\b',
    r'\bSEUMUR\b', r'\bHIDUP\b', r'\bKAWIN\b', r'\bBELUM\b', r'\bCERAI\b',
    r'\bISLAM\b', r'\bKRISTEN\b', r'\bKATHOLIK\b', r'\bHINDU\b', r'\bBUDDHA\b',
    r'\bLAKI\b', r'\bPEREMPUAN\b',
    r'\bWNI\b', r'\bWNA\b',
    r'\d{2}[./-]\d{2}[./-]\d{4}',
]

_TAIL_NOISE_KEYWORDS = [
    r'\bSEUMUR\s*HIDUP\b',
    r'\bPERKAWINAN\b', r'\bKAWIN\b', r'\bBELUM\b', r'\bCERAI\b',
    r'\bPEKERJAAN\b', r'\bKEWARGANEGARAAN\b', r'\bBERLAKU\b',
    r'\bHINGGA\b', r'\bSTATUS\b',
    r'\bISLAM\b', r'\bKRISTEN\b', r'\bKATHOLIK\b', r'\bHINDU\b', r'\bBUDDHA\b',
    r'\bLAKI\b', r'\bPEREMPUAN\b',
]

_NAMA_STOP_FRAGMENTS = [
    "TEMPAT", "TEMPAL", "TOMPAL", "REMPAT", "TOMPAU", "RAMPAT",
    "EMPAT", "EMPAYT", "EMPATTT", "LATRIR", "LEMPAT",
    "TESPAT", "TEMPAUTGL", "TEMPATTGL", "TESPATTGL", "TESPAUTGL",
    "TEMPATTG", "TEMPAUTG", "TEMPAUT", "TESPAUT",
    "TEMPAITGILAHIR", "TEMPAITGL", "TEMPAITGI", "TEMPAIT", "TEMPATGI",
    "LAHIR", "TTL", "JENIS", "KELAMIN", "KELAM1N", "TAMPANG",
    "INDRAMAYU", "SUMEDANG", "BANDUNG", "GARUT", "CIAMIS", "CIMAHI",
    "GOL", "DARAH", "GOLDARAH",
    "ALAMAT", "RT", "RW", "KEL", "DESA", "KECAMATAN",
    "AGAMA", "STATUS", "KAWIN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU",
]


def parse_label_blocks(full_text: str) -> Dict[str, str]:
    raw_matches = []
    for key, pattern in LABELS_PATTERN.items():
        for m in re.finditer(pattern, full_text, flags=re.IGNORECASE):
            raw_matches.append((m.start(), m.end(), key))

    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    filtered_matches = []
    last_end = -1
    for start, end, key in raw_matches:
        if start >= last_end:
            filtered_matches.append((start, end, key))
            last_end = end

    blocks = {}
    for idx, (start, end, key) in enumerate(filtered_matches):
        val_start = end
        val_end = filtered_matches[idx + 1][0] if idx + 1 < len(filtered_matches) else len(full_text)
        raw_block = full_text[val_start:val_end].strip()

        # Restrict raw_block to first line for single-line fields to prevent multi-line contamination
        if key not in {"ALAMAT", "TEMPAT_TGL_LAHIR", "RT_RW"}:
            first_line = raw_block.splitlines()[0] if raw_block.splitlines() else raw_block
            cleaned_block = re.sub(r'^[\s:=]+', '', first_line).strip()
        else:
            lines = [l.strip() for l in raw_block.splitlines() if l.strip()]
            valid_lines = []
            for l_idx, l in enumerate(lines[:2]):
                l_up = l.upper()
                stoppers_to_check = [s for s in _BLOCK_STOPPERS if s != r'\d{2}[./-]\d{2}[./-]\d{4}'] if key == "TEMPAT_TGL_LAHIR" else _BLOCK_STOPPERS
                if any(re.search(stopper, l_up) for stopper in stoppers_to_check):
                    break
                
                # Masalah 1: Strict label boundary for multi-line RT_RW
                if key == "RT_RW" and l_idx == 1:
                    other_labels = [r'\bTEMPAT\b', r'\bTGL\b', r'\bLAHIR\b', r'\bALAMAT\b', r'\bNAMA\b', r'\bNIK\b', r'\bKECAMATAN\b', r'\bKEL\b', r'\bDESA\b', r'\bAGAMA\b', r'\bSTATUS\b', r'\bPEKERJAAN\b']
                    if any(re.search(lbl, l_up) for lbl in other_labels):
                        break
                        
                valid_lines.append(l)
            cleaned_block = re.sub(r'^[\s:=]+', '', "\n".join(valid_lines)).strip()

        if key not in blocks or len(cleaned_block) > len(blocks[key]):
            blocks[key] = cleaned_block

    return blocks


def clean_symbol_prefix(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^[^\w\s]+', '', text).strip()
    text = re.sub(r'^[\s:\.=\-_~#\|!?]+', '', text).strip()
    text = re.sub(r'^(NP|CEE|REEF)\s+', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^[\s:\.=\-_~#\|!?]+', '', text).strip()
    return text


def sanitize_block_text(block: str, max_lines: int = 1) -> str:
    if not block:
        return ""

    raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
    valid_lines: List[str] = []

    for line in raw_lines:
        if len(valid_lines) >= max_lines:
            break
        if re.match(r'^\d+\s+', line):
            continue

        line_upper = line.upper()
        earliest_cut = len(line)
        for stopper in _BLOCK_STOPPERS:
            m = re.search(stopper, line_upper)
            if m and m.start() < earliest_cut:
                earliest_cut = m.start()

        if earliest_cut < len(line):
            line = line[:earliest_cut].strip()

        line = re.sub(r'[\s\-]+$', '', line).strip()

        if line:
            valid_lines.append(line)

    return " ".join(valid_lines).strip()


def strip_tail_noise(val: str) -> str:
    if not val:
        return val

    val = re.sub(r'\s*\d{2}[./-]\d{2}[./-]\d{4}\s*$', '', val).strip()

    changed = True
    while changed:
        changed = False
        for kw_pat in _TAIL_NOISE_KEYWORDS:
            m = re.search(kw_pat + r'\s*$', val, re.IGNORECASE)
            if m:
                val = val[:m.start()].strip()
                changed = True

    val = re.sub(r'\s*\d{2}[./-]\d{2}[./-]\d{4}\s*$', '', val).strip()

    trailing_city = re.search(r'\s+([A-Z]{4,})\s*$', val)
    if trailing_city and len(val[:trailing_city.start()].strip()) >= 3:
        city_candidate = trailing_city.group(1)
        protected_words = {
            "JALAN", "GANG", "BLOK", "PERUMAHAN", "KOMPLEK", "PERUM", "DUSUN", "KAMPUNG", "GRIYA",
            "TANGGA", "RUMAH", "HARIAN", "LEPAS", "SWASTA", "RAYA", "UTARA", "SELATAN", "BARAT",
            "TIMUR", "TENGAH", "ASRI", "INDAH", "AGUNG", "JAYA", "MEKAR", "MULYA", "MUKTI",
            "WETAN", "KULON", "KIDUL", "LOR", "KALER", "GIRANG", "HILIR", "SARI", "SEJAHTERA",
            "PASIR", "KOTA", "PARIGI"
        }
        if city_candidate not in protected_words:
            val = val[:trailing_city.start()].strip()

    return val


def correct_known_location_fuzzy(val: str, cutoff: float = 0.80) -> str:
    if not val or len(val) < 4:
        return val

    clean_v = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', val).strip().upper()
    if not clean_v or len(clean_v) < 4:
        return val

    non_alpha = sum(1 for c in clean_v if not c.isalpha() and not c.isspace())
    if non_alpha / len(clean_v) > 0.25:
        return val

    if clean_v in KNOWN_KECAMATAN_KELURAHAN:
        return clean_v

    words = clean_v.split()
    
    best_match = None
    best_score = 0.0

    for known_loc in KNOWN_KECAMATAN_KELURAHAN:
        known_words = known_loc.split()
        
        # Word count must match exactly
        if len(words) != len(known_words):
            continue
            
        is_valid = True
        total_ratio = 0.0
        
        for w, k_w in zip(words, known_words):
            # Strict length check for each word
            if len(w) < 0.5 * len(k_w):
                is_valid = False
                break
                
            # Exact match is always valid
            if w == k_w:
                total_ratio += 1.0
                continue
                
            ratio = difflib.SequenceMatcher(None, w, k_w).ratio()
            if ratio < cutoff:
                is_valid = False
                break
                
            total_ratio += ratio
            
        if is_valid:
            avg_ratio = total_ratio / len(words)
            if avg_ratio > best_score:
                best_score = avg_ratio
                best_match = known_loc

    if best_match:
        return best_match

    return val


def extract_text_field(block: Optional[str]) -> Optional[str]:
    """Generic field extractor: dipakai untuk kelurahan_desa, kecamatan, dan basis pekerjaan."""
    if not block:
        return None

    sanitized = sanitize_block_text(block, max_lines=1)
    if not sanitized:
        return None

    val = clean_symbol_prefix(sanitized)
    val = re.sub(r'^(KECAMATAN|KACAMATAN|KECAMETAN|KELURAHAN|DESA|KEL|KEC|NAMA|NIK|AGAMA|PEKERJAAN|STATUS)[\s:\.=-]+', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'[^\w\s\.\/\'-]', '', val).strip().upper()
    val = re.sub(r'^[\s:\.=-]+', '', val).strip()
    val = re.sub(r'\s*[_=]+\s*', ' ', val)
    val = re.sub(r'\s{2,}', ' ', val).strip()

    # Clean trailing noise symbols (e.g. "-R .", "- .", ".")
    val = re.sub(r'[\s\.\/\'\"\-:=_\*#\|!\?\+~]+$', '', val).strip()

    # Try fuzzy matching against known location names (cutoff 0.72 since this is anchored)
    fuzzy_val = correct_known_location_fuzzy(val, cutoff=0.72)
    if fuzzy_val != val and len(fuzzy_val) >= 4:
        return fuzzy_val

    tokens = val.split()
    if len(tokens) > 1:
        last_tok = tokens[-1]
        if len(last_tok) == 1 and last_tok not in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            tokens.pop()
            val = " ".join(tokens).strip()

    val = strip_tail_noise(val)
    val = correct_known_location_fuzzy(val, cutoff=0.72)

    return val if val else None