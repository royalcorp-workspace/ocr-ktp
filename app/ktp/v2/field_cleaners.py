import re
from typing import Optional, List

def clean_text(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    text = str(raw_text).upper().strip()
    # Strip leading/trailing colon, dash, quotes, noise symbols
    text = re.sub(r'^[:\-–—"\':;\.>\s]+', '', text)
    text = re.sub(r'[:\-–—"\':;\.>\s]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def clean_rt_rw(raw_val: Optional[str]) -> Optional[str]:
    s = clean_text(raw_val)
    if not s:
        return None
    s = re.sub(r'\s*/\s*', '/', s)
    return s if s else None


def clean_nik(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    # Clean noise labels
    s = re.sub(r'^(NIK|HIK|MIL|N|I|K)[:\s]*', '', s)
    # OCR typo fixes for numbers
    s = s.replace('O', '0').replace('D', '0').replace('Q', '0')
    s = s.replace('I', '1').replace('L', '1').replace('l', '1').replace('|', '1').replace('!', '1')
    s = s.replace('B', '8').replace('S', '5').replace('Z', '2')
    
    digits = re.sub(r'[^\d]', '', s)
    if len(digits) == 16:
        return digits
    elif len(digits) > 16:
        # Check if first 16 or last 16 make sense
        return digits[:16]
    return digits if digits else None

def clean_date(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    # Regex match for DD-MM-YYYY or DD MM YYYY or DD/MM/YYYY
    match = re.search(r'(\d{2})[\s\-\./]+(\d{2})[\s\-\./]+(\d{4})', s)
    if match:
        d, m, y = match.groups()
        return f"{d}-{m}-{y}"
    return None

def clean_gender(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper()
    if 'LAKI' in s or 'LAK' in s or 'PERIA' in s:
        return 'LAKI-LAKI'
    elif 'PEREMP' in s or 'PEREM' in s or 'WANITA' in s:
        return 'PEREMPUAN'
    return None

def clean_blood_type(raw_val: Optional[str]) -> Optional[str]:
    """
    Returns 'A', 'B', 'AB', 'O' if valid blood type is detected.
    Returns None if unreadable, empty, or '-'.
    Aligns 100% with State Matrix Section 5 (no forced '-' fallback values).
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    # Strip labels like GOL. DARAH :
    s = re.sub(r'^(GOL|DARAH|GOL\.?\s*DARAH)[:\s]*', '', s)
    s = re.sub(r'[^A-Z]', '', s)
    
    if s == 'AB':
        return 'AB'
    elif 'A' in s:
        return 'A'
    elif 'B' in s:
        return 'B'
    elif 'O' in s or '0' in s:
        return 'O'
    return None

def clean_marital_status(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper()
    if 'BELUM' in s:
        return 'BELUM KAWIN'
    elif 'KAWIN' in s or 'KAW' in s:
        return 'KAWIN'
    elif 'CERAI HIDUP' in s:
        return 'CERAI HIDUP'
    elif 'CERAI MATI' in s:
        return 'CERAI MATI'
    return None

def clean_citizenship(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper()
    if 'WNI' in s or 'INDONESIA' in s:
        return 'WNI'
    elif 'WNA' in s:
        return 'WNA'
    return None


# ─── Pekerjaan Tokenizer ────────────────────────────────────────────────────────
# Kamus kata pekerjaan umum KTP, diurutkan panjang → pendek agar greedy
# longest-match bekerja benar (misal: BURUHHARIANLEPAS → BURUH HARIAN LEPAS).
_PEKERJAAN_VOCAB: List[str] = sorted([
    "BURUH HARIAN LEPAS", "BURUH HARIAN", "BURUH TANI",
    "PELAJAR/MAHASISWA", "PELAJAR", "MAHASISWA",
    "PEGAWAI NEGERI SIPIL", "APARATUR SIPIL NEGARA",
    "IBU RUMAH TANGGA", "TIDAK BEKERJA",
    "KARYAWAN SWASTA", "KARYAWAN",
    "TNI", "POLRI", "PNS", "ASN",
    "PENSIUNAN", "WIRASWASTA", "SWASTA",
    "PETANI", "NELAYAN", "PEDAGANG",
    "DOKTER", "PERAWAT", "BIDAN",
    "GURU", "DOSEN",
    "SOPIR", "SUPIR", "OJEK",
    "BURUH", "HARIAN", "LEPAS",
    "PEGAWAI", "LAINNYA", "IRT",
], key=len, reverse=True)

def tokenize_pekerjaan(raw_val: Optional[str]) -> Optional[str]:
    """Memecah teks pekerjaan yang digabung tanpa spasi (misal BURUHHARIANLEPAS)
    menggunakan greedy longest-match terhadap kamus kata pekerjaan."""
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    # Jika sudah mengandung spasi/slash, kembalikan langsung
    if ' ' in s or '/' in s:
        return s
    # Greedy longest-match terhadap _PEKERJAAN_VOCAB
    result_tokens: List[str] = []
    remaining = s
    max_iter = 30
    iteration = 0
    while remaining and iteration < max_iter:
        iteration += 1
        matched = False
        for vocab_word in _PEKERJAAN_VOCAB:
            vw_no_space = vocab_word.replace(' ', '').replace('/', '')
            if remaining.startswith(vw_no_space):
                result_tokens.append(vocab_word)
                remaining = remaining[len(vw_no_space):]
                matched = True
                break
        if not matched:
            # Tidak ada match — pertahankan sisa string as-is
            result_tokens.append(remaining)
            break
    result = ' '.join(result_tokens).strip()
    result = re.sub(r'\s+', ' ', result)
    return result if result else s


# ─── Regional Name Normalizer ────────────────────────────────────────────────────
def normalize_regional(raw_val: Optional[str]) -> Optional[str]:
    """Normalisasi nama wilayah Indonesia:
    - Q → C (nama wilayah RI tidak mengandung Q asli, OCR sering salah baca C sebagai Q)
    Contoh: RANQAEKEK → RANCAEKEK
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    s = s.replace('Q', 'C')
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None


# ─── Compound Regional Name Tokenizer ────────────────────────────────────────────
# Suffix umum nama kelurahan/kecamatan/kota Indonesia yang sering digabung OCR
_REGIONAL_SUFFIXES: List[str] = sorted([
    "WETAN", "KULON", "KIDUL", "LOR",
    "BARAT", "TIMUR", "UTARA", "SELATAN", "TENGAH",
    "PERMAI", "INDAH", "BARU", "LAMA", "JAYA", "MAKMUR", "SEJAHTERA",
    "BLOK", "BLOKE", "BLOK", "NO", "RT", "RW",
    "KP", "KP.", "JL", "JL.",
], key=len, reverse=True)

def tokenize_compound_name(raw_val: Optional[str]) -> Optional[str]:
    """Memisahkan nama wilayah/alamat yang digabung OCR dalam 1 bounding box.
    Contoh: RANCAEKEKWETAN → RANCAEKEK WETAN
    Strategi: cari suffix wilayah umum, sisipkan spasi di titik pemisah.
    Fallback: kembalikan string asli jika tidak ada match.
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()

    # Jika sudah ada spasi, kembalikan langsung
    if ' ' in s:
        return s

    # Coba match suffix dari belakang
    for suffix in _REGIONAL_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            base = s[: -len(suffix)].strip()
            if base:
                return f"{base} {suffix}"

    return s
