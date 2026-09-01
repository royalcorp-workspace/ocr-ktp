import re
from typing import Optional, List

_V2_NAME_LEXICON = {
    "ABADI", "ABDUL", "ABDULLAH", "ADE", "ADAM", "ADI", "ADITA", "ADITYA", "ADRIAN", "AGUS",
    "AGUSTINA", "AGUSTINUS", "AGUNG", "AHMAD", "AINI", "AKBAR", "AKMAD", "ALAM", "ALAMSALAH",
    "ALBERT", "ALDI", "ALEX", "ALEXANDER", "ALFIAN", "ALI", "ALICIA", "ALIF", "ALFI",
    "AMALIA", "AMANDA", "AMBAR", "AMELIA", "AMIN", "AMINAH", "AMIR", "AMANDA", "ANANDA",
    "ANDI", "ANDIKA", "ANDRE", "ANDRI", "ANDRIAWAN", "ANDRIANI", "ANDRIAN",
    "ANGGA", "ANGGI", "ANGGRAENI", "ANGGITA", "ANISA", "ANITA", "ANNISA", "ANTON", "ANTONIUS",
    "ANWAR", "APRIANI", "APRILLIA", "APRIYANTO", "AQIL", "ARDHI", "ARDI", "ARDIAN", "ARDIANSYAH",
    "ARIEF", "ARIF", "ARIFIN", "ARIS", "ARI", "ARIYANTO", "ARMA", "ARMAND", "ARMAN", "ARYA",
    "ASRI", "ASTUTI", "ATIN", "AULIA", "AVIAN", "AYU", "AZIZ", "AZIZAH", "BAGUS", "BAHRUDIN",
    "BAMBANG", "BASUKI", "BAYU", "BENNY", "BERTI", "BIMA", "BINTANG", "BUDI", "BUDIMAN",
    "BURHAN", "CAHYO", "CAHYONO", "CHANDRA", "CHRISTIAN", "CHRISTINA", "DANIEL", "DANANG",
    "DARMA", "DARMAWAN", "DAVID", "DAYAT", "DEDE", "DEDEN", "DEDI", "DELIA", "DENI", "DENNY",
    "DESI", "DESSY", "DEVI", "DEVITA", "DEWI", "DHANI", "DIAH", "DIAN", "DIANA", "DICKY",
    "DIDIK", "DIKY", "DINA", "DINDA", "DIRGA", "DITA", "DONI", "DONNY", "DWI", "EKO", "ELI",
    "ELISA", "ELISABETH", "ELSA", "ELVI", "EMMA", "ENDANG", "ERIF", "ERIK", "ERNA", "ERNI",
    "ERWIN", "EVA", "EVI", "FACHRI", "FADLI", "FAIRUZ", "FAISAL", "FAJAR", "FAJRI", "FANNY",
    "FARAH", "FARHAN", "FARID", "FARIDA", "FEBRI", "FEBRIANI", "FEBRIANTO", "FEBY", "FELIX",
    "FERDI", "FERDIAN", "FERDINAND", "FERRY", "FIKRI", "FINA", "FITRI", "FITRIA", "FITRIANI",
    "FRANS", "FRANSISKA", "GABRIEL", "GALANG", "GALIH", "GANI", "GILANG", "GITA", "GRACE",
    "GUNAWAN", "GUSTI", "HADI", "HAFIDZ", "HAFIS", "HAIRUL", "HAKIM", "HALIM", "HAMZAH",
    "HANA", "HANAFI", "HANDOKO", "HANDAYANI", "HANIF", "HANIFAH", "HARDI", "HARDIANTO",
    "HAFIZ", "HARIS", "HARIYANTO", "HARTONO", "HARTATI", "HARUN", "HASAN", "HASANAH",
    "HASANUDDIN", "HASTUTI", "HENDRA", "HENDRI", "HENDRIK", "HENDRO", "HERI", "HERIANTO",
    "HERMAN", "HERMANSYAH", "HERU", "HIDAYAT", "HIDAYATULLOH", "HIDAYATULLAH", "HILMAN",
    "HIDAYATI", "HUSEN", "HUSIN", "HUSNA", "IBRAHIM", "IDA", "IGNATIUS", "IIN", "IKBAL",
    "ILHAM", "IMAM", "IMAN", "IMMANUEL", "INDAH", "INDRA", "INDIRA", "INDRI", "INDRIANI",
    "IRFAN", "IRMA", "IRWAN", "IRWANSYAH", "ISMAIL", "ISNA", "ISNAINI", "IVA", "IVAN",
    "IWAN", "IZZA", "JAENAL", "JAFFAR", "JAMAL", "JEFRI", "JEFRY", "JOHAN", "JOHANES",
    "JOKO", "JOSEPH", "JOSUA", "JULIA", "JULIAN", "JULIANA", "JULIANTO", "JUNAEDI",
    "JUNAIDI", "KAREN", "KARTIKA", "KASIH", "KEVIN", "KHAERUL", "KHALID", "KHOIRUL",
    "KIKI", "KRISNA", "KRISTIAN", "KUNTO", "KURNIA", "KURNIAWAN", "KUSMANI", "KUSUMA",
    "KUSUMAH", "LALA", "LATIF", "LEO", "LEONARD", "LESTARI", "LIA", "LILIS", "LINA",
    "LINDA", "LITA", "LUKMAN", "LUTFI", "MADS", "MAHESA", "MAHFUD", "MAHMUD", "MAJID",
    "MALIK", "MARCEL", "MARIA", "MARIANA", "MARINA", "MARINO", "MARIO", "MARKUS", "MARLIANA",
    "MARTA", "MARTIN", "MARTINA", "MARTINU", "MARTINUS", "MARWAN", "MARYANI", "MARYATI",
    "MAULANA", "MAULIDA", "MAYA", "MEGA", "MEGI", "MELANI", "MELIA", "MELINDA", "MELISA",
    "MERI", "MIA", "MICHAEL", "MICHELLE", "MIFTAH", "MIRA", "MITA", "MOCH", "MOCHAMAD",
    "MOCHAMMAD", "MOH", "MOHAMAD", "MOHAMMAD", "MONICA", "MUCHAMAD", "MUDI", "MUH",
    "MUHAMAD", "MUHAMMAD", "MULYADI", "MULYANA", "MULYANI", "MULYONO", "MUNANDAR",
    "MUTIA", "MUTIARA", "NADIA", "NABILA", "NADYA", "NANA", "NANDAR", "NANANG", "NASRULLAH",
    "NASUTION", "NATALIA", "NAVAL", "NAUFAL", "NAWI", "NILA", "NINA", "NINDYA", "NITA",
    "NOFI", "NOPRI", "NOVAL", "NOVIA", "NOVIANI", "NOVIANTO", "NOVITA", "NUGROHO", "NUR",
    "NURAENI", "NURUL", "NURHADI", "NURHASANAH", "NURHAYATI", "NURDIN", "OKTA", "OKTAVIA",
    "OKTAVIANI", "OKTAVIANUS", "PANJI", "PARKESIT", "PADIL", "PAUD", "PAULUS", "PETRUS",
    "PRADANA", "PRAKOSO", "PRASETYO", "PRATAMA", "PRATIWI", "PRAYOGO", "PRIYANTO", "PUJI",
    "PURWANTO", "PURWANTI", "PUTRA", "PUTRI", "RACHMAN", "RACHMAT", "RADEN", "RAHMAD",
    "RAHMAN", "RAHMAWATI", "RAHMAT", "RAIHAN", "RAKA", "RAMA", "RAMADHAN", "RAMDANI",
    "RAQIL", "RANI", "RATNA", "RAYA", "REFA", "REIF", "REJEKI", "RENITA", "RENGGA", "RENI", "RENO",
    "REZA", "RHEZA", "RIA", "RIAN", "RIANI", "RICARDO", "RICHARD", "RICKY", "RICO",
    "RIDWAN", "RIEKY", "RIFKI", "RIFKHI", "RIKI", "RIKY", "RIMA", "RINA", "RINDRA", "RINI",
    "RIO", "RISA", "RISHAD", "RISKA", "RISKI", "RISMA", "RISNANDAR", "RISKY", "RITA",
    "RIVALDI", "RIVAI", "RIZAL", "RIZKI", "RIZKY", "RIZQ", "ROBERT", "ROBY", "ROBBY",
    "ROCHMAN", "RODI", "ROSLINA", "RODIAT", "ROHMAT", "RONI", "RONNY", "ROSITA", "RUDI",
    "RUDY", "RULI", "RULLY", "RUSLAN", "RUSTAM", "RYAN", "SABAR", "SAEPUDIN", "SAFITRI",
    "SAID", "SAIFUDDIN", "SALEH", "SALIM", "SALMAN", "SALMANA", "SAMUEL", "SANDI", "SANDRA",
    "SANJAYA", "SANTOSO", "SAPUTRA", "SAPUTRI", "SARAH", "SARI", "SARIP", "SARTIKA",
    "SATRIA", "SEPTI", "SEPTIAN", "SEPTIANI", "SEPTIYANTO", "SETIAWAN", "SETIABUDI",
    "SHERLY", "SHINTA", "SILVIA", "SIMON", "SINTA", "SITI", "SOFYAN", "SONI", "SONNY",
    "SRI", "STEFANUS", "STEVEN", "SUBAGYO", "SUCI", "SUCIPTO", "SUGAR", "SUGENG", "SUHARTO",
    "SUJARWO", "SUKMA", "SULAEMAN", "SULASTRI", "SULIS", "SULTAN", "SUMARNI", "SUNARTO",
    "SUPARMAN", "SUPRIADI", "SUPRIYADI", "SUPRIYANTO", "SURYA", "SURYADI", "SURYANA",
    "SUSANTO", "SUSANTI", "SUSILO", "SUTANTO", "SUTRISNO", "SYAFRI", "SYAHPUTRA", "SYAIFUL",
    "SYAMSUL", "SYARIF", "TANIA", "TAUFIC", "TAUFIK", "TEDI", "TEDDY", "TEGUH", "TIA",
    "TIARA", "TINI", "TITA", "TITIN", "TOMI", "TOMMY", "TRI", "TRIA", "TRIYANTO", "TUTI",
    "UDIN", "UTAMI", "UTOM", "UTOMO", "UTU", "VALENTINO", "VERA", "VERONICA", "VICTOR",
    "VINA", "VINCENT", "VIONA", "VITA", "VIVI", "WAHYU", "WAHYUDI", "WALUYO", "WATI",
    "WENI", "WIBOWO", "WIDA", "WIDIA", "WIDODO", "WIDYA", "WIGUNA", "WIJAYA", "WINDA",
    "WINDY", "WINARTO", "WISNU", "WULAN", "YANA", "YANI", "YANTO", "YARI",
    "YASMIN", "YAYAT", "YENI", "YENNY", "YOPI", "YOSEP", "YOSEPH", "YOSUA", "YUDHA",
    "YUDI", "YULIA", "YULIANA", "YULIANTI", "YULIANTO", "YUNITA", "YUNUS", "YUSUF",
    "YUSUP", "ZACKY", "ZAINAL", "ZAINUDDIN", "ZAKI", "ZAKARIA", "ZULFA", "ZULFIKAR"
}

def clean_text(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    import unicodedata
    text = unicodedata.normalize('NFKC', str(raw_text)).upper().strip()
    # Strip leading/trailing colon, dash, quotes, noise symbols
    text = re.sub(r'^[:\-–—"\':;\.>\s]+', '', text)
    text = re.sub(r'[:\-–—"\':;\.>\s]+$', '', text)
    # Ensure space after period (e.g. PERUM.PASIR -> PERUM. PASIR, KP.JATISARI -> KP. JATISARI)
    text = re.sub(r'\.([A-Za-z0-9])', r'. \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None


def clean_rt_rw(raw_val: Optional[str]) -> Optional[str]:
    s = clean_text(raw_val)
    if not s:
        return None
    s = re.sub(r'\s*/\s*', '/', s)
    # Fix OCR misread 6 -> 0 in 3-digit RT/RW (e.g. 002/606 -> 002/006, 601/017 -> 001/017)
    # OCR on low-contrast cards frequently confuses leading '60' for '00'
    s = re.sub(r'\b60(\d)\b', r'00\1', s)
    s = re.sub(r'/60(\d)\b', r'/00\1', s)
    return s if s else None


def tokenize_name(raw_val: Optional[str]) -> Optional[str]:
    """
    Segmentation & spacing engine for Indonesian names (e.g. MRAQILSALMANA -> M RAQIL SALMAN A, DEDENKUSMANI -> DEDEN KUSMANI).
    Uses _V2_NAME_LEXICON and single-letter initial prefix/suffix rules.
    """
    if not raw_val:
        return None
    s = clean_text(raw_val)
    if not s:
        return None

    SINGLE_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    tokens = s.split()
    segmented_tokens: List[str] = []

    for token in tokens:
        if len(token) <= 1:
            segmented_tokens.append(token)
            continue
        if token in _V2_NAME_LEXICON:
            segmented_tokens.append(token)
            continue

        rem = token
        sub_tokens: List[str] = []
        max_steps = 20
        step = 0

        while rem and step < max_steps:
            step += 1

            # Cek Single Letter Suffix di akhir kata jika prefixnya adalah token nama valid (misal 'SALMAN' + 'A' -> 'SALMAN A')
            if len(rem) >= 4 and rem[-1] in SINGLE_LETTERS and rem[:-1] in _V2_NAME_LEXICON:
                sub_tokens.append(rem[:-1])
                sub_tokens.append(rem[-1])
                rem = ""
                break

            # Cek Direct Match paling panjang di _V2_NAME_LEXICON
            direct_match = None
            for name in sorted(_V2_NAME_LEXICON, key=len, reverse=True):
                if len(name) > 1 and rem.startswith(name):
                    direct_match = name
                    break

            # Cek Single Letter Prefix Match (misal 'M' di depan 'RAQIL')
            prefix_match = None
            prefix_sub_name = None
            if len(rem) >= 3 and rem[0] in SINGLE_LETTERS:
                rem_rest = rem[1:]
                for name in sorted(_V2_NAME_LEXICON, key=len, reverse=True):
                    if len(name) > 1 and rem_rest.startswith(name):
                        prefix_match = rem[0]
                        prefix_sub_name = name
                        break

            # Pilih antara Direct Match vs Prefix Match yang menghasilkan kata nama lebih panjang
            if direct_match and prefix_sub_name:
                if len(direct_match) >= len(prefix_sub_name):
                    sub_tokens.append(direct_match)
                    rem = rem[len(direct_match):]
                    continue
                else:
                    sub_tokens.append(prefix_match)
                    rem = rem[1:]
                    continue
            elif direct_match:
                sub_tokens.append(direct_match)
                rem = rem[len(direct_match):]
                continue
            elif prefix_match:
                sub_tokens.append(prefix_match)
                rem = rem[1:]
                continue

            # Single Letter Suffix
            if len(rem) == 1 and rem[0] in SINGLE_LETTERS:
                sub_tokens.append(rem[0])
                rem = ""
                break

            sub_tokens.append(rem)
            break

        segmented_tokens.append(" ".join(sub_tokens))

    res = " ".join(segmented_tokens)
    return re.sub(r'\s+', ' ', res).strip()


_ADDRESS_KEYWORDS: List[str] = sorted([
    "PERUMAHAN", "PERUM", "PASIR", "SEMBUNG", "BLOK", "GANG", "DSN", "DUSUN",
    "KAMPUNG", "KMP", "KP", "JALAN", "JL", "NOMOR", "NO", "RT", "RW", "KOTA", "KABUPATEN",
    "RANCAEKEK", "PERMAI", "BOJONG", "SALAM", "BABAKAN", "PARIGI", "JATISARI", "RAYA"
], key=len, reverse=True)


def tokenize_address(raw_val: Optional[str]) -> Optional[str]:
    """
    Spacing & punctuation engine for address text.
    - Adds space after period '.' (e.g. PERUM.PASIR -> PERUM. PASIR)
    - Separates letter & digit boundaries (e.g. BLOKA -> BLOK A, BLOKA9 -> BLOK A 9)
    - Separates hyphenated block numbers (e.g. BLOKE-4NO 2A -> BLOK E-4 NO 2A)
    - Tokenizes concatenated address keywords (e.g. RANCAEKEKPERMAI -> RANCAEKEK PERMAI)
    """
    if not raw_val:
        return None
    s = clean_text(raw_val)
    if not s:
        return None

    # 1. Separasi keyword alamat utama dari kata sebelumnya yang menempel
    keywords = ["PERMAI", "INDAH", "BLOK", "ASRI", "UTAMA", "JAYA", "MAKMUR", "SEJAHTERA", "GRIYA", "GRYA", "VILLA", "VILA", "PASIR", "SEMBUNG", "RANCAEKEK"]
    for kw in keywords:
        s = re.sub(rf'([A-Za-z]{{3,}})({kw})', r'\1 \2', s)

    # 2. Separasi kata BLOK / NO / KP / JL dengan huruf/angka setelahnya
    s = re.sub(r'\bBLOK([A-Z])', r'BLOK \1', s)
    s = re.sub(r'\bNO([A-Z0-9])', r'NO \1', s)
    s = re.sub(r'\bKP([A-Z])', r'KP \1', s)
    s = re.sub(r'\bJL([A-Z])', r'JL \1', s)

    # 3. Separasi berbatasan dengan angka / NO (misal: -4NO -> -4 NO)
    s = re.sub(r'(\d+)(NO)\b', r'\1 \2', s)
    s = re.sub(r'([A-Za-z]+)(\d+)', r'\1 \2', s)
    s = re.sub(r'(\d+)([A-Za-z]+)', r'\1 \2', s)

    # 4. Tokenisasi kata alamat
    tokens = s.split()
    seg_tokens: List[str] = []
    for t in tokens:
        if '/' in t or '-' in t:
            seg_tokens.append(t)
            continue

        rem = t
        sub_t: List[str] = []
        step = 0
        while rem and step < 20:
            step += 1
            matched = False
            for kw in _ADDRESS_KEYWORDS:
                if len(kw) > 2 and rem.startswith(kw):
                    sub_t.append(kw)
                    rem = rem[len(kw):]
                    matched = True
                    break
            if not matched:
                sub_t.append(rem)
                break
        seg_tokens.append(" ".join(sub_t))

    res = " ".join(seg_tokens)
    # Re-fix formatting seperti BLOK E - 4 -> BLOK E-4, NO 2 A -> NO 2A
    res = re.sub(r'\b([A-Z])\s*-\s*(\d+)\b', r'\1-\2', res)
    res = re.sub(r'\b(\d+)\s+([A-Z])\b', r'\1\2', res)
    res = re.sub(r'\s*\.\s*', '. ', res)
    return re.sub(r'\s+', ' ', res).strip()


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
    menggunakan greedy longest-match terhadap kamus kata pekerjaan.
    Juga menangani merge parsial: BURUH HARIANLEPAS → BURUH HARIAN LEPAS."""
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()

    # Pre-clean common OCR character misreads in Indonesian occupation terms
    s = s.replace("HARLAN", "HARIAN").replace("PELAIAR", "PELAJAR").replace("SWA5TA", "SWASTA")
    s = s.replace("LEPA5", "LEPAS").replace("BURUHHARLAN", "BURUH HARIAN")

    def _greedy_match(tok: str) -> str:
        """Greedy longest-match tokenizer untuk satu token tunggal tanpa spasi."""
        if '/' in tok or len(tok) <= 4:
            return tok
        result: List[str] = []
        remaining = tok
        max_iter = 30
        iteration = 0
        while remaining and iteration < max_iter:
            iteration += 1
            matched = False
            for vocab_word in _PEKERJAAN_VOCAB:
                vw_no_space = vocab_word.replace(' ', '').replace('/', '')
                if remaining.startswith(vw_no_space):
                    result.append(vocab_word)
                    remaining = remaining[len(vw_no_space):]
                    matched = True
                    break
            if not matched:
                result.append(remaining)
                break
        return ' '.join(result).strip() if result else tok

    if '/' in s:
        return s

    if ' ' in s:
        # Proses per-token: tangkap kasus BURUH HARIANLEPAS → BURUH HARIAN LEPAS
        parts = s.split()
        result_parts = [_greedy_match(p) for p in parts]
        result = ' '.join(result_parts)
        return re.sub(r'\s+', ' ', result).strip()

    # Tidak ada spasi — greedy match seluruh string
    result = _greedy_match(s)
    result = re.sub(r'\s+', ' ', result)
    return result if result else s


# ─── Regional Name Normalizer ────────────────────────────────────────────────────
def normalize_regional(raw_val: Optional[str]) -> Optional[str]:
    """Normalisasi nama wilayah Indonesia:
    - Q → C (OCR sering salah baca C sebagai Q): RANQAEKEK → RANCAEKEK
    - X → K (OCR sering salah baca K sebagai X pada gambar kualitas rendah): XELAMIN → KELAMIN
    - Truncated city names: BANDUN -> BANDUNG
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    s = s.replace('Q', 'C')
    s = s.replace('X', 'K')  # Nama wilayah Indonesia tidak menggunakan huruf X asli
    if re.search(r'\bBANDUN$', s):
        s = s + 'G'
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None


# ─── Compound Regional Name Tokenizer ────────────────────────────────────────────
_REGIONAL_PREFIXES: List[str] = sorted([
    "RANCAEKEK", "BOJONG", "BANJAR", "CIPARAY", "CIBIRU", "CICALENGKA", "CIMAHI", "CIREBON",
    "SUKA", "KARANG", "PADA", "GIRI", "MEKAR", "SUMBER", "BUMI", "PONDOK", "TAMAN",
    "GRIYA", "GRYA", "VILLA", "VILA", "PASIR", "KAMPUNG", "DUSUN", "BABAKAN", "BATU",
    "GUNUNG", "RUMAH", "KOTA", "KABUPATEN"
], key=len, reverse=True)

_REGIONAL_SUFFIXES: List[str] = sorted([
    "SALAM", "PERMAI", "INDAH", "BARU", "LAMA", "JAYA", "MAKMUR", "SEJAHTERA",
    "WETAN", "KULON", "KIDUL", "LOR", "BARAT", "TIMUR", "UTARA", "SELATAN", "TENGAH",
    "ASIH", "MULYA", "MEKAR", "SARI", "WANGI", "MANIK", "LOA", "GEDE", "KERTA",
    "PARIGI", "JATISARI", "MONYET", "BLOK", "NO", "RT", "RW", "KP", "JL"
], key=len, reverse=True)


def tokenize_compound_name(raw_val: Optional[str]) -> Optional[str]:
    """Memisahkan nama wilayah/alamat yang digabung OCR dalam 1 bounding box.
    Contoh: BOJONGSALAM → BOJONG SALAM, RANCAEKEKWETAN → RANCAEKEK WETAN
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()

    if ' ' in s:
        return s

    # Coba match prefix dari depan
    for prefix in _REGIONAL_PREFIXES:
        if s.startswith(prefix) and len(s) > len(prefix):
            suffix = s[len(prefix):].strip()
            if suffix:
                return f"{prefix} {suffix}"

    # Coba match suffix dari belakang
    for suffix in _REGIONAL_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            base = s[: -len(suffix)].strip()
            if base:
                return f"{base} {suffix}"

    return s
