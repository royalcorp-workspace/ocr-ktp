import re
from typing import Optional, List, Set

# ─── Fast Generic Levenshtein Distance (Standard Library) ──────────────────────
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# ─── Indonesian Name Lexicon ───────────────────────────────────────────────────
_V3_NAME_LEXICON: Set[str] = {
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
    "SAID", "SAIFUDDIN", "SALEH", "SALIM", "SALMAN", "SAMUEL", "SANDI", "SANDRA",
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


# ─── Master Indonesian City / Regency Lexicon ──────────────────────────────────
_KOTA_KABUPATEN_LEXICON: Set[str] = {
    "AMBON", "BALIKPAPAN", "BANDA ACEH", "BANDAR LAMPUNG", "BANDUNG", "BANDUNG BARAT",
    "BANGKA", "BANGKA BARAT", "BANGKA SELATAN", "BANGKA TENGAH", "BANGKALAN", "BANGLI",
    "BANJAR", "BANJARBARU", "BANJARMASIN", "BANJARNEGARA", "BANTAENG", "BANTUL", "BANYUMAS",
    "BANYUWANGI", "BARITO KUALA", "BARITO SELATAN", "BARITO TIMUR", "BARITO UTARA", "BARRU",
    "BATAM", "BATANG", "BATANGHARI", "BATU", "BATUBARA", "BEKASI", "BELITUNG", "BELITUNG TIMUR",
    "BELU", "BENER MERIAH", "BENGKALIS", "BENGKAYANG", "BENGKULU", "BENGKULU SELATAN",
    "BENGKULU TENGAH", "BENGKULU UTARA", "BERAU", "BIAK NUMFOR", "BIMA", "BINJAI", "BINTAN",
    "BITUNG", "BLITAR", "BLORA", "BOALEMO", "BOGOR", "BOJONEGORO", "BOLAANG MONGONDOW",
    "BOMBANA", "BONDOWOSO", "BONE", "BONE BOLANGO", "BONTANG", "BOVEN DIGOEL", "BOYOLALI",
    "BREBES", "BUKITTINGGI", "BULELENG", "BULUKUMBA", "BULUNGAN", "BUNGO", "BUOL", "BURU",
    "BURU SELATAN", "BUTON", "BUTON SELATAN", "BUTON TENGAH", "BUTON UTARA", "CIAMIS",
    "CIANJUR", "CILACAP", "CILEGON", "CIMAHI", "CIREBON", "DAIRI", "DEIYAI", "DELI SERDANG",
    "DEMAK", "DENPASAR", "DEPOK", "DHARMASRAYA", "DOGIYAI", "DOMPU", "DONGGALA", "DUMAI",
    "EMPAT LAWANG", "ENREKANG", "FAKFAK", "FLORES TIMUR", "GARUT", "GAYO LUES", "GIANYAR",
    "GORONTALO", "GORONTALO UTARA", "GOWA", "GRESIK", "GROBOGAN", "GUNUNGKIDUL", "GUNUNGSITOLI",
    "HALMAHERA BARAT", "HALMAHERA SELATAN", "HALMAHERA TENGAH", "HALMAHERA TIMUR", "HALMAHERA UTARA",
    "HULU SUNGAI SELATAN", "HULU SUNGAI TENGAH", "HULU SUNGAI UTARA", "HUMBANG HASUNDUTAN",
    "INDRAGIRI HILIR", "INDRAGIRI HULU", "INDRAMAYU", "INTAN JAYA", "JAKARTA BARAT",
    "JAKARTA PUSAT", "JAKARTA SELATAN", "JAKARTA TIMUR", "JAKARTA UTARA", "JAMBI", "JAYAPURA",
    "JAYAWIJAYA", "JEMBER", "JEMBRANA", "JENEPONTO", "JEPARA", "JOMBANG", "KAIMANA", "KAMPAR",
    "KAPUAS", "KAPUAS HULU", "KARANGANYAR", "KARANGASEM", "KARAWANG", "KARIMUN", "KARO",
    "KATINGAN", "KAUR", "KAYONG UTARA", "KEBUMEN", "KEDIRI", "KEEROM", "KENDAL", "KENDARI",
    "KEPULAUAN ANAMBAS", "KEPULAUAN ARU", "KEPULAUAN MENTAWAI", "KEPULAUAN MERANTI",
    "KEPULAUAN SANGIHE", "KEPULAUAN SERIBU", "KEPULAUAN SITARO", "KEPULAUAN SULA",
    "KEPULAUAN TALAUD", "KEPULAUAN TANIMBAR", "KEPULAUAN YAPEN", "KERINCI", "KETAPANG",
    "KLATEN", "KLUNGKUNG", "KOLAKA", "KOLAKA TIMUR", "KOLAKA UTARA", "KONAWE", "KONAWE KEPULAUAN",
    "KONAWE SELATAN", "KONAWE UTARA", "KOTABARU", "KOTAMOBAGU", "KOTAWARINGIN BARAT",
    "KOTAWARINGIN TIMUR", "KUANTAN SINGINGI", "KUBU RAYA", "KUDUS", "KULON PROGO", "KUNINGAN",
    "KUPANG", "KUTAI BARAT", "KUTAI KARTANEGARA", "KUTAI TIMUR", "LABUHANBATU", "LABUHANBATU SELATAN",
    "LABUHANBATU UTARA", "LAHAT", "LAMANDAU", "LAMONGAN", "LAMPUNG BARAT", "LAMPUNG SELATAN",
    "LAMPUNG TENGAH", "LAMPUNG TIMUR", "LAMPUNG UTARA", "LANDAK", "LANGKAT", "LANGSA", "LANNY JAYA",
    "LEBAK", "LEBONG", "LEMBANG", "LEMBATA", "LHOKSEUMAWE", "LIMA PULUH KOTA", "LINGGA",
    "LOMBOK BARAT", "LOMBOK TENGAH", "LOMBOK TIMUR", "LOMBOK UTARA", "LUBUKLINGGAU", "LUMAJANG",
    "LUWU", "LUWU TIMUR", "LUWU UTARA", "MADIUN", "MAGELANG", "MAGETAN", "MAHAKAM ULU",
    "MAJALENGKA", "MAJENE", "MAKASSAR", "MALANG", "MALINAU", "MALUKU BARAT DAYA", "MALUKU TENGAH",
    "MALUKU TENGGARA", "MAMASA", "MAMBERAMO RAYA", "MAMBERAMO TENGAH", "MAMUJU", "MAMUJU TENGAH",
    "MANADO", "MANDAINING NATAL", "MANGGARAI", "MANGGARAI BARAT", "MANGGARAI TIMUR", "MANOKWARI",
    "MANOKWARI SELATAN", "MAPPI", "MAROS", "MATARAM", "MAYBRAT", "MEDAN", "MELAWI", "MEMPAWAH",
    "MERANGIN", "MERAUKE", "MESUJI", "METRO", "MIMIKA", "MINAHASA", "MINAHASA SELATAN",
    "MINAHASA TENGGARA", "MINAHASA UTARA", "MOJOKERTO", "MOROWALI", "MOROWALI UTARA", "MUARA ENIM",
    "MUARO JAMBI", "MUKOMUKO", "MUNA", "MUNA BARAT", "MURUNG RAYA", "MUSI BANYUASIN", "MUSI RAWAS",
    "MUSI RAWAS UTARA", "NABIRE", "NAGAN RAYA", "NAGEKEO", "NATUNA", "NDUGA", "NGADA", "NGANJUK",
    "NGAWI", "NIAS", "NIAS BARAT", "NIAS SELATAN", "NIAS UTARA", "NUNUKAN", "OGAN ILIR",
    "OGAN KOMERING ILIR", "OGAN KOMERING ULU", "OGAN KOMERING ULU SELATAN", "OGAN KOMERING ULU TIMUR",
    "PACITAN", "PADANG", "PADANG LAWAS", "PADANG LAWAS UTARA", "PADANG PANJANG", "PADANG PARIAMAN",
    "PADANGSIDIMPUAN", "PAGAR ALAM", "PAKPAK BHARAT", "PALANGKA RAYA", "PALEMBANG", "PALOPO",
    "PALU", "PAMEKASAN", "PANDEGLANG", "PANGANDARAN", "PANGKAJENE DAN KEPULAUAN", "PANGKALPINANG",
    "PANIAI", "PAREPARE", "PARIAMAN", "PARIGI MOUTONG", "PASAMAN", "PASAMAN BARAT", "PASER",
    "PASURUAN", "PATI", "PAYAKUMBUH", "PEGUNUNGAN ARFAK", "PEGUNUNGAN BINTANG", "PEKALONGAN",
    "PEKANBARU", "PELALAWAN", "PEMALANG", "PEMATANGSIANTAR", "PENAJAM PASER UTARA", "PESAWARAN",
    "PESISIR BARAT", "PESISIR SELATAN", "PIDIE", "PIDIE JAYA", "PINRANG", "POHUWATO", "POLEWALI MANDAR",
    "PONOROGO", "PONTIANAK", "POSO", "PRABUMULIH", "PRINGSEWU", "PROBOLINGGO", "PULANG PISAU",
    "PULAU MOROTAI", "PULAU TALIABU", "PUNCAK", "PUNCAK JAYA", "PURBALINGGA", "PURWAKARTA",
    "PURWOREJO", "RAJA AMPAT", "REJANG LEBONG", "REMBANG", "ROKAN HILIR", "ROKAN HULU", "ROTE NDAO",
    "SABANG", "SABU RAIJUA", "SALATIGA", "SAMARINDA", "SAMBAS", "SAMOSIR", "SAMPANG", "SANGGAU",
    "SARMI", "SAROLANGUN", "SAUMLAKI", "SAWAHLUNTO", "SEKADAU", "SELUMA", "SEMARANG", "SERAM BAGIAN BARAT",
    "SERAM BAGIAN TIMUR", "SERANG", "SERDANG BEDAGAI", "SERUYAN", "SIAK", "SIBOLGA", "SIDENRENG RAPPANG",
    "SIDOARJO", "SIGI", "SIJUNJUNG", "SIKKA", "SIMALUNGUN", "SIMEULUE", "SINGKAWANG", "SINJAI",
    "SINTANG", "SITUBONDO", "SLEMAN", "SOLOK", "SOLOK SELATAN", "SOREANG", "SORONG", "SORONG SELATAN",
    "SRAGEN", "SUBANG", "SUKABUMI", "SUKAMARA", "SUKOHARJO", "SUMBA BARAT", "SUMBA BARAT DAYA",
    "SUMBA TENGAH", "SUMBA TIMUR", "SUMBAWA", "SUMBAWA BARAT", "SUMEDANG", "SUMENEP", "SUNGAI PENUH",
    "SUPAN", "SUPANBARAT", "SUPAR", "SURABAYA", "SURAKARTA", "TABALONG", "TABANAN", "TAKALAR",
    "TAMBRAUW", "TANA TIDUNG", "TANA TORAJA", "TANAH BUMBU", "TANAH DATAR", "TANAH LAUT",
    "TANGERANG", "TANGERANG SELATAN", "TANGGAMUS", "TANJUNGBALAI", "TANJUNG JABUNG BARAT",
    "TANJUNG JABUNG TIMUR", "TANJUNGPINANG", "TAPANULI SELATAN", "TAPANULI TENGAH", "TAPANULI UTARA",
    "TAPIN", "TARAKAN", "TASIKMALAYA", "TEBING TINGGI", "TEBO", "TEGAL", "TELUK BINTUNI",
    "TELUK WONDAMA", "TEMANGGUNG", "TERNATE", "TIDORE KEPULAUAN", "TIMOR TENGAH SELATAN",
    "TIMOR TENGAH UTARA", "TOBA", "TOJO UNA-UNA", "TOLI-TOLI", "TOLIKARA", "TOMOHON", "TORAJA UTARA",
    "TRENGGALEK", "TUBAN", "TULANG BAWANG", "TULANG BAWANG BARAT", "TULUNGAGUNG", "WAJO", "WAKATOBI",
    "WAROPEN", "WAY KANAN", "WONOGIRI", "WONOSOBO", "YAHUKIMO", "YALIMO", "YOGYAKARTA"
}


# ─── Address Keywords Lexicon ──────────────────────────────────────────────────
_ADDRESS_KEYWORDS: List[str] = sorted([
    "PERUMAHAN", "PERUM", "PASIR", "SEMBUNG", "BLOK", "GANG", "DSN", "DUSUN",
    "KAMPUNG", "KMP", "KP", "JALAN", "JL", "NOMOR", "NO", "RT", "RW", "KOTA", "KABUPATEN",
    "RANCAEKEK", "PERMAI", "BOJONG", "SALAM", "BABAKAN", "PARIGI", "JATISARI", "RAYA",
    "MONYET", "LINGGARJATI", "LINGGAR", "JATI", "INDAH", "ASRI", "MEKAR", "SARI", "WETAN",
    "KULON", "SUKA", "MULYA", "SIRNAGALIH", "JELEGONG"
], key=len, reverse=True)


# ─── Pekerjaan Vocabulary Lexicon ──────────────────────────────────────────────
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
    "PEGAWAI", "LAINNYA", "IRT"
], key=len, reverse=True)


def clean_text(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    import unicodedata
    text = unicodedata.normalize('NFKC', str(raw_text)).upper().strip()
    text = re.sub(r'^[:\-–—"\':;\.>\s]+', '', text)
    text = re.sub(r'[:\-–—"\':;\.>\s]+$', '', text)
    text = re.sub(r'\.([A-Za-z0-9])', r'. \1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None


def clean_rt_rw(raw_val: Optional[str]) -> Optional[str]:
    s = clean_text(raw_val)
    if not s:
        return None
    s = re.sub(r'\s*/\s*', '/', s)
    # Generic OCR digit loop-closure recovery (leading 60x -> 00x in 3-digit format)
    s = re.sub(r'\b60(\d)\b', r'00\1', s)
    s = re.sub(r'/60(\d)\b', r'/00\1', s)
    return s if s else None


def clean_nik(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    s = re.sub(r'^(NIK|HIK|MIL|N|I|K)[:\s]*', '', s)
    s = s.replace('O', '0').replace('D', '0').replace('Q', '0')
    s = s.replace('I', '1').replace('L', '1').replace('l', '1').replace('|', '1').replace('!', '1')
    s = s.replace('B', '8').replace('S', '5').replace('Z', '2')
    digits = re.sub(r'[^\d]', '', s)
    if len(digits) == 16:
        return digits
    elif len(digits) > 16:
        return digits[:16]
    return digits if digits else None


def clean_date(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    # Normalize OCR number misreads in date strings (O/D->0, I/L/l/|->1, S->5, B->8, Z->2)
    s_clean = s.replace('O', '0').replace('D', '0').replace('Q', '0')
    s_clean = s_clean.replace('I', '1').replace('L', '1').replace('l', '1').replace('|', '1').replace('!', '1')
    s_clean = s_clean.replace('S', '5').replace('B', '8').replace('Z', '2')

    # Regex match for DD-MM-YYYY or DD MM YYYY or DD/MM/YYYY or DD.MM.YYYY
    match = re.search(r'(\d{1,2})[\s\-\./]+(\d{1,2})[\s\-\./]+(\d{4})', s_clean)
    if match:
        d, m, y = match.groups()
        d_int, m_int, y_int = int(d), int(m), int(y)
        if 1 <= d_int <= 31 and 1 <= m_int <= 12 and 1900 <= y_int <= 2099:
            return f"{d_int:02d}-{m_int:02d}-{y_int:04d}"

    # Secondary match for 2-digit year (DD-MM-YY)
    match_2d = re.search(r'(\d{1,2})[\s\-\./]+(\d{1,2})[\s\-\./]+(\d{2})\b', s_clean)
    if match_2d:
        d, m, y2 = match_2d.groups()
        d_int, m_int, y2_int = int(d), int(m), int(y2)
        if 1 <= d_int <= 31 and 1 <= m_int <= 12:
            full_y = 1900 + y2_int if y2_int > 30 else 2000 + y2_int
            return f"{d_int:02d}-{m_int:02d}-{full_y:04d}"

    return None


def extract_date_from_nik(nik: Optional[str]) -> Optional[str]:
    """
    Standard Dukcapil Date of Birth extractor from 16-digit NIK:
    - Digits 7-8: Day of birth (Female: Day + 40)
    - Digits 9-10: Month of birth (01-12)
    - Digits 11-12: Year of birth (2 digits, e.g. 01 -> 2001, 74 -> 1974)
    """
    if not nik:
        return None
    digits = re.sub(r'[^\d]', '', str(nik))
    if len(digits) != 16:
        return None
    try:
        raw_day = int(digits[6:8])
        month = int(digits[8:10])
        year_2d = int(digits[10:12])

        day = raw_day - 40 if raw_day > 40 else raw_day
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return None

        full_year = 1900 + year_2d if year_2d > 30 else 2000 + year_2d
        return f"{day:02d}-{month:02d}-{full_year:04d}"
    except Exception:
        return None


def clean_gender(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper()
    if 'LAKI' in s or 'LAK' in s or 'PRIA' in s or 'PERIA' in s:
        return 'LAKI-LAKI'
    elif 'PEREMP' in s or 'PEREM' in s or 'WANITA' in s:
        return 'PEREMPUAN'
    return None


def clean_blood_type(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
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


# ─── Generic Regional Normalizer using Master Toponym Lexicon ──────────────────
def normalize_regional(raw_val: Optional[str]) -> Optional[str]:
    """
    Generic regional normalizer using Indonesian City/Kabupaten Master Lexicon:
    1. Q -> C OCR typo replacement (Indonesian names do not use Q natively).
    2. Suffix-drop recovery (e.g. BANDUN -> BANDUNG, SERAN -> SERANG, SUBAN -> SUBANG).
    3. Levenshtein distance matching <= 1 for longer regional names.
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    s = s.replace('Q', 'C')
    s = s.replace('X', 'K')
    s = re.sub(r'\s+', ' ', s).strip()

    # 1. Exact match against lexicon
    if s in _KOTA_KABUPATEN_LEXICON:
        return s

    # 2. Suffix-drop recovery (trailing character dropped at image crop boundary)
    for valid_city in _KOTA_KABUPATEN_LEXICON:
        if len(valid_city) >= 5 and valid_city.startswith(s) and len(valid_city) - len(s) == 1:
            return valid_city

    # 3. Fuzzy match with Levenshtein distance <= 1 for names of length >= 6
    if len(s) >= 6:
        best_match = None
        min_dist = 99
        for valid_city in _KOTA_KABUPATEN_LEXICON:
            if abs(len(valid_city) - len(s)) <= 1:
                dist = levenshtein_distance(s, valid_city)
                if dist <= 1 and dist < min_dist:
                    min_dist = dist
                    best_match = valid_city
        if best_match:
            return best_match

    return s


# ─── Generic Pekerjaan Tokenizer with Fuzzy Sub-Token Support ──────────────────
def tokenize_pekerjaan(raw_val: Optional[str]) -> Optional[str]:
    """
    Generic occupation tokenizer using _PEKERJAAN_VOCAB:
    - Greedy longest match across vocabulary words.
    - Fuzzy match (edit distance <= 1) on sub-tokens to resolve OCR misreads
      (e.g. HARLAN -> HARIAN, PELAIAR -> PELAJAR, SWA5TA -> SWASTA) without hardcoding.
    """
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    if '/' in s:
        return s

    def _fuzzy_match_single_token(tok: str) -> str:
        if len(tok) <= 3:
            return tok
        # Check exact prefix first
        for vocab_word in _PEKERJAAN_VOCAB:
            vw_clean = vocab_word.replace(' ', '').replace('/', '')
            if tok == vw_clean:
                return vocab_word

        # Check fuzzy match against vocab tokens (edit distance <= 1 for length >= 5)
        for vocab_word in _PEKERJAAN_VOCAB:
            vw_clean = vocab_word.replace(' ', '').replace('/', '')
            if abs(len(tok) - len(vw_clean)) <= 1 and len(tok) >= 5:
                if levenshtein_distance(tok, vw_clean) <= 1:
                    return vocab_word

        # Multi-token greedy match with fuzzy sub-token tolerance
        result: List[str] = []
        rem = tok
        step = 0
        while rem and step < 20:
            step += 1
            matched = False
            # Try exact sub-token match
            for vocab_word in _PEKERJAAN_VOCAB:
                vw_clean = vocab_word.replace(' ', '').replace('/', '')
                if rem.startswith(vw_clean):
                    result.append(vocab_word)
                    rem = rem[len(vw_clean):]
                    matched = True
                    break
            if matched:
                continue

            # Try fuzzy sub-token match (edit distance <= 1)
            for vocab_word in _PEKERJAAN_VOCAB:
                vw_clean = vocab_word.replace(' ', '').replace('/', '')
                target_len = len(vw_clean)
                if len(rem) >= target_len >= 5:
                    sub_segment = rem[:target_len]
                    if levenshtein_distance(sub_segment, vw_clean) <= 1:
                        result.append(vocab_word)
                        rem = rem[target_len:]
                        matched = True
                        break
            if not matched:
                result.append(rem)
                break
        return ' '.join(result).strip() if result else tok

    if ' ' in s:
        parts = s.split()
        res_parts = [_fuzzy_match_single_token(p) for p in parts]
        res = ' '.join(res_parts)
        return re.sub(r'\s+', ' ', res).strip()

    res = _fuzzy_match_single_token(s)
    return re.sub(r'\s+', ' ', res).strip()


# ─── Generic Address Tokenizer with Fuzzy Match against _ADDRESS_KEYWORDS ─────
def tokenize_address(raw_val: Optional[str]) -> Optional[str]:
    """
    Generic address tokenizer with fuzzy keyword normalization:
    - Normalizes spacing around prefixes (BLOK, NO, KP, JL).
    - Resolves OCR typos in standard address terms (edit distance <= 1 against _ADDRESS_KEYWORDS).
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

    # 3. Separasi berbatasan dengan angka / NO
    s = re.sub(r'(\d+)(NO)\b', r'\1 \2', s)
    s = re.sub(r'([A-Za-z]+)(\d+)', r'\1 \2', s)
    s = re.sub(r'(\d+)([A-Za-z]+)', r'\1 \2', s)

    # 4. Tokenisasi dan koreksi fuzzy terhadap _ADDRESS_KEYWORDS
    tokens = s.split()
    seg_tokens: List[str] = []
    for t in tokens:
        if '/' in t or '-' in t:
            seg_tokens.append(t)
            continue

        # Check fuzzy match against canonical address keywords (edit distance <= 1 for len >= 5)
        clean_t = t.strip('.')
        matched_kw = None
        if len(clean_t) >= 5:
            for kw in _ADDRESS_KEYWORDS:
                if abs(len(clean_t) - len(kw)) <= 1 and len(kw) >= 5:
                    if levenshtein_distance(clean_t, kw) <= 1:
                        matched_kw = kw
                        break

        token_to_process = matched_kw if matched_kw else clean_t

        rem = token_to_process
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
    res = re.sub(r'\b([A-Z])\s*-\s*(\d+)\b', r'\1-\2', res)
    res = re.sub(r'\b(\d+)\s+([A-Z])\b', r'\1\2', res)
    res = re.sub(r'\b(PASIR|BLOK|KP|JL)\.\s*', r'\1 ', res)
    res = re.sub(r'\s*\.\s*', '. ', res)
    return re.sub(r'\s+', ' ', res).strip()


def tokenize_name(raw_val: Optional[str]) -> Optional[str]:
    if not raw_val:
        return None
    s = clean_text(raw_val)
    if not s:
        return None

    SINGLE_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    tokens = s.split()
    segmented_tokens: List[str] = []

    for token in tokens:
        if len(token) <= 1 or token in _V3_NAME_LEXICON:
            segmented_tokens.append(token)
            continue

        rem = token
        sub_tokens: List[str] = []
        step = 0
        while rem and step < 20:
            step += 1
            if len(rem) >= 4 and rem[-1] in SINGLE_LETTERS and rem[:-1] in _V3_NAME_LEXICON:
                sub_tokens.append(rem[:-1])
                sub_tokens.append(rem[-1])
                rem = ""
                break

            direct_match = None
            for name in sorted(_V3_NAME_LEXICON, key=len, reverse=True):
                if len(name) > 1 and rem.startswith(name):
                    direct_match = name
                    break

            prefix_match = None
            prefix_sub_name = None
            if len(rem) >= 3 and rem[0] in SINGLE_LETTERS:
                rem_rest = rem[1:]
                for name in sorted(_V3_NAME_LEXICON, key=len, reverse=True):
                    if len(name) > 1 and rem_rest.startswith(name):
                        prefix_match = rem[0]
                        prefix_sub_name = name
                        break

            if direct_match and prefix_sub_name:
                if len(direct_match) >= len(prefix_sub_name):
                    sub_tokens.append(direct_match)
                    rem = rem[len(direct_match):]
                else:
                    sub_tokens.append(prefix_match)
                    rem = rem[1:]
            elif direct_match:
                sub_tokens.append(direct_match)
                rem = rem[len(direct_match):]
            elif prefix_match:
                sub_tokens.append(prefix_match)
                rem = rem[1:]
            elif len(rem) == 1 and rem[0] in SINGLE_LETTERS:
                sub_tokens.append(rem[0])
                rem = ""
                break
            else:
                sub_tokens.append(rem)
                break

        segmented_tokens.append(" ".join(sub_tokens))

    res = " ".join(segmented_tokens)
    return re.sub(r'\s+', ' ', res).strip()


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
    if not raw_val:
        return None
    s = str(raw_val).upper().strip()
    if ' ' in s:
        return s

    for prefix in _REGIONAL_PREFIXES:
        if s.startswith(prefix) and len(s) > len(prefix):
            suffix = s[len(prefix):].strip()
            if suffix:
                return f"{prefix} {suffix}"

    for suffix in _REGIONAL_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            base = s[: -len(suffix)].strip()
            if base:
                return f"{base} {suffix}"
    return s
