import json
import html
from typing import Dict, Any, Tuple, Optional
from app.ktp.v1.schemas_v1 import MobileDataInputV1

ALIAS_MAP: Dict[str, str] = {
    # NIK
    "nik": "nik",
    "idnumber": "nik",
    "id_number": "nik",
    "ktp_number": "nik",
    "no_ktp": "nik",
    "id": "nik",
    "identitynumber": "nik",

    # NAMA
    "nama": "nama",
    "name": "nama",
    "fullname": "nama",
    "full_name": "nama",
    "nama_lengkap": "nama",

    # TEMPAT LAHIR
    "tempat_lahir": "tempat_lahir",
    "tempatlahir": "tempat_lahir",
    "birthplace": "tempat_lahir",
    "birth_place": "tempat_lahir",
    "pob": "tempat_lahir",
    "tmpt_lahir": "tempat_lahir",

    # TANGGAL LAHIR
    "tanggal_lahir": "tanggal_lahir",
    "tanggallahir": "tanggal_lahir",
    "birthdate": "tanggal_lahir",
    "birth_date": "tanggal_lahir",
    "dob": "tanggal_lahir",
    "tgl_lahir": "tanggal_lahir",

    # JENIS KELAMIN
    "jenis_kelamin": "jenis_kelamin",
    "jeniskelamin": "jenis_kelamin",
    "gender": "jenis_kelamin",
    "sex": "jenis_kelamin",
    "kelamin": "jenis_kelamin",

    # GOLONGAN DARAH
    "golongan_darah": "golongan_darah",
    "golongandarah": "golongan_darah",
    "bloodtype": "golongan_darah",
    "blood_type": "golongan_darah",
    "gol_darah": "golongan_darah",
    "goldar": "golongan_darah",

    # ALAMAT
    "alamat": "alamat",
    "address": "alamat",
    "fulladdress": "alamat",
    "full_address": "alamat",

    # RT_RW
    "rt_rw": "rt_rw",
    "rtrw": "rt_rw",
    "rt_rw_number": "rt_rw",
    "rt_dan_rw": "rt_rw",

    # KELURAHAN DESA
    "kelurahan_desa": "kelurahan_desa",
    "kel_desa": "kelurahan_desa",
    "keldesa": "kelurahan_desa",
    "kelurahan": "kelurahan_desa",
    "desa": "kelurahan_desa",
    "village": "kelurahan_desa",
    "subdistrict": "kelurahan_desa",
    "sub_district": "kelurahan_desa",

    # KECAMATAN
    "kecamatan": "kecamatan",
    "district": "kecamatan",
    "distrik": "kecamatan",
    "kec": "kecamatan",

    # AGAMA
    "agama": "agama",
    "religion": "agama",

    # STATUS PERKAWINAN
    "status_perkawinan": "status_perkawinan",
    "statusperkawinan": "status_perkawinan",
    "maritalstatus": "status_perkawinan",
    "marital_status": "status_perkawinan",
    "status_nikah": "status_perkawinan",
    "status": "status_perkawinan",

    # PEKERJAAN
    "pekerjaan": "pekerjaan",
    "occupation": "pekerjaan",
    "job": "pekerjaan",
    "work": "pekerjaan",

    # KEWARGANEGARAAN
    "kewarganegaraan": "kewarganegaraan",
    "nationality": "kewarganegaraan",
    "warga_negara": "kewarganegaraan",
    "kewarganegaraan_id": "kewarganegaraan",

    # BERLAKU HINGGA
    "berlaku_hingga": "berlaku_hingga",
    "berlakuhingga": "berlaku_hingga",
    "expirydate": "berlaku_hingga",
    "expiry_date": "berlaku_hingga",
    "valid_until": "berlaku_hingga",
}

CANONICAL_KEYS = [
    "nik", "nama", "tempat_lahir", "tanggal_lahir", "jenis_kelamin",
    "golongan_darah", "alamat", "rt_rw", "kelurahan_desa", "kecamatan",
    "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"
]


class GlobalPayloadNormalizer:
    """
    Normalizer terpusat untuk memetakan seluruh bentuk payload JSON dari Mobile/Backend
    menjadi 15 field baku KTP Indonesia.
    """

    @staticmethod
    def _unwrap_payload(raw_input: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Unwrap payload jika dibungkus key 'data', 'payload', 'result', atau 'extracted_data'.
        Mengembalikan tuple (fields_dict, top_level_metadata).
        """
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except Exception:
                return {}, {}

        if not isinstance(raw_input, dict):
            return {}, {}

        current = raw_input
        metadata = {}

        for meta_key in ["quality", "warnings", "success", "metrics"]:
            if meta_key in current:
                metadata[meta_key] = current[meta_key]

        for _ in range(5):
            found_unwrap = False
            for wrapper in ["data", "payload", "result", "extracted_data"]:
                if wrapper in current and isinstance(current[wrapper], dict):
                    current = current[wrapper]
                    found_unwrap = True
                    break
            if not found_unwrap:
                break

        return current, metadata

    @staticmethod
    def _sanitize_string(val: Any) -> Optional[str]:
        """Sanitasi nilai string: strip, unescape, dan normalisasi None/empty string."""
        if val is None:
            return None
        text = str(val).strip()
        if not text or text.lower() in ["none", "null", ""]:
            return None
        text = html.unescape(text)
        return text.strip() or None

    @classmethod
    def normalize(cls, raw_input: Any) -> Dict[str, Any]:
        fields_dict, _ = cls._unwrap_payload(raw_input)
        result = {key: None for key in CANONICAL_KEYS}

        for key, val in fields_dict.items():
            norm_key = key.lower().replace("-", "_").strip()
            canonical_key = ALIAS_MAP.get(norm_key)

            if not canonical_key:
                continue

            extracted_val = None
            if isinstance(val, dict):
                raw_val = val.get("value")
                if raw_val is None:
                    raw_val = val.get("val") or val.get("text") or val.get("content")
                extracted_val = cls._sanitize_string(raw_val)
            else:
                extracted_val = cls._sanitize_string(val)

            if extracted_val is not None or result[canonical_key] is None:
                result[canonical_key] = extracted_val

        return result

    @classmethod
    def to_mobile_input_v1(cls, raw_input: Any) -> MobileDataInputV1:
        """
        Mengonversi payload mentah langsung ke Pydantic model MobileDataInputV1
        yang diharapkan oleh endpoint v1 /validate.
        """
        fields_dict, _ = cls._unwrap_payload(raw_input)
        normalized_flat = cls.normalize(raw_input)

        v1_dict = {}
        for key in CANONICAL_KEYS:
            flat_val = normalized_flat.get(key)
            
            orig_conf = 0.0
            orig_src = "mobile"
            
            for raw_k, raw_v in fields_dict.items():
                if ALIAS_MAP.get(raw_k.lower().replace("-", "_").strip()) == key and isinstance(raw_v, dict):
                    if "confidence" in raw_v:
                        try:
                            orig_conf = float(raw_v["confidence"])
                        except (ValueError, TypeError):
                            pass
                    if "source" in raw_v:
                        orig_src = str(raw_v["source"])
                    break

            if flat_val is not None:
                v1_dict[key] = {
                    "value": flat_val,
                    "confidence": orig_conf,
                    "source": orig_src
                }
            else:
                v1_dict[key] = None

        return MobileDataInputV1(**v1_dict)
