from typing import Optional
from pydantic import BaseModel


class KTPData(BaseModel):
    nik: Optional[str] = None
    nama: Optional[str] = None
    tempat_lahir: Optional[str] = None
    tanggal_lahir: Optional[str] = None
    jenis_kelamin: Optional[str] = None
    golongan_darah: Optional[str] = None
    alamat: Optional[str] = None
    rt_rw: Optional[str] = None
    kelurahan_desa: Optional[str] = None
    kecamatan: Optional[str] = None
    agama: Optional[str] = None
    status_perkawinan: Optional[str] = None
    pekerjaan: Optional[str] = None
class FieldWithConfidence(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0


class KTPOcrResponse(BaseModel):
    nik: FieldWithConfidence
    nama: FieldWithConfidence
    tempat_lahir: FieldWithConfidence
    tanggal_lahir: FieldWithConfidence




class MobileDataInput(BaseModel):
    """Data ekstraksi sementara dari aplikasi Mobile (Sistem A)."""
    nik: Optional[str] = None
    nama: Optional[str] = None


class CorrectionDetail(BaseModel):
    """Detail koreksi per-field: nilai asal dari Mobile vs nilai koreksi dari Engine OCR."""
    field: str
    mobile_value: Optional[str] = None
    corrected_value: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response endpoint /ktp/validate: data final yang sudah direkonsiliasi."""
    nik: Optional[str] = None
    nama: Optional[str] = None
    is_corrected: bool = False
    corrections: list[CorrectionDetail] = []


# ============================================================
# Schema untuk Weighted Consensus OCR (endpoint /ktp/validate)
# ============================================================

class FieldData(BaseModel):
    """Data per-field dari Mobile, dilengkapi confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0


class MobileOCRInput(BaseModel):
    """
    Input dari Mobile (Sistem A) berformat structured per-field.
    Setiap field berisi { "value": "...", "confidence": 91.2 }.
    """
    nik: Optional[FieldData] = None
    nama: Optional[FieldData] = None
    tempat_lahir: Optional[FieldData] = None
    tanggal_lahir: Optional[FieldData] = None


class ValidatedField(BaseModel):
    """Hasil konsensus untuk satu field KTP."""
    value: Optional[str] = None
    confidence: float = 0.0
    source: str = "none"
    validated: bool = False


class QualityMetrics(BaseModel):
    """Metrik kualitas gambar KTP (dihitung dari OpenCV)."""
    score: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0


class ConsensusResponse(BaseModel):
    """Response endpoint /ktp/validate (Weighted Consensus)."""
    success: bool = True
    data: dict[str, ValidatedField] = {}
    quality: QualityMetrics = QualityMetrics()
    warnings: list[str] = []