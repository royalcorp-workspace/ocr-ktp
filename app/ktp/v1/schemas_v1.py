from typing import Optional
from pydantic import BaseModel

class FieldWithSource(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    source: str = "NONE" # ROI, GENERAL, atau CONSENSUS

class KTPOcrResponseV1(BaseModel):
    nik: FieldWithSource
    nama: FieldWithSource
    tempat_lahir: FieldWithSource
    tanggal_lahir: FieldWithSource
    jenis_kelamin: FieldWithSource
    golongan_darah: FieldWithSource
    alamat: FieldWithSource
    rt_rw: FieldWithSource
    kelurahan_desa: FieldWithSource
    kecamatan: FieldWithSource
    agama: FieldWithSource
    status_perkawinan: FieldWithSource
    pekerjaan: FieldWithSource
    kewarganegaraan: FieldWithSource
    berlaku_hingga: FieldWithSource

class QualityMetricsV1(BaseModel):
    score: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0

class ConsensusResponseV1(BaseModel):
    success: bool = True
    data: KTPOcrResponseV1
    quality: QualityMetricsV1 = QualityMetricsV1()
    warnings: list[str] = []

class MobileDataInputV1(BaseModel):
    """Input dari mobile (Sistem A) untuk 14 field di v1."""
    nik: Optional[dict] = None
    nama: Optional[dict] = None
    tempat_lahir: Optional[dict] = None
    tanggal_lahir: Optional[dict] = None
    jenis_kelamin: Optional[dict] = None
    golongan_darah: Optional[dict] = None
    alamat: Optional[dict] = None
    rt_rw: Optional[dict] = None
    kelurahan_desa: Optional[dict] = None
    kecamatan: Optional[dict] = None
    agama: Optional[dict] = None
    status_perkawinan: Optional[dict] = None
    pekerjaan: Optional[dict] = None
    kewarganegaraan: Optional[dict] = None
    berlaku_hingga: Optional[dict] = None
