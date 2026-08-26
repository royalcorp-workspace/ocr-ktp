from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class FieldWithSourceV2(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    source: Literal["OCR", "MOBILE"] = "OCR"

class KTPOcrResponseV2(BaseModel):
    nik: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    nama: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    tempat_lahir: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    tanggal_lahir: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    jenis_kelamin: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    golongan_darah: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    alamat: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    rt_rw: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    kelurahan_desa: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    kecamatan: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    agama: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    status_perkawinan: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    pekerjaan: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    kewarganegaraan: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)
    berlaku_hingga: FieldWithSourceV2 = Field(default_factory=FieldWithSourceV2)

class ConsensusResponseV2(BaseModel):
    success: bool = True
    data: KTPOcrResponseV2 = Field(default_factory=KTPOcrResponseV2)
    warnings: List[str] = Field(default_factory=list)

class MobileDataInputV2(BaseModel):
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
    kewarganegaraan: Optional[str] = None
    berlaku_hingga: Optional[str] = None
