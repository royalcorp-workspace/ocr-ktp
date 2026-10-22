from app.ktp.schemas import KTPData
from app.ktp.extractor.common import parse_label_blocks, extract_text_field
from app.ktp.extractor import identity, address, civil, validators
from app.ktp.v1.regional_normalizer import normalize_regional_text, fix_common_ocr_typos


class KTPExtractor:
    """
    Orkestrator ekstraksi KTP-el Indonesia. Logic per-field ada di identity.py,
    address.py, civil.py; validasi lintas-field ada di validators.py.
    """

    def extract(self, raw_text: str) -> KTPData:
        if not raw_text or not raw_text.strip():
            return KTPData()

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        blocks = parse_label_blocks(cleaned_text)
        is_cropped_or_low_block = len(blocks) < 5

        tempat_lahir, tanggal_lahir = identity.extract_tempat_tanggal_lahir(blocks.get("TEMPAT_TGL_LAHIR"), cleaned_text)
        jenis_kelamin = identity.extract_jenis_kelamin(blocks.get("JENIS_KELAMIN"), cleaned_text)
        nik = identity.extract_nik(blocks.get("NIK"), cleaned_text)
        nama = identity.extract_nama(blocks.get("NAMA"), cleaned_text)
        golongan_darah = identity.extract_golongan_darah(blocks.get("GOL_DARAH"), cleaned_text)
        alamat = address.extract_alamat(blocks.get("ALAMAT"), cleaned_text)
        rt_rw, rt_spillover = address.extract_rt_rw_with_spillover(blocks.get("RT_RW"), cleaned_text, blocks.get("ALAMAT"))
        if rt_spillover:
            if not alamat:
                alamat = rt_spillover
            elif rt_spillover.upper() not in alamat.upper():
                alamat = f"{alamat} {rt_spillover}".strip()

        kelurahan_desa = extract_text_field(blocks.get("KEL_DESA"))
        kecamatan = extract_text_field(blocks.get("KECAMATAN"))
        agama = civil.extract_agama(blocks.get("AGAMA"), cleaned_text)
        status_perkawinan = civil.extract_status_perkawinan(blocks.get("STATUS_PERKAWINAN"), cleaned_text)
        pekerjaan = civil.extract_pekerjaan(blocks.get("PEKERJAAN"), cleaned_text)
        kewarganegaraan = civil.extract_kewarganegaraan(blocks.get("KEWARGANEGARAAN"), cleaned_text)
        berlaku_hingga = civil.extract_berlaku_hingga(blocks.get("BERLAKU_HINGGA"), cleaned_text)

        # Dual-Source Date-of-Birth Resolver (OCR Fisik vs NIK Redundancy)
        if not tanggal_lahir and nik:
            tanggal_lahir = validators.derive_dob_from_nik(nik)

        # Legal Dukcapil Default untuk Kewarganegaraan jika label pudar/hilang
        if not kewarganegaraan:
            if "WNA" in cleaned_text.upper():
                kewarganegaraan = "WNA"
            else:
                has_regular_header = any(hdr in cleaned_text.upper() for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "REPUBLIK"])
                if has_regular_header or (nik and len(nik) == 16):
                    kewarganegaraan = "WNI"

        # Heuristic fallback untuk gambar tercrop / label terdeteksi < 5
        if is_cropped_or_low_block:
            if not kelurahan_desa:
                for line in lines:
                    line_up = line.upper()
                    if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "NAMA"]):
                        continue
                    if any(kw in line_up for kw in ["KEL", "DESA", "KELURAHAN"]):
                        cand = extract_text_field(line)
                        if cand:
                            kelurahan_desa = cand
                            break

            if not kecamatan:
                for line in lines:
                    line_up = line.upper()
                    if any(hdr in line_up for hdr in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "NAMA"]):
                        continue
                    if any(kw in line_up for kw in ["KEC", "KACAMATAN", "KECAMATAN"]):
                        cand = extract_text_field(line)
                        if cand:
                            kecamatan = cand
                            break

            if not agama:
                agama = civil.extract_agama(None, cleaned_text)
            if not status_perkawinan:
                status_perkawinan = civil.extract_status_perkawinan(None, cleaned_text)
            if not pekerjaan:
                pekerjaan = civil.extract_pekerjaan(None, cleaned_text)
            if not kewarganegaraan:
                kewarganegaraan = civil.extract_kewarganegaraan(None, cleaned_text)
                if not kewarganegaraan:
                    kewarganegaraan = "WNI" if "WNA" not in cleaned_text.upper() else "WNA"
            if not berlaku_hingga:
                berlaku_hingga = civil.extract_berlaku_hingga(None, cleaned_text)


        # === Post-Processing & Validasi Kontekstual (Read-only consistency) ===
        is_dob_consistent = validators.check_nik_dob_consistency(nik, tanggal_lahir, jenis_kelamin)
        tempat_lahir = validators.correct_tempat_lahir_fuzzy(tempat_lahir)

        if alamat:
            alamat = fix_common_ocr_typos(alamat)
        if kelurahan_desa:
            kelurahan_desa = normalize_regional_text(kelurahan_desa, "kelurahan_desa")
        if kecamatan:
            kecamatan = normalize_regional_text(kecamatan, "kecamatan")

        return KTPData(
            nik=nik,
            nama=nama,
            tempat_lahir=tempat_lahir,
            tanggal_lahir=tanggal_lahir,
            jenis_kelamin=jenis_kelamin,
            golongan_darah=golongan_darah,
            alamat=alamat,
            rt_rw=rt_rw,
            kelurahan_desa=kelurahan_desa,
            kecamatan=kecamatan,
            agama=agama,
            status_perkawinan=status_perkawinan,
            pekerjaan=pekerjaan,
            kewarganegaraan=kewarganegaraan,
            berlaku_hingga=berlaku_hingga,
        )