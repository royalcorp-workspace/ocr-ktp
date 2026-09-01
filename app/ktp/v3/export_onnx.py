import os
import glob
import paddle2onnx

def convert_paddle_to_onnx():
    os.makedirs("/app/models_onnx", exist_ok=True)
    print("Searching for downloaded Paddle recognition models in /root...")
    all_files = glob.glob('/root/**/*', recursive=True)
    pdmodels = [f for f in all_files if f.endswith('.pdmodel')]
    print(f"Found pdmodel files: {pdmodels}")

    rec_models = [f for f in pdmodels if 'rec' in f.lower()]
    target_pdmodel = rec_models[0] if rec_models else (pdmodels[0] if pdmodels else None)

    if target_pdmodel:
        model_dir = os.path.dirname(target_pdmodel)
        p_candidates = [f for f in os.listdir(model_dir) if f.endswith('.pdiparams') or f.endswith('.pdparams')]
        if p_candidates:
            p_file = os.path.join(model_dir, p_candidates[0])
            out_file = "/app/models_onnx/en_PP-OCRv4_rec.onnx"
            print(f"Converting {target_pdmodel} and {p_file} to {out_file}...")
            try:
                paddle2onnx.command.c_paddle_to_onnx(
                    model_file=target_pdmodel,
                    params_file=p_file,
                    save_file=out_file,
                    opset_version=11,
                    enable_onnx_checker=True
                )
                print("ONNX conversion completed successfully.")
            except Exception as e:
                print(f"paddle2onnx conversion warning: {e}")
        else:
            print("No params file found in model directory.")
    else:
        print("No pdmodel files found. RapidOCR will use optimized default runtime.")

if __name__ == "__main__":
    convert_paddle_to_onnx()
