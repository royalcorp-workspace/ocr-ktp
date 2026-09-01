import os
import glob
import paddle2onnx

def convert_paddle_to_onnx():
    print("Searching for downloaded Paddle recognition models...")
    rec_candidates = [
        d for d in glob.glob('/root/**/*', recursive=True)
        if os.path.isdir(d) and ('rec' in d.lower() or 'en' in d.lower()) and any(f.endswith('.pdmodel') for f in os.listdir(d))
    ]
    model_dir = rec_candidates[0] if rec_candidates else None
    print(f"Found recognition model directory: {model_dir}")

    if model_dir:
        m_files = [f for f in os.listdir(model_dir) if f.endswith('.pdmodel')]
        p_files = [f for f in os.listdir(model_dir) if f.endswith('.pdiparams')]
        if m_files and p_files:
            os.makedirs("/app/models_onnx", exist_ok=True)
            out_file = "/app/models_onnx/en_PP-OCRv4_rec.onnx"
            print(f"Converting {m_files[0]} to {out_file}...")
            paddle2onnx.command.c_paddle_to_onnx(
                model_file=os.path.join(model_dir, m_files[0]),
                params_file=os.path.join(model_dir, p_files[0]),
                save_file=out_file,
                opset_version=11,
                enable_onnx_checker=True
            )
            print("ONNX conversion completed successfully.")
        else:
            print("No .pdmodel or .pdiparams found in model directory.")
    else:
        print("No downloaded Paddle recognition models found in /root.")

if __name__ == "__main__":
    convert_paddle_to_onnx()
