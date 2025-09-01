# FujiFilm Simulation Detector

Smart web app to predict optimal FujiFilm film simulation settings from an image. Inference runs entirely in the browser via ONNX Runtime Web.

- Drag & drop image upload with preview
- Client-side inference (no server required for prediction)
- Clean UI with dark mode

## Tech Stack

- Frontend: Next.js (App Router), TypeScript, Tailwind CSS
- Inference: onnxruntime-web (WASM)
- Model: ONNX exported from PyTorch
- Training: Python 3.9+, PyTorch, scikit-learn, pandas

## Repository Structure

- `frontend/` — Next.js app
  - `public/model/` — Place `filmnet.onnx` and `metadata.json` here
  - `src/components/FilmPredictor.ts` — Browser-side inferencing
  - `src/app/page.tsx` — UI (upload, preview, predict)
- `torch_train.py` — Training scripts
- `torch_to_onnx.py` — Convert PyTorch to ONNX
- `torch_predict.py` — Local Python/ONNX prediction (sanity check)
- `data/`, `model/` — Optional datasets and artifacts (gitignored)
- `pyproject.toml` — Python deps

## Frontend Setup (Next.js)

Prereqs: Node.js 18+ and npm/pnpm/yarn.

1) Install dependencies
```bash
cd frontend
npm install
```

2) Add model files
- Copy your model to:
  - `frontend/public/model/filmnet.onnx`
  - `frontend/public/model/metadata.json`

3) Run the dev server
```bash
npm run dev
# Open http://localhost:3000
```

4) Build and start production
```bash
npm run build
npm start
```

Notes:
- The app loads model files from `/model/filmnet.onnx` and `/model/metadata.json` (public path).
- Use the drag-and-drop area or “Select Image”, then click “Predict Settings”.

## Python Environment (Training/Conversion)

Prereqs: Python 3.9+

Install dependencies:
```bash
uv venv
. .venv/bin/activate
uv sync
```

Train (example):
```bash
uv python torch_train.py
```

Export to ONNX (ensure image size matches frontend):
```bash
python torch_to_onnx.py
```

Optional: local ONNX prediction sanity check
```bash
python torch_predict.py --image ./test_image.jpg
```

Then copy to frontend:
- `./model/filmnet.onnx` → `frontend/public/model/filmnet.onnx`
- `./model/metadata.json` → `frontend/public/model/metadata.json`

## Metadata Contract

`frontend/src/components/FilmPredictor.ts` expects `metadata.json` like:
- `model_info.image_size` (number)
- `model_info.input_shape` (array)
- `model_info.categorical_outputs` (string[])
- `model_info.numerical_outputs` (string[])
- `class_counts` (map: head → class count)
- `num_ranges` (map: numeric head → [min, max])
- `preprocessing.mean`, `preprocessing.std` (length-3 arrays)

Ensure these match your training pipeline.

## Troubleshooting

- Model fails to load:
  - Verify files exist at `frontend/public/model/*` and are served at `/model/*`.
  - Check browser devtools for 404/CORS issues.
- Predictions look wrong:
  - Confirm `image_size`, `mean`, `std` match training.
  - Ensure head ordering and `class_counts` align.
- Performance/size:
  - Consider model quantization for smaller ONNX.

## Development Notes

- Drag-and-drop handling and UI are in `frontend/src/app/page.tsx`.
- Inference logic is encapsulated in `FilmPredictor`.
- Dark mode supported with Tailwind’s `dark:` classes.

## License

MIT

## Acknowledgements

- ONNX Runtime Web
- Next.js
- PyTorch
