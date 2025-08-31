import * as ort from 'onnxruntime-web';

interface ModelMetadata {
    model_info: {
        input_shape: number[];
        image_size: number;
        categorical_outputs: string[];
        numerical_outputs: string[];
    };
    encoders: {
        forward: Record<string, Record<string, number>>;
        inverse: Record<string, Record<number, string>>;
    };
    class_counts: Record<string, number>;
    num_ranges: Record<string, [number, number]>;
    preprocessing: {
        mean: number[];
        std: number[];
    };
}

export class FilmPredictor {
    private session: ort.InferenceSession | null = null;
    private metadata: ModelMetadata | null = null;

    constructor() {
        // Configure ONNX Runtime for web
        ort.env.wasm.wasmPaths = '/';
        ort.env.wasm.numThreads = 1; // Single thread for better compatibility
    }

    private async loadModel(): Promise<void> {
        if (this.session && this.metadata) return;

        try {
            // Load metadata first
            const metadataResponse = await fetch('/model/metadata.json');
            if (!metadataResponse.ok) {
                throw new Error('Failed to load model metadata');
            }
            this.metadata = await metadataResponse.json();

            // Load ONNX model
            const modelResponse = await fetch('/model/filmnet.onnx');
            if (!modelResponse.ok) {
                throw new Error('Failed to load ONNX model');
            }
            const modelBuffer = await modelResponse.arrayBuffer();

            this.session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm', 'cpu']
            });

            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Failed to load model:', error);
            throw error;
        }
    }

    private async preprocessImage(file: File): Promise<Float32Array> {
        if (!this.metadata) throw new Error('Metadata not loaded');

        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                try {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    if (!ctx) throw new Error('Failed to get canvas context');

                    const imageSize = this.metadata!.model_info.image_size;
                    canvas.width = imageSize;
                    canvas.height = imageSize;

                    // Draw and resize image
                    ctx.drawImage(img, 0, 0, imageSize, imageSize);

                    // Get image data
                    const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
                    const data = imageData.data;

                    // Convert to tensor format [1, 3, H, W] and normalize
                    const tensorData = new Float32Array(1 * 3 * imageSize * imageSize);
                    const { mean, std } = this.metadata!.preprocessing;

                    for (let i = 0; i < imageSize * imageSize; i++) {
                        const pixelIndex = i * 4; // RGBA

                        // Normalize and convert to CHW format
                        tensorData[i] = (data[pixelIndex] / 255.0 - mean[0]) / std[0]; // R
                        tensorData[imageSize * imageSize + i] = (data[pixelIndex + 1] / 255.0 - mean[1]) / std[1]; // G
                        tensorData[imageSize * imageSize * 2 + i] = (data[pixelIndex + 2] / 255.0 - mean[2]) / std[2]; // B
                    }

                    resolve(tensorData);
                } catch (error) {
                    reject(error);
                }
            };

            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = URL.createObjectURL(file);
        });
    }

    private decodeOutputs(outputs: Record<string, ort.Tensor>): Record<string, any> {
        if (!this.metadata) throw new Error('Metadata not loaded');

        const results: Record<string, any> = {};

        // Handle categorical outputs
        for (const taskName of this.metadata.model_info.categorical_outputs) {
            const taskOutput = outputs[`${taskName.toLowerCase()}_output`] || outputs['categorical_outputs'];
            if (taskOutput) {
                // Find the head index for this task
                const taskIndex = this.metadata.model_info.categorical_outputs.indexOf(taskName);

                // Get logits for this task (assuming multi-head output)
                const logits = taskOutput.data as Float32Array;
                const classCount = this.metadata.class_counts[taskName];

                // Extract logits for this specific task
                const taskLogits = logits.slice(taskIndex * classCount, (taskIndex + 1) * classCount);

                // Find argmax
                let maxIdx = 0;
                let maxVal = taskLogits[0];
                for (let i = 1; i < taskLogits.length; i++) {
                    if (taskLogits[i] > maxVal) {
                        maxVal = taskLogits[i];
                        maxIdx = i;
                    }
                }

                // Decode using inverse encoder
                const inverse = this.metadata.encoders.inverse[taskName];
                results[taskName] = inverse ? inverse[maxIdx] : maxIdx.toString();
            }
        }

        // Handle numerical outputs
        const numOutput = outputs['numerical_outputs'];
        console.log('Numerical output:', numOutput);
        if (numOutput) {
            const numData = numOutput.data as Float32Array;
            console.log('Numerical data:', numData);

            // Optional: quick sanity log to help debug shape mismatches
            if (numData.length < this.metadata.model_info.numerical_outputs.length) {
                console.warn('Numerical output length is less than expected', {
                    got: numData.length,
                    expected: this.metadata.model_info.numerical_outputs.length
                });
            }

            for (let i = 0; i < this.metadata.model_info.numerical_outputs.length; i++) {
                const taskName = this.metadata.model_info.numerical_outputs[i];
                const raw = numData[i];

                // Guard against NaN/Infinity and clamp to [0, 1]
                const safe = Number.isFinite(raw) ? raw : 0.5;
                const clamped = Math.min(1, Math.max(0, safe));

                // Denormalize
                const [min, max] = this.metadata.num_ranges[taskName] || [0, 1];
                const denormalized = clamped * (max - min) + min;
                results[taskName] = Math.round(denormalized);
            }
        }

        return results;
    }

    async predict(file: File): Promise<Record<string, any>> {
        await this.loadModel();

        if (!this.session || !this.metadata) {
            throw new Error('Model not loaded');
        }

        try {
            // Preprocess image
            const inputTensor = await this.preprocessImage(file);
            const imageSize = this.metadata.model_info.image_size;

            // Create ONNX tensor
            const tensor = new ort.Tensor('float32', inputTensor, [1, 3, imageSize, imageSize]);

            // Run inference
            const feeds = { image: tensor };
            const results = await this.session.run(feeds);

            // Debug: output names and result shapes
            try {
                console.log('Session outputNames:', this.session.outputNames);
                console.log('Result keys:', Object.keys(results));
                for (const k of Object.keys(results)) {
                    const t = results[k];
                    console.log(`Output[${k}] dims=`, (t as any).dims, 'length=', t.data.length);
                }
            } catch (e) {
                console.warn('Logging outputs failed', e);
            }

            // Decode outputs
            const predictions = this.decodeOutputs(results);

            return predictions;
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }
}
