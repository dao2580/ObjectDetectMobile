package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import androidx.annotation.NonNull;
import ai.onnxruntime.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.Arrays;

import static java.lang.Math.*;

public class ObjectDetector implements AutoCloseable {
    public static class Detection {
        public final float x1, y1, x2, y2, score, depth;
        public final int cls;

        public Detection(float x1, float y1, float x2, float y2, float score, int cls) {
            this(x1, y1, x2, y2, score, cls, Float.NaN);
        }

        private Detection(float x1, float y1, float x2, float y2,
                          float score, int cls, float depth) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.cls = cls;
            this.depth = depth;
        }

        public Detection withDepth(float depthValue) {
            return new Detection(x1, y1, x2, y2, score, cls, depthValue);
        }
    }

    // ---------------------------------------------------------------------------------------------
    //  ORT + model config
    // ---------------------------------------------------------------------------------------------
    private final OrtEnvironment env;
    private final OrtSession session;
    private final int inputW = 640, inputH = 640;
    private final float confThresh = 0.25f, iouThresh = 0.45f;
    private final String inputName;

    // Reused input tensor buffer: [1, 3, H, W] in CHW format
    private final float[] inputTensor;
    private final FloatBuffer inputBuffer;

    public ObjectDetector(@NonNull Context ctx) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        String modelPath = Util.cacheAsset(ctx, "yolov8m_compatible.onnx");

        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        // NO enable NNAPI on supported devices (falls back to CPU if not available)
        // (Optional) tune threads & optimization if you want:
        // so.setIntraOpNumThreads(1);
        // so.setInterOpNumThreads(1);
        // so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL);

        session = env.createSession(modelPath, so);
        inputName = session.getInputInfo().keySet().iterator().next();

        // Allocate reusable input buffer once
        inputTensor = new float[3 * inputW * inputH];
        inputBuffer = FloatBuffer.wrap(inputTensor);
    }

    // Struct to carry letterbox parameters
    private static class LetterboxParams {
        final float scale;
        final float padX;
        final float padY;

        LetterboxParams(float scale, float padX, float padY) {
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }
    }

    /**
     * Fused resize + letterbox + CHW normalize into a single pass.
     * - src: ARGB int[] from camera
     * - srcW/H: original frame size
     * - Writes into inputTensor (reused) in CHW order.
     * - Returns scale + padding so we can map boxes back to src coords.
     */
    private LetterboxParams fillInputTensorFromSrc(int[] src, int srcW, int srcH) {
        // Standard YOLO-style letterbox scaling
        float r = Math.min(inputW / (float) srcW, inputH / (float) srcH);
        int nw = (int) (srcW * r);
        int nh = (int) (srcH * r);
        int dx = (inputW - nw) / 2;
        int dy = (inputH - nh) / 2;

        int area = inputW * inputH;
        int rOffset = 0;
        int gOffset = area;
        int bOffset = 2 * area;

        // Zero padding (background 0.0)
        Arrays.fill(inputTensor, 0f);

        // Resize + letterbox + CHW in one pass
        for (int y = 0; y < nh; y++) {
            int sy = Math.min((int) (y / r), srcH - 1);
            int srcRow = sy * srcW;
            int dstY = y + dy;
            int dstRow = dstY * inputW;

            for (int x = 0; x < nw; x++) {
                int sx = Math.min((int) (x / r), srcW - 1);
                int dstX = x + dx;
                int dstIndex = dstRow + dstX;

                int p = src[srcRow + sx];
                float rf = ((p >>> 16) & 0xFF) / 255f;
                float gf = ((p >>> 8) & 0xFF) / 255f;
                float bf = (p & 0xFF) / 255f;

                inputTensor[rOffset + dstIndex] = rf;
                inputTensor[gOffset + dstIndex] = gf;
                inputTensor[bOffset + dstIndex] = bf;
            }
        }

        return new LetterboxParams(r, dx, dy);
    }

    // ---------------------------------------------------------------------------------------------
    //  Public detect API
    // ---------------------------------------------------------------------------------------------
    public List<Detection> detect(int[] argb, int srcW, int srcH) throws OrtException {
        // Fused preprocessing: fills reusable inputTensor
        LetterboxParams lb = fillInputTensorFromSrc(argb, srcW, srcH);

        // Make sure buffer position is at 0 before creating tensor
        inputBuffer.rewind();

        try (OnnxTensor input = OnnxTensor.createTensor(
                env, inputBuffer, new long[]{1, 3, inputH, inputW});
             OrtSession.Result out = session.run(
                     Collections.singletonMap(inputName, input))) {

            OnnxValue ov = out.get(0);
            return parse(ov, lb.scale, lb.padX, lb.padY, srcW, srcH);
        }
    }

    // ---------------------------------------------------------------------------------------------
    //  Parse YOLOv8 output + NMS
    // ---------------------------------------------------------------------------------------------
    private List<Detection> parse(OnnxValue val,
                                  float scale, float padX, float padY,
                                  int imgW, int imgH) throws OrtException {
        OnnxTensor t = (OnnxTensor) val;
        long[] shape = t.getInfo().getShape(); // expect [1,84,N] or [1,N,84]

        // Get flat float array from tensor
        FloatBuffer fb = t.getFloatBuffer();
        float[] flat = new float[fb.remaining()];
        fb.get(flat);

        int dim1 = (int) shape[1];
        int dim2 = (int) shape[2];
        boolean colsAreProps = (dim1 == 84); // [1,84,N] if true
        int props = colsAreProps ? dim1 : dim2;
        int clsCount = props - 4;
        int N = colsAreProps ? dim2 : dim1;

        List<Detection> dets = new ArrayList<>(N);

        if (colsAreProps) {
            // layout [1, 84, N]
            int stride = N; // properties are stored in contiguous rows
            for (int i = 0; i < N; i++) {
                float x = flat[i];
                float y = flat[stride + i];
                float w = flat[2 * stride + i];
                float h = flat[3 * stride + i];

                int bestC = -1;
                float bestS = 0f;
                for (int c = 0; c < clsCount; c++) {
                    float s = flat[(4 + c) * stride + i];
                    if (s > bestS) {
                        bestS = s;
                        bestC = c;
                    }
                }
                if (bestS < confThresh) continue;

                float bx = x - w / 2f;
                float by = y - h / 2f;
                float ex = x + w / 2f;
                float ey = y + h / 2f;

                float x1 = clamp((bx - padX) / scale, 0, imgW);
                float y1 = clamp((by - padY) / scale, 0, imgH);
                float x2 = clamp((ex - padX) / scale, 0, imgW);
                float y2 = clamp((ey - padY) / scale, 0, imgH);

                dets.add(new Detection(x1, y1, x2, y2, bestS, bestC));
            }
        } else {
            // layout [1, N, 84]
            for (int i = 0; i < N; i++) {
                int base = i * props;
                float x = flat[base + 0];
                float y = flat[base + 1];
                float w = flat[base + 2];
                float h = flat[base + 3];

                int bestC = -1;
                float bestS = 0f;
                for (int c = 0; c < clsCount; c++) {
                    float s = flat[base + 4 + c];
                    if (s > bestS) {
                        bestS = s;
                        bestC = c;
                    }
                }
                if (bestS < confThresh) continue;

                float bx = x - w / 2f;
                float by = y - h / 2f;
                float ex = x + w / 2f;
                float ey = y + h / 2f;

                float x1 = clamp((bx - padX) / scale, 0, imgW);
                float y1 = clamp((by - padY) / scale, 0, imgH);
                float x2 = clamp((ex - padX) / scale, 0, imgW);
                float y2 = clamp((ey - padY) / scale, 0, imgH);

                dets.add(new Detection(x1, y1, x2, y2, bestS, bestC));
            }
        }

        return nms(dets, iouThresh);
    }

    private static float clamp(float v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static float iou(Detection A, Detection B) {
        float ix1 = max(A.x1, B.x1), iy1 = max(A.y1, B.y1);
        float ix2 = min(A.x2, B.x2), iy2 = min(A.y2, B.y2);
        float iw = max(0f, ix2 - ix1), ih = max(0f, iy2 - iy1);
        float inter = iw * ih;
        float a = (A.x2 - A.x1) * (A.y2 - A.y1);
        float b = (B.x2 - B.x1) * (B.y2 - B.y1);
        return inter / (a + b - inter + 1e-6f);
    }

    private static List<Detection> nms(List<Detection> in, float iouTh) {
        ArrayList<Detection> dets = new ArrayList<>(in);
        dets.sort((d1, d2) -> Float.compare(d2.score, d1.score));
        List<Detection> keep = new ArrayList<>();
        while (!dets.isEmpty()) {
            Detection a = dets.remove(0);
            keep.add(a);
            dets.removeIf(b -> b.cls == a.cls && iou(a, b) > iouTh);
        }
        return keep;
    }

    @Override
    public void close() throws Exception {
        session.close();
        // env is a singleton managed by ORT; you usually don’t close it here
        // to avoid interfering with other sessions.
    }

    // Utility to read asset fully
    static class Util {
        static byte[] readAllBytes(android.content.res.AssetManager am, String name) {
            try (java.io.InputStream is = am.open(name);
                 java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream()) {
                byte[] buf = new byte[1 << 16];
                int r;
                while ((r = is.read(buf)) != -1) bos.write(buf, 0, r);
                return bos.toByteArray();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        static String cacheAsset(Context ctx, String assetName) {
            File dir = new File(ctx.getFilesDir(), "models");
            if (!dir.exists()) dir.mkdirs();
            File out = new File(dir, assetName);
            if (out.exists() && out.length() > 0) return out.getAbsolutePath();
            try (InputStream is = ctx.getAssets().open(assetName);
                 FileOutputStream fos = new FileOutputStream(out)) {
                byte[] buf = new byte[1 << 16];
                int r;
                while ((r = is.read(buf)) != -1) fos.write(buf, 0, r);
                fos.flush();
                return out.getAbsolutePath();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
