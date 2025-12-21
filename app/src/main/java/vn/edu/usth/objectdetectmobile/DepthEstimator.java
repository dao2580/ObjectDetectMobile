package vn.edu.usth.objectdetectmobile;

import android.content.Context;

import androidx.annotation.NonNull;
import android.util.Log;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Lightweight wrapper around the Depth Anything ONNX model.
 * Creates a fresh ORT session for each inference so the detector can continue running even
 * if depth runs out of memory. This keeps the two pipelines decoupled.
 */
public class DepthEstimator implements AutoCloseable {
    private static final String TAG = "DepthEstimator";
    private static final String MODEL_NAME = "depth_anything_v2_metric_hypersim_vits.onnx";
    private static final boolean LOG_RAW_DEPTH = true;

    public static class DepthMap {
        public final float[] depth;
        public final int width, height;
        public final float min, max;

        public DepthMap(float[] depth, int width, int height, float min, float max) {
            this.depth = depth;
            this.width = width;
            this.height = height;
            this.min = min;
            this.max = max;
        }
    }

    private static final float NEAR_CM = 20f;  // clamp for extreme near noise
    private static final float FAR_CM = 200f;  // clamp for extreme far noise
    private static final float BASE_SCALE = 0.33f;
    private static final float MIN_USER_SCALE = 0.25f;
    private static final float MAX_USER_SCALE = 4f;
    private static volatile float userScale = 1f;

    // Các giá trị raw trung bình bạn đo được (ví dụ giữ vật thể ở 30cm và 200cm, log raw rồi sửa ở đây)

    // private static final float[] CAL_CM  = {20f, 40f, 50f, 75f, 100f, 150f, 200f, 300f, 400f, 500f};
    // private static final float[] CAL_RAW = {3.22f, 2.32f, 1.95f, 1.72f, 1.59f, 1.49f, 1.232f, 1.0f, 0.7f, 0.3f};

    // Depth Anything v2 metric outputs depth in meters; convert directly to centimeters.
    private static float rawToCentimeters(float raw) {
        if (Float.isNaN(raw)) return Float.NaN;
        float cm = raw * 100f * BASE_SCALE;
        cm = applyCalibration(cm);
        // Optional clamp to avoid extreme outliers from propagating.
        return Math.max(0f, Math.min(cm, 2000f));
    }

    public static void setUserScale(float scale) {
        float clamped = Math.max(MIN_USER_SCALE, Math.min(MAX_USER_SCALE, scale));
        userScale = clamped;
    }

    public static float getUserScale() {
        return userScale;
    }

    public static float applyCalibration(float depthCm) {
        if (Float.isNaN(depthCm)) return depthCm;
        return depthCm * userScale;
    }


    private final OrtEnvironment env;
    private final OrtSession.SessionOptions sessionOptions;
    private final String modelPath;

    private final int inputSize = 518;
    private final int multiple = 14;
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};

    public DepthEstimator(@NonNull Context ctx) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        modelPath = ObjectDetector.Util.cacheAsset(ctx, MODEL_NAME);
        sessionOptions = new OrtSession.SessionOptions();
    }

    public List<ObjectDetector.Detection> attachDepth(List<ObjectDetector.Detection> dets,
                                                      DepthMap depthMap) {
        if (dets == null || depthMap == null) return dets;
        List<ObjectDetector.Detection> enriched = new ArrayList<>(dets.size());
        for (ObjectDetector.Detection d : dets) {
            enriched.add(d.withDepth(averageDepth(depthMap, d)));
        }
        return enriched;
    }

    public DepthMap estimate(int[] argb, int srcW, int srcH) throws OrtException {
        Prep prep = preprocess(argb, srcW, srcH);
        long[] shape = new long[]{1, 3, prep.modelSize, prep.modelSize};
        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(prep.chw), shape);

        float[] rawDepth;
        int rawH, rawW;
        try (OrtSession session = env.createSession(modelPath, sessionOptions);
             OnnxTensor tensor = input) {
            String inputName = session.getInputInfo().keySet().iterator().next();
            try (OrtSession.Result out = session.run(Collections.singletonMap(inputName, tensor))) {
                OnnxValue ov = out.get(0);
                OnnxTensor depthTensor = (OnnxTensor) ov;
                long[] outShape = depthTensor.getInfo().getShape(); // expect [1,H,W]
                rawH = (int) outShape[1];
                rawW = (int) outShape[2];
                FloatBuffer buf = depthTensor.getFloatBuffer();
                rawDepth = new float[buf.remaining()];
                buf.get(rawDepth);
            }
        }

        float[] cropped = crop(rawDepth, rawW, rawH, prep.padX, prep.padY, prep.contentW, prep.contentH);
        float[] depthFull = resizeBilinear(cropped, prep.contentW, prep.contentH, srcW, srcH);
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (float v : depthFull) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return new DepthMap(depthFull, srcW, srcH, min, max);
    }

    private static float averageDepth(DepthMap map, ObjectDetector.Detection d) {
        if (map.width == 0 || map.height == 0) return Float.NaN;
        int x1 = clamp((int)Math.floor(d.x1), 0, map.width-1);
        int y1 = clamp((int)Math.floor(d.y1), 0, map.height-1);
        int x2 = clamp((int)Math.ceil(d.x2), 0, map.width-1);
        int y2 = clamp((int)Math.ceil(d.y2), 0, map.height-1);
        int spanX = Math.max(1, x2 - x1);
        int spanY = Math.max(1, y2 - y1);
        int stepX = Math.max(1, spanX / 12);
        int stepY = Math.max(1, spanY / 12);
        float sum = 0f; int count = 0;
        for (int y=y1; y<=y2; y+=stepY) {
            int base = y * map.width;
            for (int x=x1; x<=x2; x+=stepX) {
                sum += map.depth[base + x];
                count++;
            }
        }
        if (count == 0) return Float.NaN;
        float raw = sum / count;
        if (LOG_RAW_DEPTH) {
            Log.d(TAG, String.format(Locale.US,
                    "rawDepth=%.3f (frame min=%.3f max=%.3f, cls=%d)",
                    raw, map.min, map.max, d.cls));
        }
        return rawToCentimeters(raw);
    }


    private static float normalize(float v, float min, float max) {
        float range = max - min;
        if (range < 1e-6f) return 0f;
        float n = (v - min) / range;
        return Math.max(0f, Math.min(1f, n));
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
    }

    private static float toCentimeters(float normalized) {
        if (Float.isNaN(normalized)) return Float.NaN;
        return NEAR_CM + normalized * (FAR_CM - NEAR_CM);
    }

    private static class Prep {
        final float[] chw;
        final int modelSize;
        final int contentW, contentH;
        final int padX, padY;
        Prep(float[] chw, int modelSize, int contentW, int contentH, int padX, int padY) {
            this.chw = chw;
            this.modelSize = modelSize;
            this.contentW = contentW;
            this.contentH = contentH;
            this.padX = padX;
            this.padY = padY;
        }
    }

    private Prep preprocess(int[] argb, int srcW, int srcH) {
        int target = inputSize;
        int longest = Math.max(srcW, srcH);
        float scale = target / (float) longest;

        int scaledW = clampToRange(roundToMultiple(Math.round(srcW * scale), multiple), multiple, target);
        int scaledH = clampToRange(roundToMultiple(Math.round(srcH * scale), multiple), multiple, target);
        int[] scaled = resizeNearest(argb, srcW, srcH, scaledW, scaledH);

        int padX = Math.max(0, (target - scaledW) / 2);
        int padY = Math.max(0, (target - scaledH) / 2);

        int plane = target * target;
        float[] chw = new float[3 * plane];
        for (int y = 0; y < scaledH; y++) {
            int srcRow = y * scaledW;
            int dstRow = (y + padY) * target;
            for (int x = 0; x < scaledW; x++) {
                int p = scaled[srcRow + x];
                float r = ((p >> 16) & 0xFF) / 255f;
                float g = ((p >> 8) & 0xFF) / 255f;
                float b = (p & 0xFF) / 255f;
                int idx = dstRow + padX + x;
                chw[idx] = (r - mean[0]) / std[0];
                chw[plane + idx] = (g - mean[1]) / std[1];
                chw[2 * plane + idx] = (b - mean[2]) / std[2];
            }
        }

        return new Prep(chw, target, scaledW, scaledH, padX, padY);
    }

    private static int[] resizeNearest(int[] src, int srcW, int srcH, int dstW, int dstH) {
        int[] dst = new int[dstW*dstH];
        float sx = dstW / (float)srcW;
        float sy = dstH / (float)srcH;
        for (int y=0;y<dstH;y++){
            int py = Math.min((int)(y / sy), srcH-1);
            for (int x=0;x<dstW;x++){
                int px = Math.min((int)(x / sx), srcW-1);
                dst[y*dstW + x] = src[py*srcW + px];
            }
        }
        return dst;
    }

    private static float[] resizeBilinear(float[] src, int srcW, int srcH, int dstW, int dstH) {
        if (srcW==dstW && srcH==dstH) return src.clone();
        float[] dst = new float[dstW*dstH];
        float xRatio = dstW>1 ? (srcW-1f)/(dstW-1f):0f;
        float yRatio = dstH>1 ? (srcH-1f)/(dstH-1f):0f;
        for (int y=0;y<dstH;y++){
            float sy = y * yRatio;
            int y0 = (int)Math.floor(sy);
            int y1 = Math.min(y0+1, srcH-1);
            float ly = sy - y0;
            for (int x=0;x<dstW;x++){
                float sx = x * xRatio;
                int x0 = (int)Math.floor(sx);
                int x1 = Math.min(x0+1, srcW-1);
                float lx = sx - x0;
                float top = lerp(src[y0*srcW + x0], src[y0*srcW + x1], lx);
                float bottom = lerp(src[y1*srcW + x0], src[y1*srcW + x1], lx);
                dst[y*dstW + x] = lerp(top, bottom, ly);
            }
        }
        return dst;
    }

    private static float lerp(float a,float b,float t){ return a + (b-a)*t; }
    private static int clamp(int v,int lo,int hi){ return v<lo?lo:(Math.min(v, hi)); }
    private static int roundToMultiple(int value,int multiple){
        if (multiple<=1) return value;
        int q = Math.round(value/(float)multiple);
        return Math.max(multiple, q*multiple);
    }

    private static int clampToRange(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float[] crop(float[] src, int srcW, int srcH, int offsetX, int offsetY, int outW, int outH) {
        if (offsetX == 0 && offsetY == 0 && outW == srcW && outH == srcH) {
            return src.clone();
        }
        float[] dst = new float[outW * outH];
        for (int y = 0; y < outH; y++) {
            int srcBase = (y + offsetY) * srcW + offsetX;
            int dstBase = y * outW;
            System.arraycopy(src, srcBase, dst, dstBase, outW);
        }
        return dst;
    }

    @Override
    public void close() throws Exception {
        sessionOptions.close();
    }
}

