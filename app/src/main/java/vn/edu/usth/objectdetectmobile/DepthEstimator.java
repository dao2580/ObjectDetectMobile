package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import androidx.annotation.NonNull;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
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
    private static final String DEPTH_MODEL_PREFS = "depth_models";
    private static final String MODEL_NAME_INDOOR  =
            "depth_anything_v2_metric_hypersim_vits_fp16.onnx";
    private static final String MODEL_NAME_OUTDOOR =
            "depth_anything_v2_metric_vkitti_vits_fp16.onnx";

    // Keys MUST match MainActivity's PREF_* strings
    private static final String PREF_DEPTH_MODEL_INDOOR_PATH  = "pref_depth_model_indoor_path";
    private static final String PREF_DEPTH_MODEL_OUTDOOR_PATH = "pref_depth_model_outdoor_path";

    // Kiểm tra xem đã có model usable cho mode hiện tại chưa
    // ƯU TIÊN: asset trong app  → nếu không có thì mới check file đã tải.
    public static boolean isModelAvailable(@NonNull Context ctx,
                                           @NonNull MainActivity.EnvMode mode) {
        String modelName = (mode == MainActivity.EnvMode.OUTDOOR)
                ? MODEL_NAME_OUTDOOR
                : MODEL_NAME_INDOOR;

        // 1) asset?
        AssetManager am = ctx.getAssets();
        try (InputStream is = am.open(modelName)) {
            Log.i(TAG, "Depth model asset FOUND: " + modelName);
            return true;
        } catch (IOException e) {
            Log.w(TAG, "Depth model asset missing: " + modelName);
        }

        // 2) downloaded file?
        SharedPreferences sp =
                ctx.getSharedPreferences(DEPTH_MODEL_PREFS, Context.MODE_PRIVATE);
        String prefKey = (mode == MainActivity.EnvMode.OUTDOOR)
                ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                : PREF_DEPTH_MODEL_INDOOR_PATH;

        String downloadedPath = sp.getString(prefKey, null);
        if (downloadedPath != null && new java.io.File(downloadedPath).exists()) {
            Log.i(TAG, "Depth model downloaded file FOUND: " + downloadedPath);
            return true;
        }

        Log.w(TAG, "No depth model available for mode=" + mode);
        return false;
    }


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

    // fraction of bbox used for "danger region"
    private static final float DANGER_REGION_FRAC = 0.2f;

    // region mode, mirroring Python's "center" / "bottom"
    private enum DangerRegionMode { CENTER, BOTTOM }

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

    public DepthEstimator(@NonNull Context ctx,
                          @NonNull MainActivity.EnvMode mode) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        String modelName = (mode == MainActivity.EnvMode.OUTDOOR)
                ? MODEL_NAME_OUTDOOR
                : MODEL_NAME_INDOOR;

        String finalModelPath;

        // 1) Prefer asset if present
        boolean assetExists;
        try (InputStream is = ctx.getAssets().open(modelName)) {
            assetExists = true;
        } catch (IOException e) {
            assetExists = false;
        }

        if (assetExists) {
            finalModelPath = ObjectDetector.Util.cacheAsset(ctx, modelName);
            Log.i(TAG, "Using depth model asset: " + modelName);
        } else {
            // 2) otherwise use downloaded file
            SharedPreferences sp =
                    ctx.getSharedPreferences(DEPTH_MODEL_PREFS, Context.MODE_PRIVATE);
            String prefKey = (mode == MainActivity.EnvMode.OUTDOOR)
                    ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                    : PREF_DEPTH_MODEL_INDOOR_PATH;
            String downloadedPath = sp.getString(prefKey, null);

            if (downloadedPath != null && new java.io.File(downloadedPath).exists()) {
                finalModelPath = downloadedPath;
                Log.i(TAG, "Using downloaded depth model: " + downloadedPath);
            } else {
                throw new IllegalStateException("No depth model found for mode=" + mode);
            }
        }

        modelPath = finalModelPath;
        sessionOptions = new OrtSession.SessionOptions();
    }


    public List<ObjectDetector.Detection> attachDepth(List<ObjectDetector.Detection> dets,
                                                      DepthMap depthMap) {
        if (dets == null || depthMap == null) return dets;
        List<ObjectDetector.Detection> enriched = new ArrayList<>(dets.size());
        for (ObjectDetector.Detection d : dets) {
            enriched.add(d.withDepth(minDepth(depthMap, d)));
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

        float[] cropped = crop(rawDepth, rawW, rawH,
                prep.padX, prep.padY, prep.contentW, prep.contentH);
        float[] depthFull = resizeBilinear(
                cropped, prep.contentW, prep.contentH, srcW, srcH);

        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (float v : depthFull) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return new DepthMap(depthFull, srcW, srcH, min, max);
    }

    // --- existing helper methods below unchanged ---

    private static boolean isBottomRegionClass(int cls) {
        switch (cls) {
            case 1: // bicycle
            case 2: // car
            case 3: // motorcycle
            case 5: // bus
            case 7: // truck
                return true;
            default:
                return false;
        }
    }

    private static float sampleDangerRegionRaw(DepthMap map,
                                               ObjectDetector.Detection d,
                                               float frac,
                                               DangerRegionMode mode) {
        if (map.width == 0 || map.height == 0) return Float.NaN;

        int W = map.width;
        int H = map.height;

        int x1 = clamp((int) Math.floor(d.x1), 0, W - 1);
        int y1 = clamp((int) Math.floor(d.y1), 0, H - 1);
        int x2 = clamp((int) Math.ceil(d.x2), 0, W);
        int y2 = clamp((int) Math.ceil(d.y2), 0, H);

        if (x2 <= x1 || y2 <= y1) return Float.NaN;

        int w = x2 - x1;
        int h = y2 - y1;
        if (w <= 0 || h <= 0) return Float.NaN;

        int sx1, sy1, sx2, sy2;

        if (mode == DangerRegionMode.BOTTOM) {
            int ch = (int) (h * frac);
            if (ch <= 0) return Float.NaN;

            int yStart = Math.max(y1, y2 - ch);

            int centerBandWidth = (int) (w * 0.5f);
            if (centerBandWidth <= 0) return Float.NaN;

            int cx = (x1 + x2) / 2;
            int xStart = Math.max(x1, cx - centerBandWidth / 2);
            int xEnd = Math.min(x2, xStart + centerBandWidth);
            if (xEnd <= xStart) return Float.NaN;

            sx1 = xStart;
            sx2 = xEnd;
            sy1 = yStart;
            sy2 = y2;
        } else {
            int cw = (int) (w * frac);
            int ch = (int) (h * frac);
            if (cw <= 0 || ch <= 0) return Float.NaN;

            int cx = (x1 + x2) / 2;
            int cy = (y1 + y2) / 2;

            int cx1 = Math.max(0, cx - cw / 2);
            int cy1 = Math.max(0, cy - ch / 2);
            int cx2 = Math.min(W, cx1 + cw);
            int cy2 = Math.min(H, cy1 + ch);
            if (cx2 <= cx1 || cy2 <= cy1) return Float.NaN;

            sx1 = cx1;
            sx2 = cx2;
            sy1 = cy1;
            sy2 = cy2;
        }

        if (sx2 <= sx1 || sy2 <= sy1) return Float.NaN;

        int regionW = sx2 - sx1;
        int regionH = sy2 - sy1;
        int capacity = regionW * regionH;
        if (capacity <= 0) return Float.NaN;

        float[] vals = new float[capacity];
        int n = 0;

        for (int y = sy1; y < sy2; y++) {
            int base = y * W;
            for (int x = sx1; x < sx2; x++) {
                float v = map.depth[base + x];
                if (v > 0f) {
                    vals[n++] = v;
                }
            }
        }

        if (n == 0) return Float.NaN;

        float nearest = Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            float v = vals[i];
            if (v > 0f && v < nearest) {
                nearest = v;
            }
        }

        return (nearest == Float.MAX_VALUE) ? Float.NaN : nearest;
    }

    private static float minDepth(DepthMap map, ObjectDetector.Detection d) {
        if (map == null || map.width == 0 || map.height == 0) return Float.NaN;

        DangerRegionMode mode = isBottomRegionClass(d.cls)
                ? DangerRegionMode.BOTTOM
                : DangerRegionMode.CENTER;

        float raw = sampleDangerRegionRaw(map, d, DANGER_REGION_FRAC, mode);

        if (Float.isNaN(raw)) return Float.NaN;

        if (LOG_RAW_DEPTH) {
            Log.d(TAG, String.format(Locale.US,
                    "rawDepthMedian=%.3f (frame min=%.3f max=%.3f, cls=%d, mode=%s)",
                    raw, map.min, map.max, d.cls, mode.name()));
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

        int scaledW = clampToRange(
                roundToMultiple(Math.round(srcW * scale), multiple), multiple, target);
        int scaledH = clampToRange(
                roundToMultiple(Math.round(srcH * scale), multiple), multiple, target);
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
        int[] dst = new int[dstW * dstH];
        float sx = dstW / (float) srcW;
        float sy = dstH / (float) srcH;
        for (int y = 0; y < dstH; y++) {
            int py = Math.min((int) (y / sy), srcH - 1);
            for (int x = 0; x < dstW; x++) {
                int px = Math.min((int) (x / sx), srcW - 1);
                dst[y * dstW + x] = src[py * srcW + px];
            }
        }
        return dst;
    }

    private static float[] resizeBilinear(float[] src, int srcW, int srcH,
                                          int dstW, int dstH) {
        if (srcW == dstW && srcH == dstH) return src.clone();
        float[] dst = new float[dstW * dstH];
        float xRatio = dstW > 1 ? (srcW - 1f) / (dstW - 1f) : 0f;
        float yRatio = dstH > 1 ? (srcH - 1f) / (dstH - 1f) : 0f;
        for (int y = 0; y < dstH; y++) {
            float sy = y * yRatio;
            int y0 = (int) Math.floor(sy);
            int y1 = Math.min(y0 + 1, srcH - 1);
            float ly = sy - y0;
            for (int x = 0; x < dstW; x++) {
                float sx = x * xRatio;
                int x0 = (int) Math.floor(sx);
                int x1 = Math.min(x0 + 1, srcW - 1);
                float lx = sx - x0;
                float top = lerp(src[y0 * srcW + x0], src[y0 * srcW + x1], lx);
                float bottom = lerp(src[y1 * srcW + x0], src[y1 * srcW + x1], lx);
                dst[y * dstW + x] = lerp(top, bottom, ly);
            }
        }
        return dst;
    }

    private static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
    private static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : Math.min(v, hi);
    }
    private static int roundToMultiple(int value, int multiple) {
        if (multiple <= 1) return value;
        int q = Math.round(value / (float) multiple);
        return Math.max(multiple, q * multiple);
    }
    private static int clampToRange(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float[] crop(float[] src, int srcW, int srcH,
                                int offsetX, int offsetY, int outW, int outH) {
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
