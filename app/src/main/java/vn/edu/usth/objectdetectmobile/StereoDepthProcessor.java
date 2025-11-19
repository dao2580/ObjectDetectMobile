package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Converts the monocular depth map into a stereo-inspired estimate by using the real focal length
 * and baseline of the active camera module. The per-object disparity is inferred from the raw depth
 * map and refined with the learned model output.
 */
public class StereoDepthProcessor {
    private static final String TAG = "StereoDepthProcessor";
    private static final float DEFAULT_BASELINE_METERS = 0.06f;
    private static final float DISPARITY_EPS = 1e-4f;
    private static final float STEREO_WEIGHT = 0.65f;
    private static final float NEAR_METERS = 0.2f;
    private static final float FAR_METERS = 5.0f;

    private final CameraCharacteristics logicalCharacteristics;
    private final float baselineMeters;
    private volatile int referenceWidth = 0;
    private volatile int referenceHeight = 0;
    private volatile float focalLengthPixels = 0f;

    public StereoDepthProcessor(@NonNull Context ctx,
                                @NonNull CameraCharacteristics logicalCharacteristics) {
        this.logicalCharacteristics = logicalCharacteristics;
        this.baselineMeters = resolveBaseline(ctx, logicalCharacteristics);
    }

    public void setReferenceSize(int width, int height) {
        if (width <= 0 || height <= 0) return;
        if (width == referenceWidth && height == referenceHeight && focalLengthPixels > 0f) {
            return;
        }
        this.referenceWidth = width;
        this.referenceHeight = height;
        this.focalLengthPixels = computeFocalLengthPixels(width);
    }

    public List<ObjectDetector.Detection> fuseDepth(@NonNull DepthEstimator.DepthMap depthMap,
                                                    @NonNull List<ObjectDetector.Detection> dets,
                                                    int colorWidth,
                                                    int colorHeight) {
        if (dets.isEmpty()) return dets;
        if (referenceWidth == 0 || referenceHeight == 0) {
            setReferenceSize(colorWidth, colorHeight);
        }
        List<ObjectDetector.Detection> out = new ArrayList<>(dets.size());
        for (ObjectDetector.Detection d : dets) {
            float raw = sampleRawDepth(depthMap, d);
            float stereoDepth = convertRawToStereoDepth(raw, depthMap);
            float fused = fuseDepthValues(d.depth, stereoDepth);
            out.add(d.withDepth(fused));
        }
        return out;
    }

    private float sampleRawDepth(@NonNull DepthEstimator.DepthMap map,
                                 @NonNull ObjectDetector.Detection d) {
        if (map.width <= 0 || map.height <= 0) return Float.NaN;
        int x1 = clamp(Math.round(d.x1), 0, map.width - 1);
        int y1 = clamp(Math.round(d.y1), 0, map.height - 1);
        int x2 = clamp(Math.round(d.x2), 0, map.width - 1);
        int y2 = clamp(Math.round(d.y2), 0, map.height - 1);
        int spanX = Math.max(1, x2 - x1);
        int spanY = Math.max(1, y2 - y1);
        int stepX = Math.max(1, spanX / 10);
        int stepY = Math.max(1, spanY / 10);
        float sum = 0f;
        int count = 0;
        for (int y = y1; y <= y2; y += stepY) {
            int base = y * map.width;
            for (int x = x1; x <= x2; x += stepX) {
                float v = map.depth[base + x];
                if (!Float.isNaN(v) && v > 0f) {
                    sum += v;
                    count++;
                }
            }
        }
        if (count == 0) return Float.NaN;
        return sum / count;
    }

    private float convertRawToStereoDepth(float raw,
                                          @NonNull DepthEstimator.DepthMap map) {
        if (Float.isNaN(raw) || focalLengthPixels <= 0f || baselineMeters <= 0f) {
            return Float.NaN;
        }
        float span = map.max - map.min;
        if (span < 1e-6f) return Float.NaN;
        float normalized = clamp01((raw - map.min) / span);
        float disparityNear = (focalLengthPixels * baselineMeters) / NEAR_METERS;
        float disparityFar = (focalLengthPixels * baselineMeters) / FAR_METERS;
        float disparity = disparityFar + (1f - normalized) * (disparityNear - disparityFar);
        float depthMeters = (focalLengthPixels * baselineMeters) /
                Math.max(disparity, DISPARITY_EPS);
        return depthMeters * 100f;
    }

    private float fuseDepthValues(float modelDepthCm, float stereoDepthCm) {
        boolean hasModel = !Float.isNaN(modelDepthCm) && modelDepthCm > 0f;
        boolean hasStereo = !Float.isNaN(stereoDepthCm) && stereoDepthCm > 0f;
        if (!hasModel && !hasStereo) {
            return Float.NaN;
        }
        if (!hasStereo) return modelDepthCm;
        if (!hasModel) return stereoDepthCm;
        float modelMeters = Math.max(modelDepthCm / 100f, 1e-3f);
        float stereoMeters = Math.max(stereoDepthCm / 100f, 1e-3f);
        float disparityModel = (focalLengthPixels * baselineMeters) / modelMeters;
        float disparityStereo = (focalLengthPixels * baselineMeters) / stereoMeters;
        float fusedDisparity = (1f - STEREO_WEIGHT) * disparityModel
                + STEREO_WEIGHT * disparityStereo;
        float fusedDepthMeters = (focalLengthPixels * baselineMeters) /
                Math.max(fusedDisparity, DISPARITY_EPS);
        return fusedDepthMeters * 100f;
    }

    private float computeFocalLengthPixels(int targetWidth) {
        float[] intrinsics = logicalCharacteristics.get(
                CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        Size pixelArray = logicalCharacteristics.get(
                CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
        if (intrinsics != null && intrinsics.length >= 2 && pixelArray != null
                && pixelArray.getWidth() > 0) {
            float fx = intrinsics[0];
            return fx * (targetWidth / (float) pixelArray.getWidth());
        }
        float[] focals = logicalCharacteristics.get(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        SizeF sensorSize = logicalCharacteristics.get(
                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        if (focals != null && focals.length > 0 && sensorSize != null
                && sensorSize.getWidth() > 0 && pixelArray != null
                && pixelArray.getWidth() > 0) {
            float pxPerMm = pixelArray.getWidth() / sensorSize.getWidth();
            float fxSensor = focals[0] * pxPerMm;
            return fxSensor * (targetWidth / (float) pixelArray.getWidth());
        }
        return 0f;
    }

    private float resolveBaseline(@NonNull Context ctx,
                                  @NonNull CameraCharacteristics logicalChars) {
        CameraManager mgr = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
        if (mgr == null) return DEFAULT_BASELINE_METERS;
        Set<String> ids = logicalChars.getPhysicalCameraIds();
        if (ids != null && ids.size() >= 2) {
            List<String> list = new ArrayList<>(ids);
            float best = 0f;
            for (int i = 0; i < list.size(); i++) {
                for (int j = i + 1; j < list.size(); j++) {
                    float delta = computeBaselineForPair(mgr, list.get(i), list.get(j));
                    if (delta > best) best = delta;
                }
            }
            if (best > 0f) return best;
        }
        float[] pose = logicalChars.get(CameraCharacteristics.LENS_POSE_TRANSLATION);
        if (pose != null && pose.length >= 3) {
            float magnitude = Math.abs(pose[0]) + Math.abs(pose[1]) + Math.abs(pose[2]);
            if (magnitude > 0f) {
                return Math.max(magnitude, DEFAULT_BASELINE_METERS);
            }
        }
        return DEFAULT_BASELINE_METERS;
    }

    private float computeBaselineForPair(CameraManager mgr, String a, String b) {
        try {
            CameraCharacteristics ca = mgr.getCameraCharacteristics(a);
            CameraCharacteristics cb = mgr.getCameraCharacteristics(b);
            float[] ta = ca.get(CameraCharacteristics.LENS_POSE_TRANSLATION);
            float[] tb = cb.get(CameraCharacteristics.LENS_POSE_TRANSLATION);
            if (ta == null || tb == null || ta.length < 3 || tb.length < 3) return 0f;
            float dx = ta[0] - tb[0];
            float dy = ta[1] - tb[1];
            float dz = ta[2] - tb[2];
            return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        } catch (Exception e) {
            Log.w(TAG, "baseline query failed", e);
            return 0f;
        }
    }

    private static int clamp(int v, int min, int max) {
        if (v < min) return min;
        return Math.min(v, max);
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }

    public void clear() {
        referenceWidth = 0;
        referenceHeight = 0;
        focalLengthPixels = 0f;
    }
}
