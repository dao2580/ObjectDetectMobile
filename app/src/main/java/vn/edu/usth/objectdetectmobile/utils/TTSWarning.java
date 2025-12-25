package vn.edu.usth.objectdetectmobile.utils;

import android.content.Context;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/*
* NOW: it handles multiple objects in the sense that it chooses the most dangerous one to speak
*       (and ignores the rest to avoid spamming).
* TODO: check this algorithm
*
* */

public class TTSWarning {
    private static final String TAG = "TTSWarning";

    // ----- Tuning -----
    private static final long SPEAK_INTERVAL_MS = 5000;
    private static final float MAX_WARNING_DISTANCE = 15.0f;
    private static final float DANGER_DISTANCE = 3.5f;
    private static final float STOP_DISTANCE = 1.0f;

    private static final float LEFT_BOUND = 0.35f;
    private static final float RIGHT_BOUND = 0.65f;

    private static final List<String> DANGEROUS_OBJECTS = Arrays.asList(
            "car", "truck", "bus", "motorcycle", "bicycle",
            "person" //, "pedestrian crossing sign", "electric pole", "tree"
    );

    private static volatile TTSWarning instance;

    public static TTSWarning getInstance(Context context) {
        if (instance == null) {
            synchronized (TTSWarning.class) {
                if (instance == null) {
                    instance = new TTSWarning(context.getApplicationContext());
                }
            }
        }
        return instance;
    }

    private TextToSpeech tts;
    private volatile boolean ready = false;
    private volatile boolean enabled = true;

    private long lastSpeakElapsedMs = 0;
    private final AtomicInteger utteranceCounter = new AtomicInteger(0);

    private TTSWarning(Context context) {
        tts = new TextToSpeech(context, this::onTtsInit);
    }

    // Called when the TTS engine finishes initialization
    private void onTtsInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            if (tts == null) return;

            int result = tts.setLanguage(new Locale("vi", "VN"));
            if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(TAG, "Vietnamese not supported, using English");
                tts.setLanguage(Locale.US);
            }

            // Callbacks can be invoked from multiple threads
            tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                @Override public void onStart(String utteranceId) {
                    Log.d(TAG, "TTS onStart: " + utteranceId);
                }
                @Override public void onDone(String utteranceId) {
                    Log.d(TAG, "TTS onDone: " + utteranceId);
                }
                @Override public void onError(String utteranceId) {
                    Log.w(TAG, "TTS onError: " + utteranceId);
                }
            });

            ready = true;
            Log.i(TAG, "TTS initialized successfully");
        } else {
            Log.e(TAG, "TTS initialization failed");
        }
    }

    public void processDetections(List<Detection> detections) {
        if (!ready || !enabled || detections == null || detections.isEmpty()) return;

        long now = SystemClock.elapsedRealtime();
        if (now - lastSpeakElapsedMs < SPEAK_INTERVAL_MS) return;

        Detection toWarn = findObjectToWarn(detections);
        if (toWarn == null) return;

        String message = buildWarningMessage(toWarn, detections);
        speak(message);

        lastSpeakElapsedMs = now;
    }

    public void stop() {
        if (tts != null && ready) tts.stop();
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) stop();
    }

    public boolean isReady() {
        return ready;
    }

    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            tts = null;
        }
        ready = false;
    }

    // --------------------- Decision logic ---------------------

    private Detection findObjectToWarn(List<Detection> detections) {
        Detection best = null;
        float bestScore = 0f;

        for (Detection det : detections) {
            float s = dangerScore(det);
            if (s > bestScore) {
                bestScore = s;
                best = det;
            }
        }
        return best;
    }

    // Per-object danger score:
    // - closer => larger score
    // - dangerous label => x1.5
    private float dangerScore(Detection det) {
        if (det == null) return 0f;
        if (det.distance <= 0f || det.distance > MAX_WARNING_DISTANCE) return 0f;

        float d = Math.max(0.30f, det.distance); // clamp for stability
        float score = 1.0f / d;
        if (isDangerous(det.label)) score *= 1.5f;
        return score;
    }

    private boolean isDangerous(String label) {
        if (label == null) return false;
        String lower = label.toLowerCase(Locale.US);
        for (String dangerous : DANGEROUS_OBJECTS) {
            if (lower.contains(dangerous)) return true;
        }
        return false;
    }

    private enum Zone { LEFT, CENTER, RIGHT }

    private Zone zoneOf(float xCenterNorm) {
        if (xCenterNorm < LEFT_BOUND) return Zone.LEFT;
        if (xCenterNorm > RIGHT_BOUND) return Zone.RIGHT;
        return Zone.CENTER;
    }

    private String avoidanceCue(Detection warn) {
        if (warn.distance <= STOP_DISTANCE) return "Dừng lại";

        switch (zoneOf(warn.xCenterNorm)) {
            case LEFT:  return "Bên trái";
            case RIGHT: return "Bên phải";
            default:    return "Phía trước";
        }
    }

    // 1) Action words (only 3)
    private enum Action { STOP, GO_LEFT, GO_RIGHT }

    private String actionWord(Action a) {
        switch (a) {
            case GO_LEFT:  return "Go left";
            case GO_RIGHT: return "Go right";
            default:       return "Stop";
        }
    }

    private String directionWord(Zone z) {
        switch (z) {
            case LEFT:  return "left";
            case RIGHT: return "right";
            default:    return "ahead";
        }
    }

    // 2) Decide action from zone (and distance)
    // - obstacle on LEFT => GO_RIGHT
    // - obstacle on RIGHT => GO_LEFT
    // - obstacle CENTER => choose safer side by threat; if both bad => STOP
    private Action pickAction(Detection warn, List<Detection> all) {
        Zone z = zoneOf(warn.xCenterNorm);

        // Stop only if very close AND in front
        if (z == Zone.CENTER && warn.distance <= STOP_DISTANCE) return Action.STOP;

        if (z == Zone.LEFT) return Action.GO_RIGHT;
        if (z == Zone.RIGHT) return Action.GO_LEFT;

        // CENTER: choose the safer side using your existing dangerScore logic
        float leftThreat = sideThreatScore(all, Zone.LEFT);
        float rightThreat = sideThreatScore(all, Zone.RIGHT);

        // if both sides are risky, keep it simple: STOP
        if (Math.min(leftThreat, rightThreat) > 2.0f) return Action.STOP;

        return (leftThreat <= rightThreat) ? Action.GO_LEFT : Action.GO_RIGHT;
    }

    private float sideThreatScore(List<Detection> all, Zone side) {
        float score = 0f;
        for (Detection d : all) {
            if (d == null) continue;
            if (d.distance <= 0f || d.distance > MAX_WARNING_DISTANCE) continue;
            if (zoneOf(d.xCenterNorm) != side) continue;

            score += dangerScore(d); // your existing per-object score
        }
        return score;
    }

    private String buildWarningMessage(Detection det, List<Detection> all) {
        Action action = pickAction(det, all);

        String obj = getVietnameseName(det.label); // or keep raw label if you want English
        Zone z = zoneOf(det.xCenterNorm);

        String distStr = String.format(Locale.US, "%.1f", det.distance);

        // Example: "Go right, car ahead 2.0 meters"
        return actionWord(action) + ", " + obj + " " + directionWord(z) + " " + distStr + " meters";
    }

    private String getVietnameseName(String label) {
        if (label == null) return "Vật thể";
        String lower = label.toLowerCase(Locale.US);

        if (lower.contains("car")) return "Xe hơi";
        if (lower.contains("truck")) return "Xe tải";
        if (lower.contains("bus")) return "Xe buýt";
        if (lower.contains("motorcycle")) return "Xe máy";
        if (lower.contains("bicycle")) return "Xe đạp";
        if (lower.contains("person")) return "Người";
        if (lower.contains("tree")) return "Cây";
        if (lower.contains("electric pole")) return "Cột điện";
        if (lower.contains("pedestrian crossing sign")) return "Biển báo";

        return label;
    }

    private void speak(String message) {
        if (!ready || tts == null) return;
        String utteranceId = "warn_" + utteranceCounter.incrementAndGet();
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, utteranceId);
    }

    // --------------------- Data model ---------------------

    public static class Detection {
        public final String label;
        public final float distance;     // meters
        public final float xCenterNorm;  // 0..1

        public Detection(String label, float distance, float xCenterNorm) {
            this.label = label;
            this.distance = distance;
            this.xCenterNorm = xCenterNorm;
        }
    }
}
