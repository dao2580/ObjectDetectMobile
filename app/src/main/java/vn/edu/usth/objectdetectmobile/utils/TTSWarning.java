package vn.edu.usth.objectdetectmobile.utils;

import android.content.Context;
import android.speech.tts.TextToSpeech;
import android.util.Log;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;


public class TTSWarning {
    private static final String TAG = "TTSWarning";


    private static final long SPEAK_INTERVAL = 5000; // 3 giây giữa các cảnh báo
    private static final float MAX_WARNING_DISTANCE = 5.0f; // Cảnh báo khi < 5m
    private static final float DANGER_DISTANCE = 2.0f; // Nguy hiểm khi < 2m


    private static final List<String> DANGEROUS_OBJECTS = Arrays.asList(
            "car", "truck", "bus", "motorcycle", "bicycle",
            "person", "pedestrian crossing sign", "electric pole", "tree"
    );


    private static TTSWarning instance;

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
    private boolean ready = false;
    private long lastSpeakTime = 0;
    private boolean enabled = true;


    private TTSWarning(Context context) {
        tts = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {

                int result = tts.setLanguage(new Locale("vi", "VN"));

                if (result == TextToSpeech.LANG_MISSING_DATA ||
                        result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Vietnamese not supported, using English");
                    tts.setLanguage(Locale.US);
                }

                ready = true;
                Log.i(TAG, "TTS initialized successfully");
            } else {
                Log.e(TAG, "TTS initialization failed");
            }
        });
    }



    public void processDetections(List<Detection> detections) {
        if (!ready || !enabled || detections == null || detections.isEmpty()) {
            return;
        }

        // Throttle: chỉ phát cảnh báo mỗi SPEAK_INTERVAL ms
        long now = System.currentTimeMillis();
        if (now - lastSpeakTime < SPEAK_INTERVAL) {
            return;
        }

        // Tìm object cần cảnh báo
        Detection toWarn = findObjectToWarn(detections);

        if (toWarn != null) {
            String message = buildWarningMessage(toWarn);
            speak(message);
            lastSpeakTime = now;
        }
    }


    public void stop() {
        if (tts != null && ready) {
            tts.stop();
        }
    }


    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            stop();
        }
    }


    public boolean isReady() {
        return ready;
    }


    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            ready = false;
        }
    }



    /**
     * Tìm object cần cảnh báo với ưu tiên:
     * 1. Vật nguy hiểm gần nhất
     * 2. Vật gần nhất < MAX_WARNING_DISTANCE
     */
    private Detection findObjectToWarn(List<Detection> detections) {
        Detection closestDanger = null;
        Detection closestObject = null;
        float minDangerDist = Float.MAX_VALUE;
        float minObjectDist = Float.MAX_VALUE;

        for (Detection det : detections) {
            if (det.distance <= 0 || det.distance > MAX_WARNING_DISTANCE) {
                continue;
            }

            if (isDangerous(det.label)) {
                if (det.distance < minDangerDist) {
                    minDangerDist = det.distance;
                    closestDanger = det;
                }
            }

            if (det.distance < minObjectDist) {
                minObjectDist = det.distance;
                closestObject = det;
            }
        }

        // Ưu tiên vật nguy hiểm
        return closestDanger != null ? closestDanger : closestObject;
    }

    /**
     * Kiểm tra object có nguy hiểm không
     */
    private boolean isDangerous(String label) {
        if (label == null) return false;
        String lower = label.toLowerCase();
        for (String dangerous : DANGEROUS_OBJECTS) {
            if (lower.contains(dangerous)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Tạo câu cảnh báo
     */
    private String buildWarningMessage(Detection det) {
        String name = getVietnameseName(det.label);
        String distanceStr = String.format("%.1f", det.distance);

        if (det.distance < DANGER_DISTANCE) {
            return "Nguy hiểm! " + name + " rất gần, " + distanceStr + " mét!";
        } else {
            return "Cảnh báo! " + name + " phía trước, " + distanceStr + " mét";
        }
    }

    /**
     * Chuyển tên tiếng Anh sang tiếng Việt
     */
    private String getVietnameseName(String label) {
        if (label == null) return "Vật thể";

        String lower = label.toLowerCase();

        // Phương tiện
        if (lower.contains("car")) return "Xe hơi";
        if (lower.contains("truck")) return "Xe tải";
        if (lower.contains("bus")) return "Xe buýt";
        if (lower.contains("motorcycle")) return "Xe máy";
        if (lower.contains("bicycle")) return "Xe đạp";

        // Người và động vật
        if (lower.contains("person")) return "Người";

        // Đồ vật
        if (lower.contains("tree")) return "Cây";
        if (lower.contains("electric pole")) return "Cột điện";
        if (lower.contains("pedestrian crossing sign")) return "Biển báo";


        // Mặc định
        return label;
    }

    /**
     * Phát âm thanh
     */
    private void speak(String message) {
        if (tts != null && ready) {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null);
            Log.d(TAG, "Speaking: " + message);
        }
    }

    /**
     * Cấu trúc dữ liệu detection đơn giản
     */
    public static class Detection {
        public String label;
        public float distance;

        public Detection(String label, float distance) {
            this.label = label;
            this.distance = distance;
        }

        @Override
        public String toString() {
            return String.format("Detection{label='%s', distance=%.2f}", label, distance);
        }
    }
}