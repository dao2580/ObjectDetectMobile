package vn.edu.usth.objectdetectmobile;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

// NEW imports for the non-deprecated API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtException;

public class MainActivity extends ComponentActivity {
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    private PreviewView previewView;
    private OverlayView overlay;
    private ObjectDetector detector;
    private ExecutorService exec;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        overlay.setLabels(loadLabels());

        exec = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            start();
        }
    }

    private void start() {
        try {
            // Your ObjectDetector should load the ONNX model from a FILE PATH (not byte[])
            // to avoid OOM with large models.
            detector = new ObjectDetector(this);
        } catch (Throwable e) {
            Log.e(TAG, "ORT init failed", e);
            Toast.makeText(this, "Model load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return; // don't start CameraX if detector failed
        }

        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(this).get();
                provider.unbindAll();

                // Build Preview with ResolutionSelector (non-deprecated)
                Preview preview =
                        new Preview.Builder()
                                .setResolutionSelector(
                                        new ResolutionSelector.Builder()
                                                .setAspectRatioStrategy(
                                                        AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                                )
                                                .build()
                                )
                                .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Build ImageAnalysis with ResolutionSelector (prefer ~640 square input, but let CameraX choose closest)
                ImageAnalysis analysis =
                        new ImageAnalysis.Builder()
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .setResolutionSelector(
                                        new ResolutionSelector.Builder()
                                                .setAspectRatioStrategy(
                                                        AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                                )
                                                .setResolutionStrategy(
                                                        new ResolutionStrategy(
                                                                new Size(640, 640),
                                                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                                        )
                                                )
                                                .build()
                                )
                                .build();

                analysis.setAnalyzer(exec, image -> {
                    try {
                        // Convert YUV_420_888 -> ARGB (your helper)
                        int[] argb = Yuv.toArgb(image);

                        // Run detection off the UI thread
                        List<ObjectDetector.Detection> dets =
                                detector.detect(argb, image.getWidth(), image.getHeight());

                        // Draw on overlay on UI thread
                        runOnUiThread(() -> overlay.setDetections(dets));
                    } catch (OrtException t) {
                        Log.e(TAG, "detect failed", t);
                    } catch (Throwable t) {
                        Log.e(TAG, "analyzer crash", t);
                    } finally {
                        // ALWAYS close frame or pipeline can stall/crash
                        image.close();
                    }
                });

                CameraSelector selector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                provider.bindToLifecycle(this, selector, preview, analysis);
            } catch (Throwable e) {
                Log.e(TAG, "Camera bind error", e);
                Toast.makeText(this, "Camera error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private String[] loadLabels() {
        List<String> list = new ArrayList<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line;
            while ((line = br.readLine()) != null) list.add(line);
        } catch (Exception ignored) {
        } finally {
            try { if (br != null) br.close(); } catch (Exception ignored) {}
        }
        return list.toArray(new String[0]);
    }

    @Override public void onRequestPermissionsResult(int c, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(c,p,r);
        if (c == REQ && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) start();
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (exec != null) exec.shutdownNow();
        if (detector != null) {
            try {
                detector.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
