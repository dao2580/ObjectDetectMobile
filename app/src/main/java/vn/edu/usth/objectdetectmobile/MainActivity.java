package vn.edu.usth.objectdetectmobile;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

// Non-deprecated resolution selector API
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ai.onnxruntime.OrtException;

public class MainActivity extends ComponentActivity {

    // ---------------------------------------------------------------------------------------------
    //  Constants
    // ---------------------------------------------------------------------------------------------
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    // Depth throttling / cache
    private static final long DEPTH_INTERVAL_MS = 1500L;
    private static final long DEPTH_CACHE_MS = 3000L;

    // Input blur
    private static final boolean ENABLE_INPUT_BLUR = true;
    private static final int BLUR_RADIUS = 1; // 1 => kernel 3x3

    // ---------------------------------------------------------------------------------------------
    //  UI views
    // ---------------------------------------------------------------------------------------------
    private PreviewView previewView;
    private OverlayView overlay;
    private SwitchMaterial realtimeSwitch;
    private SwitchMaterial blurSwitch;
    private SwitchMaterial stereoSwitch;
    private MaterialButton detectOnceButton;
    private MaterialButton settingsButton;
    private MaterialButton dualShotButton;
    private MaterialButton flipCameraButton;
    private View controlPanel;
    private TextView depthModeText;
    private SeekBar calibrationSeek;
    private TextView calibrationValue;
    private SeekBar zoomSeek;
    private TextView zoomValue;

    // ---------------------------------------------------------------------------------------------
    //  Core components
    // ---------------------------------------------------------------------------------------------
    private ObjectDetector detector;
    private DepthEstimator depthEstimator;
    private StereoDepthProcessor stereoProcessor;
    private ProcessCameraProvider cameraProvider;
    private Camera currentCamera;

    // CameraX analysis executor (single thread)
    private ExecutorService exec;
    // Inference executor (YOLO + depth in parallel)
    private ExecutorService inferenceExec;

    // ---------------------------------------------------------------------------------------------
    //  Depth & stereo state
    // ---------------------------------------------------------------------------------------------
    // Reuse your existing helper state holder
    final DepthPipelineHelper.DepthState depthState =
            new DepthPipelineHelper.DepthState();

    private volatile boolean stereoFusionEnabled = false;
    private boolean stereoPipelineAvailable = false;
    private volatile boolean sequentialStereoRunning = false;
    private List<CameraUtils.CamInfo> backCameraInfos = new ArrayList<>();

    // ---------------------------------------------------------------------------------------------
    //  Detection / realtime state
    // ---------------------------------------------------------------------------------------------
    private volatile boolean realtimeEnabled = true;
    private volatile boolean blurEnabled = ENABLE_INPUT_BLUR;
    private volatile boolean singleShotRequested = false;
    private volatile boolean singleShotRunning = false;

    // ---------------------------------------------------------------------------------------------
    //  Calibration & prefs
    // ---------------------------------------------------------------------------------------------
    private SharedPreferences prefs;
    private String calibrationPrefKey;
    private float calibrationScale = 1f;

    // ---------------------------------------------------------------------------------------------
    //  Zoom & camera facing
    // ---------------------------------------------------------------------------------------------
    private volatile int lensFacing = CameraSelector.LENS_FACING_BACK;
    private volatile float zoomMinRatio = 1f;
    private volatile float zoomMaxRatio = 1f;
    private boolean stereoSwitchInternalChange = false;

    // ---------------------------------------------------------------------------------------------
    //  Lifecycle & entry point
    // ---------------------------------------------------------------------------------------------
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Single-thread CameraX analyzer
        exec = Executors.newSingleThreadExecutor();
        // Two-thread inference pool: YOLO + depth
        inferenceExec = Executors.newFixedThreadPool(2);

        initViews();
        initPreferencesAndCalibrationKey();
        initControls();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            startPipelines();
        }
    }

    @Override
    public void onRequestPermissionsResult(int c, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(c, p, r);
        if (c == REQ && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) {
            startPipelines();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (exec != null) exec.shutdownNow();
        if (inferenceExec != null) inferenceExec.shutdownNow();
        if (detector != null) {
            try {
                detector.close();
            } catch (Exception e) {
                Log.e(TAG, "Detector close failed", e);
            }
        }
        if (depthEstimator != null) {
            try {
                depthEstimator.close();
            } catch (Exception e) {
                Log.e(TAG, "DepthEstimator close failed", e);
            }
        }
        stereoProcessor = null;
        // Clear depth cache state
        depthState.lastDepthMap = null;
        depthState.lastDepthMillis = 0L;
        depthState.lastDepthCacheTime = 0L;
    }

    // ---------------------------------------------------------------------------------------------
    //  UI init & listeners
    // ---------------------------------------------------------------------------------------------
    private void initViews() {
        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        realtimeSwitch = findViewById(R.id.switchRealtime);
        blurSwitch = findViewById(R.id.switchBlur);
        stereoSwitch = findViewById(R.id.switchStereo);
        detectOnceButton = findViewById(R.id.buttonDetectOnce);
        dualShotButton = findViewById(R.id.buttonDualShot);
        settingsButton = findViewById(R.id.buttonToggleSettings);
        flipCameraButton = findViewById(R.id.buttonFlipCamera);
        controlPanel = findViewById(R.id.controlPanel);
        depthModeText = findViewById(R.id.textDepthMode);
        calibrationSeek = findViewById(R.id.seekCalibration);
        calibrationValue = findViewById(R.id.textCalibrationValue);
        zoomSeek = findViewById(R.id.seekZoom);
        zoomValue = findViewById(R.id.textZoomValue);

        overlay.setLabels(LabelHelper.loadLabels(this, "labels.txt"));
    }

    private void initPreferencesAndCalibrationKey() {
        prefs = DepthCalibrationHelper.getPrefs(this);
        calibrationPrefKey = DepthCalibrationHelper.buildCalibrationKey(this);
        calibrationScale = DepthCalibrationHelper.loadSavedCalibrationScale(
                prefs,
                calibrationPrefKey,
                DepthEstimator.getUserScale()
        );
        DepthEstimator.setUserScale(calibrationScale);
    }

    private void initControls() {
        initRealtimeSwitch();
        initDetectOnceButton();
        initDualShotButton();
        initBlurSwitch();
        initStereoSwitch();
        initSettingsButton();
        initFlipCameraButton();
        setupCalibrationControls();
        setupZoomControls();
        updateDepthModeLabel();
    }

    private void initRealtimeSwitch() {
        if (realtimeSwitch == null) return;
        realtimeSwitch.setChecked(true);
        realtimeSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            realtimeEnabled = isChecked;
            if (detectOnceButton != null) {
                detectOnceButton.setVisibility(isChecked ? View.GONE : View.VISIBLE);
                detectOnceButton.setEnabled(true);
            }
            if (isChecked) {
                singleShotRequested = false;
            }
        });
    }

    private void initDetectOnceButton() {
        if (detectOnceButton == null) return;
        detectOnceButton.setVisibility(View.GONE);
        detectOnceButton.setOnClickListener(v -> {
            if (singleShotRunning) return;
            singleShotRequested = true;
            detectOnceButton.setEnabled(false);
        });
    }

    private void initDualShotButton() {
        if (dualShotButton == null) return;
        dualShotButton.setOnClickListener(v -> {
            if (sequentialStereoRunning) return;
            handleSequentialDualShot();
        });
    }

    private void initBlurSwitch() {
        if (blurSwitch == null) return;
        blurSwitch.setChecked(ENABLE_INPUT_BLUR);
        blurSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> blurEnabled = isChecked);
    }

    private void initStereoSwitch() {
        if (stereoSwitch == null) return;
        stereoSwitch.setEnabled(false);
        stereoSwitch.setText(R.string.stereo_toggle_disabled_hint);
        stereoSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (stereoSwitchInternalChange) return;
            if (!stereoPipelineAvailable) {
                if (isChecked) {
                    Toast.makeText(this, R.string.stereo_toggle_disabled_hint,
                            Toast.LENGTH_SHORT).show();
                }
                stereoSwitchInternalChange = true;
                buttonView.setChecked(false);
                stereoSwitchInternalChange = false;
                stereoFusionEnabled = false;
                updateDepthModeLabel();
                return;
            }
            stereoFusionEnabled = isChecked;
            updateDepthModeLabel();
        });
    }

    private void initSettingsButton() {
        if (settingsButton == null) {
            applySettingsVisibility(true);
            return;
        }
        settingsButton.setOnClickListener(v -> toggleSettingsPanel());
        applySettingsVisibility(false);
    }

    private void initFlipCameraButton() {
        if (flipCameraButton == null) return;
        flipCameraButton.setOnClickListener(v -> switchCameraFacing());
    }

    private void toggleSettingsPanel() {
        boolean visible = controlPanel != null
                && controlPanel.getVisibility() != View.VISIBLE;
        applySettingsVisibility(visible);
    }

    private void applySettingsVisibility(boolean visible) {
        if (controlPanel != null) {
            controlPanel.setVisibility(visible ? View.VISIBLE : View.GONE);
        }
        if (settingsButton != null) {
            settingsButton.setIconResource(
                    visible ? android.R.drawable.ic_menu_close_clear_cancel
                            : android.R.drawable.ic_menu_manage
            );
            settingsButton.setContentDescription(
                    getString(visible ? R.string.settings_hide : R.string.settings_show)
            );
        }
    }

    // ---------------------------------------------------------------------------------------------
    //  Pipelines startup (detector + depth + camera)
    // ---------------------------------------------------------------------------------------------
    private void startPipelines() {
        initDetectorAndDepth();
        initCameraProvider();
    }

    private void initDetectorAndDepth() {
        try {
            detector = new ObjectDetector(this);
        } catch (Throwable e) {
            Log.e(TAG, "Detector init failed", e);
            Toast.makeText(this, "Detector load failed: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        try {
            depthEstimator = new DepthEstimator(this);
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
        } catch (Throwable e) {
            Log.w(TAG, "Depth estimator disabled", e);
            depthEstimator = null;
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
        }
        stereoProcessor = null;
        updateStereoSwitchAvailability(false);
    }

    private void initCameraProvider() {
        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider =
                        ProcessCameraProvider.getInstance(this).get();
                cameraProvider = provider;
                provider.unbindAll();
                backCameraInfos = CameraUtils.cacheBackCameraInfos(provider);
                bindCameraUseCases();
            } catch (Throwable e) {
                Log.e(TAG, "Camera bind error", e);
                Toast.makeText(this, "Camera error: " + e.getMessage(),
                        Toast.LENGTH_LONG).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // ---------------------------------------------------------------------------------------------
    //  Camera pipeline (realtime analysis)
    // ---------------------------------------------------------------------------------------------
    private void bindCameraUseCases() {
        if (cameraProvider == null) return;
        try {
            cameraProvider.unbindAll();

            Preview preview = new Preview.Builder()
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setAspectRatioStrategy(
                                            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                    )
                                    .build()
                    )
                    .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            ImageAnalysis analysis = new ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setResolutionSelector(
                            new ResolutionSelector.Builder()
                                    .setAspectRatioStrategy(
                                            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                                    )
                                    .setResolutionStrategy(
                                            new ResolutionStrategy(
                                                    new Size(360, 360),
                                                    ResolutionStrategy
                                                            .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                            )
                                    )
                                    .build()
                    )
                    .build();

            analysis.setAnalyzer(exec, this::analyzeFrame);

            CameraSelector selector =
                    CameraUtils.buildBackSelector(cameraProvider, lensFacing);

            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) this, selector, preview, analysis);
            currentCamera = camera;

            observeZoom(camera);
            setupStereoProcessorForCurrentCamera(camera);

        } catch (Throwable e) {
            Log.e(TAG, "Camera bind error", e);
            Toast.makeText(this, "Camera error: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Runs depth estimation if the interval has passed, otherwise returns cached depth.
     * Thread-safe; can be called from background threads.
     */
    private DepthEstimator.DepthMap maybeRunDepthSync(
            int[] argb,
            int width,
            int height,
            long nowMs
    ) {
        if (depthEstimator == null) return null;

        synchronized (depthState) {
            boolean hasDepth = (depthState.lastDepthMap != null);
            boolean tooSoon = (nowMs - depthState.lastDepthMillis) < DEPTH_INTERVAL_MS;
            boolean cacheValid = hasDepth &&
                    (nowMs - depthState.lastDepthCacheTime) <= DEPTH_CACHE_MS;

            if (tooSoon && cacheValid) {
                depthState.lastDepthCacheTime = nowMs;
                return depthState.lastDepthMap;
            }
        }

        try {
            DepthEstimator.DepthMap map =
                    depthEstimator.estimate(argb, width, height);

            synchronized (depthState) {
                depthState.lastDepthMap = map;
                depthState.lastDepthMillis = nowMs;
                depthState.lastDepthCacheTime = nowMs;
            }
            return map;
        } catch (Exception e) {
            Log.e(TAG, "Depth estimation failed", e);
            return null;
        }
    }

    private void analyzeFrame(ImageProxy image) {
        boolean singleShotFrame = false;
        try {
            boolean shouldProcess = realtimeEnabled;
            if (!shouldProcess && singleShotRequested && !singleShotRunning) {
                singleShotRequested = false;
                singleShotRunning = true;
                singleShotFrame = true;
                shouldProcess = true;
                if (stereoFusionEnabled && !stereoPipelineAvailable) {
                    singleShotFrame = false;
                    handleSequentialDualShot();
                    return;
                }
            }
            if (!shouldProcess) return;

            // Basic frame info
            int frameW = image.getWidth();
            int frameH = image.getHeight();
            int rotation = image.getImageInfo().getRotationDegrees();

            // YUV → ARGB (+ rotation)
            int[] argb = Yuv.toArgb(image);
            if (rotation != 0) {
                argb = Yuv.rotate(argb, frameW, frameH, rotation);
                if (rotation == 90 || rotation == 270) {
                    int tmp = frameW;
                    frameW = frameH;
                    frameH = tmp;
                }
            }

            if (stereoProcessor != null) {
                stereoProcessor.setReferenceSize(frameW, frameH);
            }

            int[] detectorInput = (blurEnabled && BLUR_RADIUS > 0)
                    ? ImageUtils.boxBlur(argb, frameW, frameH, BLUR_RADIUS)
                    : argb;

            final long nowMs = SystemClock.elapsedRealtime();

            // Run YOLO + depth in parallel on inferenceExec
            int finalFrameW1 = frameW;
            int finalFrameH1 = frameH;
            Future<List<ObjectDetector.Detection>> detFuture =
                    inferenceExec.submit(() -> {
                        try {
                            return detector.detect(detectorInput, finalFrameW1, finalFrameH1);
                        } catch (OrtException e) {
                            Log.e(TAG, "detect failed", e);
                            return null;
                        } catch (Throwable t) {
                            Log.e(TAG, "detect crashed", t);
                            return null;
                        }
                    });

            Future<DepthEstimator.DepthMap> depthFuture = null;
            if (depthEstimator != null) {
                int[] finalArgb = argb;
                int finalFrameW = frameW;
                int finalFrameH = frameH;
                depthFuture = inferenceExec.submit(() ->
                        maybeRunDepthSync(finalArgb, finalFrameW, finalFrameH, nowMs)
                );
            }

            // Wait for results
            List<ObjectDetector.Detection> dets = detFuture.get();

            DepthEstimator.DepthMap depthMap = null;
            if (depthFuture != null) {
                depthMap = depthFuture.get();
            }

            if (depthMap != null && dets != null) {
                dets = depthEstimator.attachDepth(dets, depthMap);
            }

            if (stereoFusionEnabled && stereoProcessor != null
                    && depthMap != null && dets != null) {
                dets = stereoProcessor.fuseDepth(depthMap, dets, frameW, frameH);
            }

            int finalW = frameW;
            int finalH = frameH;
            List<ObjectDetector.Detection> finalDets = dets;
            runOnUiThread(() -> overlay.setDetections(finalDets, finalW, finalH));

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "analyzeFrame interrupted", e);
        } catch (Throwable t) {
            Log.e(TAG, "analyzer crash", t);
        } finally {
            image.close();
            if (singleShotFrame) {
                singleShotRunning = false;
                runOnUiThread(() -> {
                    if (detectOnceButton != null) detectOnceButton.setEnabled(true);
                });
            }
        }
    }

    @SuppressLint("RestrictedApi")
    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private void setupStereoProcessorForCurrentCamera(Camera camera) {
        try {
            if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                stereoProcessor = new StereoDepthProcessor(
                        this,
                        Camera2CameraInfo.extractCameraCharacteristics(
                                camera.getCameraInfo())
                );
                updateStereoSwitchAvailability(true);
            } else {
                stereoProcessor = null;
                updateStereoSwitchAvailability(false);
            }
        } catch (Throwable processorErr) {
            Log.w(TAG, "Stereo processor init failed", processorErr);
            stereoProcessor = null;
            updateStereoSwitchAvailability(false);
        }
    }

    // ---------------------------------------------------------------------------------------------
    //  Stereo single-shot pipeline
    // ---------------------------------------------------------------------------------------------
    private void handleSequentialDualShot() {
        if (cameraProvider == null) {
            singleShotRunning = false;
            return;
        }
        if (lensFacing != CameraSelector.LENS_FACING_BACK) {
            sequentialStereoRunning = false;
            singleShotRunning = false;
            runOnUiThread(() -> {
                if (detectOnceButton != null) detectOnceButton.setEnabled(true);
                Toast.makeText(this, R.string.stereo_toggle_disabled_hint,
                        Toast.LENGTH_SHORT).show();
            });
            return;
        }
        if (sequentialStereoRunning) return;

        singleShotRunning = true;
        sequentialStereoRunning = true;

        runOnUiThread(() ->
                Toast.makeText(this, R.string.sequential_dual_shot,
                        Toast.LENGTH_SHORT).show());

        if (backCameraInfos == null || backCameraInfos.isEmpty()) {
            backCameraInfos = CameraUtils.cacheBackCameraInfos(cameraProvider);
        }

        SequentialStereoHelper.runSequentialStereoShot(
                this,
                cameraProvider,
                previewView,
                exec,
                backCameraInfos,
                detector,
                depthEstimator,
                blurEnabled,
                BLUR_RADIUS,
                stereoFusionEnabled,
                stereoProcessor,
                new SequentialStereoHelper.Callback() {
                    @Override
                    public void onResult(SequentialStereoHelper.Result result) {
                        if (result == null || result.detections == null) return;
                        overlay.setDetections(
                                result.detections,
                                result.width,
                                result.height
                        );
                    }

                    @Override
                    public void onError(Throwable t) {
                        Log.e(TAG, "Sequential stereo single shot failed", t);
                    }

                    @Override
                    public void onFinished() {
                        sequentialStereoRunning = false;
                        singleShotRunning = false;
                        if (detectOnceButton != null) {
                            detectOnceButton.setEnabled(true);
                        }
                        bindCameraUseCases();
                    }
                }
        );
    }

    // ---------------------------------------------------------------------------------------------
    //  Depth calibration & UI
    // ---------------------------------------------------------------------------------------------
    private void setupCalibrationControls() {
        DepthCalibrationHelper.bindCalibrationSeekBar(
                calibrationSeek,
                calibrationValue,
                calibrationScale,
                prefs,
                calibrationPrefKey
        );
    }

    private void updateDepthModeLabel() {
        if (depthModeText == null) return;
        boolean stereoActive = stereoFusionEnabled
                && stereoPipelineAvailable
                && stereoProcessor != null;
        depthModeText.setText(getString(
                stereoActive ? R.string.depth_mode_stereo : R.string.depth_mode_mono));
    }

    private void updateStereoSwitchAvailability(boolean available) {
        stereoPipelineAvailable = available;
        if (stereoSwitch == null) return;
        runOnUiThread(() -> {
            stereoSwitchInternalChange = true;
            if (!available) {
                stereoSwitch.setChecked(false);
                stereoSwitch.setEnabled(false);
                stereoSwitch.setText(R.string.stereo_toggle_disabled_hint);
                stereoFusionEnabled = false;
            } else {
                stereoSwitch.setText(R.string.stereo_toggle);
                stereoSwitch.setEnabled(true);
            }
            stereoSwitchInternalChange = false;
            updateDepthModeLabel();
        });
    }

    // ---------------------------------------------------------------------------------------------
    //  Zoom & camera utils
    // ---------------------------------------------------------------------------------------------
    private void setupZoomControls() {
        if (zoomSeek == null || zoomValue == null) return;
        ZoomHelper.setupZoomSeekBar(
                zoomSeek,
                zoomValue,
                new ZoomHelper.ZoomControl() {
                    @Override
                    public float getMinZoomRatio() {
                        return zoomMinRatio;
                    }

                    @Override
                    public float getMaxZoomRatio() {
                        return zoomMaxRatio;
                    }

                    @Override
                    public void setZoomRatio(float ratio) {
                        if (currentCamera != null) {
                            currentCamera.getCameraControl().setZoomRatio(ratio);
                        }
                    }
                }
        );
    }

    private void observeZoom(Camera camera) {
        if (zoomSeek == null || zoomValue == null) return;
        camera.getCameraInfo().getZoomState().observe(this, state -> {
            if (state == null) return;
            zoomMinRatio = state.getMinZoomRatio();
            zoomMaxRatio = state.getMaxZoomRatio();
            int progress = ZoomHelper.zoomRatioToProgress(
                    state.getZoomRatio(),
                    zoomMinRatio,
                    zoomMaxRatio
            );
            zoomSeek.setProgress(progress);
            zoomValue.setText(getString(R.string.zoom_value, state.getZoomRatio()));
        });
    }

    private void switchCameraFacing() {
        lensFacing = (lensFacing == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT
                : CameraSelector.LENS_FACING_BACK;
        updateStereoSwitchAvailability(false);
        bindCameraUseCases();
        updateDepthModeLabel();
    }
}
