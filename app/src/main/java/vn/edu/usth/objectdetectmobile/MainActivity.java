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

import android.content.Intent;
import android.net.Uri;
import androidx.appcompat.app.AlertDialog;
import android.app.DownloadManager;
import android.os.Environment;
import android.database.Cursor;
import android.os.Build;
import android.content.IntentFilter;

public class MainActivity extends ComponentActivity {
    // ---------------------------------------------------------------------------------------------
    //  Environment mode (indoor / outdoor)
    // ---------------------------------------------------------------------------------------------
    public enum EnvMode {
        INDOOR,
        OUTDOOR
    }
    // -----------------------------------------------------------------------------------------
    //  Model download URLs (GitHub Pages)
    // -----------------------------------------------------------------------------------------
    // Base folder chứa 2 model indoor/outdoor
    private static final String DEPTH_DOWNLOAD_INDOOR_URL =
            "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_hypersim_vits_fp16.onnx";

    private static final String DEPTH_DOWNLOAD_OUTDOOR_URL =
            "https://haidreamer.github.io/models_mobile_app_gp_for_visually_impaired/depth_anything_v2_metric_vkitti_vits_fp16.onnx";

    // Lưu thông tin tải depth model
    private static final String PREF_DEPTH_MODEL_INDOOR_PATH  = "pref_depth_model_indoor_path";
    private static final String PREF_DEPTH_MODEL_OUTDOOR_PATH = "pref_depth_model_outdoor_path";
    private static final String PREF_LAST_DEPTH_DOWNLOAD_ID   = "pref_last_depth_download_id";
    private static final String PREF_LAST_DEPTH_DOWNLOAD_MODE = "pref_last_depth_download_mode";
    // Depth model prefs live in a separate file
    private static final String DEPTH_MODEL_PREFS = "depth_models";
    private SharedPreferences depthModelPrefs;
    private static final String PREF_ENV_MODE = "pref_env_mode";
    private EnvMode envMode = EnvMode.INDOOR;  // default = Indoor
    private SwitchMaterial environmentSwitch;
    // ---------------------------------------------------------------------------------------------
    //  Constants
    // ---------------------------------------------------------------------------------------------
    private static final int REQ = 42;
    private static final String TAG = "MainActivity";

    // Depth throttling / cache
    private static final short DEPTH_INTERVAL_MS = 1500;
    private static final short DEPTH_CACHE_MS = 3000;

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
    private volatile DepthEstimator depthEstimator;     //Marking the field volatile guarantees visibility of the latest reference across threads
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
    private boolean envSwitchInternalChange = false;

    // ---------------------------------------------------------------------------------------------
    //  Lifecycle & entry point
    // ---------------------------------------------------------------------------------------------
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        depthModelPrefs = getSharedPreferences(DEPTH_MODEL_PREFS, MODE_PRIVATE);

        // Single-thread CameraX analyzer
        exec = Executors.newSingleThreadExecutor();
        // Two-thread inference pool: YOLO + depth
        inferenceExec = Executors.newFixedThreadPool(2);
        initViews();
        initPreferencesAndCalibrationKey();

        IntentFilter filter = new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ requires flags: exported vs not exported
            ContextCompat.registerReceiver(
                    this,
                    downloadReceiver,
                    filter,
                    ContextCompat.RECEIVER_NOT_EXPORTED
            );
        } else {
            // Old behavior
            registerReceiver(downloadReceiver, filter);
        }

        initControls();

        // camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQ);
        } else {
            startPipelines();
        }
    }

    // react to user choice permission
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
        try {
            unregisterReceiver(downloadReceiver);
        } catch (Exception ignore) {}
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
        environmentSwitch = findViewById(R.id.switchEnvironment);

        //labels of object detection
        overlay.setLabels(LabelHelper.loadLabels(this, "labels.txt"));
    }

    private void initPreferencesAndCalibrationKey() {
        prefs = DepthCalibrationHelper.getPrefs(this);

        // Calibration
        calibrationPrefKey = DepthCalibrationHelper.buildCalibrationKey(this);
        calibrationScale = DepthCalibrationHelper.loadSavedCalibrationScale(
                prefs,
                calibrationPrefKey,
                DepthEstimator.getUserScale()
        );
        DepthEstimator.setUserScale(calibrationScale);

        // Environment mode (load từ prefs, default = INDOOR)
        String savedEnv = prefs.getString(PREF_ENV_MODE, EnvMode.INDOOR.name());
        try {
            envMode = EnvMode.valueOf(savedEnv);
        } catch (IllegalArgumentException e) {
            envMode = EnvMode.INDOOR;
        }

        // Sync với UI switch (ON = OUTDOOR, OFF = INDOOR)
        if (environmentSwitch != null) {
            environmentSwitch.setChecked(envMode == EnvMode.OUTDOOR);
        }
    }


    private void initControls() {
        initRealtimeSwitch();
        initDetectOnceButton();
        initDualShotButton();
        initBlurSwitch();
        initStereoSwitch();
        initEnvironmentSwitch();
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

    private void initEnvironmentSwitch() {
        // If the app itself changed the switch state (via setChecked()), set flag to avoid triggering the logic again
        if (environmentSwitch == null) return;

        // Sync lại trạng thái hiện tại
        environmentSwitch.setChecked(envMode == EnvMode.OUTDOOR);

        environmentSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (envSwitchInternalChange) return;

            // User is trying to switch to this mode
            EnvMode targetMode = isChecked ? EnvMode.OUTDOOR : EnvMode.INDOOR;

            // Nếu chưa có model cho mode này thì không cho switch, gợi ý download
            if (!DepthEstimator.isModelAvailable(this, targetMode)) {
                envSwitchInternalChange = true;
                buttonView.setChecked(!isChecked);  // revert về mode cũ
                envSwitchInternalChange = false;

                showMissingDepthModelDialog(targetMode);
                return;
            }

            // Model đã có -> switch thật sự
            switchEnvironment(targetMode);
        });
    }

    private void switchEnvironment(EnvMode newMode) {
        envMode = newMode;

        // Lưu vào prefs
        prefs.edit().putString(PREF_ENV_MODE, envMode.name()).apply();

        Toast.makeText(
                this,
                "Environment: " + (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                Toast.LENGTH_SHORT
        ).show();

        // Reload lại detector/depth cho mode mới
        reloadPipelinesForEnvChange();
    }


    private void reloadPipelinesForEnvChange() {
        // Pause realtime so we don't process frames while reloading depth
        realtimeEnabled = false;

        runOnUiThread(() -> {
            Log.i(TAG, "Reloading depth pipeline for envMode = " + envMode);

            // 1) Check if we have ANY model for this mode (asset or downloaded)
            boolean depthModelOk = DepthEstimator.isModelAvailable(this, envMode);
            if (!depthModelOk) {
                // No model yet -> show download dialog for this mode
                showMissingDepthModelDialog(envMode);

                // Keep depthEstimator = null, YOLO-only mode
                depthEstimator = null;
                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }

                // Re-enable realtime (but without depth)
                realtimeEnabled = true;
                return;
            }

            // 2) We DO have a model (asset or downloaded) -> try to create DepthEstimator
            try {
                DepthEstimator newDepth = new DepthEstimator(this, envMode);
                depthEstimator = newDepth;

                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }

                Toast.makeText(
                        this,
                        "Depth model loaded for " +
                                (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                        Toast.LENGTH_SHORT
                ).show();

            } catch (Throwable e) {
                Log.w(TAG, "Depth estimator re-init failed", e);
                depthEstimator = null;
                synchronized (depthState) {
                    depthState.lastDepthMap = null;
                    depthState.lastDepthMillis = 0L;
                    depthState.lastDepthCacheTime = 0L;
                }
                Toast.makeText(
                        this,
                        "Failed to init depth for " +
                                (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                        Toast.LENGTH_LONG
                ).show();
            }

            // Re-enable realtime (with or without depth depending on success)
            realtimeEnabled = true;
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

        // --- NEW: kiểm tra depth model có trong assets hay chưa ---
        boolean depthModelOk = DepthEstimator.isModelAvailable(this, envMode);
        if (!depthModelOk) {
            // Không có model -> thông báo & gợi ý mở link download
            showMissingDepthModelDialog(envMode);
            // Không tạo depthEstimator, app vẫn chạy YOLO-only
            depthEstimator = null;
            depthState.lastDepthMap = null;
            depthState.lastDepthMillis = 0L;
            depthState.lastDepthCacheTime = 0L;
            stereoProcessor = null;
            updateStereoSwitchAvailability(false);
            return;
        }

        try {
            depthEstimator = new DepthEstimator(this, envMode);
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

    private void showMissingDepthModelDialog(EnvMode targetMode) {
        String modeLabel = (targetMode == EnvMode.OUTDOOR) ? "Outdoor" : "Indoor";

        String downloadUrl = (targetMode == EnvMode.OUTDOOR)
                ? DEPTH_DOWNLOAD_OUTDOOR_URL
                : DEPTH_DOWNLOAD_INDOOR_URL;

        new AlertDialog.Builder(this)
                .setTitle("Depth model missing")
                .setMessage(
                        "Depth model for " + modeLabel + " mode is not available inside the app.\n\n" +
                                "Do you want to download it now?"
                )
                .setPositiveButton("Download", (dialog, which) -> {
                    startDepthModelDownload(downloadUrl, targetMode);
                })
                .setNegativeButton("Cancel", (dialog, which) -> {
                    Toast.makeText(this,
                            "Running without depth estimation",
                            Toast.LENGTH_SHORT).show();
                })
                .setCancelable(true)
                .show();
    }


    private void startDepthModelDownload(String url, EnvMode mode) {
        DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
        if (dm == null) {
            Toast.makeText(this, "DownloadManager not available", Toast.LENGTH_LONG).show();
            return;
        }

        java.io.File outFile = getDepthModelFileForMode(mode);
        if (outFile == null) {
            Toast.makeText(this, "No external files dir for downloads", Toast.LENGTH_LONG).show();
            return;
        }

        String fileName = outFile.getName();

        DownloadManager.Request req = new DownloadManager.Request(Uri.parse(url))
                .setTitle("Downloading " + fileName)
                .setDescription("Depth model for " + (mode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"))
                .setNotificationVisibility(
                        DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                // We know exactly where the file will be:
                .setDestinationInExternalFilesDir(
                        this,
                        Environment.DIRECTORY_DOWNLOADS,
                        fileName
                );

        long id = dm.enqueue(req);

        String keyPath = (mode == EnvMode.OUTDOOR)
                ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                : PREF_DEPTH_MODEL_INDOOR_PATH;

        // Save: which download, which mode, and where we expect the file to be
        depthModelPrefs.edit()
                .putLong(PREF_LAST_DEPTH_DOWNLOAD_ID, id)
                .putString(PREF_LAST_DEPTH_DOWNLOAD_MODE, mode.name())
                .putString(keyPath, outFile.getAbsolutePath())
                .apply();

        Toast.makeText(this, "Downloading depth model...", Toast.LENGTH_SHORT).show();
    }

    private final android.content.BroadcastReceiver downloadReceiver =
            new android.content.BroadcastReceiver() {
                @Override
                public void onReceive(android.content.Context ctx, Intent intent) {
                    if (!DownloadManager.ACTION_DOWNLOAD_COMPLETE.equals(intent.getAction())) {
                        return;
                    }

                    long id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1L);
                    long expectedId = depthModelPrefs.getLong(PREF_LAST_DEPTH_DOWNLOAD_ID, -1L);
                    if (id != expectedId) return; // Not our download

                    DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
                    if (dm == null) return;

                    DownloadManager.Query q = new DownloadManager.Query().setFilterById(id);
                    try (Cursor c = dm.query(q)) {
                        if (c != null && c.moveToFirst()) {
                            int statusIdx = c.getColumnIndex(DownloadManager.COLUMN_STATUS);
                            if (statusIdx == -1) return;

                            int status = c.getInt(statusIdx);
                            if (status == DownloadManager.STATUS_SUCCESSFUL) {

                                // Which mode was this download for?
                                String modeName = depthModelPrefs.getString(
                                        PREF_LAST_DEPTH_DOWNLOAD_MODE,
                                        EnvMode.INDOOR.name()
                                );
                                final EnvMode mode = EnvMode.valueOf(modeName);

                                String keyPath = (mode == EnvMode.OUTDOOR)
                                        ? PREF_DEPTH_MODEL_OUTDOOR_PATH
                                        : PREF_DEPTH_MODEL_INDOOR_PATH;

                                final String path = depthModelPrefs.getString(keyPath, null);

                                if (path != null && new java.io.File(path).exists()) {
                                    Log.i(TAG, "Depth model downloaded OK for mode=" + mode + " at " + path);

                                    runOnUiThread(() -> {
                                        try {
                                            // Make sure envMode matches the model we just downloaded
                                            if (envMode == mode) {
                                                reloadPipelinesForEnvChange();
                                            }

                                            Toast.makeText(
                                                    MainActivity.this,
                                                    "Depth model loaded for " +
                                                            (envMode == EnvMode.OUTDOOR ? "Outdoor" : "Indoor"),
                                                    Toast.LENGTH_SHORT
                                            ).show();

                                        } catch (Throwable e) {
                                            Log.w(TAG, "Failed to init depth after download", e);
                                            Toast.makeText(
                                                    MainActivity.this,
                                                    "Depth model downloaded but failed to init",
                                                    Toast.LENGTH_LONG
                                            ).show();
                                        }
                                    });

                                } else {
                                    Log.w(TAG, "Depth model download reported success but file missing at " + path);
                                    Toast.makeText(MainActivity.this,
                                            "Depth model file missing after download",
                                            Toast.LENGTH_LONG).show();
                                }

                            } else {
                                Toast.makeText(MainActivity.this,
                                        "Depth model download failed",
                                        Toast.LENGTH_LONG).show();
                            }
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error checking download status", e);
                    }
                }
            };



    private java.io.File getDepthModelFileForMode(EnvMode mode) {
        String fileName = (mode == EnvMode.OUTDOOR)
                ? "depth_anything_v2_metric_vkitti_vits_fp16.onnx"
                : "depth_anything_v2_metric_hypersim_vits_fp16.onnx";

        java.io.File dir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
        if (dir == null) return null;
        return new java.io.File(dir, fileName);
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
