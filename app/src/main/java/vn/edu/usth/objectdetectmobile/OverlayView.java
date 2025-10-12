package vn.edu.usth.objectdetectmobile;

import android.content.Context;
import android.graphics.*;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.*;

public class OverlayView extends View {
    private final Paint box = new Paint();
    private final Paint text = new Paint();
    private List<ObjectDetector.Detection> dets = new ArrayList<>();
    private String[] labels = new String[0];

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        box.setStyle(Paint.Style.STROKE);
        box.setStrokeWidth(4f);
        box.setAntiAlias(true);
        text.setColor(Color.WHITE);
        text.setTextSize(36f);
        text.setAntiAlias(true);
    }

    public void setLabels(String[] labels) { this.labels = labels; }

    public void setDetections(List<ObjectDetector.Detection> dets) {
        this.dets = dets != null ? dets : new ArrayList<>();
        invalidate();
    }

    @Override protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        for (ObjectDetector.Detection d : dets) {
            box.setColor(Color.GREEN);
            canvas.drawRect(d.x1, d.y1, d.x2, d.y2, box);
            String lab = (d.cls >= 0 && d.cls < labels.length) ? labels[d.cls] : ("cls " + d.cls);
            String t = lab + String.format(" %.2f", d.score);
            canvas.drawText(t, d.x1 + 6, Math.max(0, d.y1 - 8), text);
        }
    }
}
