package com.aicandy.objectdetection.yolo;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;

public class DetectionOverlay extends View {
    private final static int LABEL_PADDING = 10;
    private final static int LABEL_Y = 35;

    private Paint boxPaint;
    private Paint textPaint;
    private Paint labelPaint;
    private ArrayList<DetectionResult> detectionResults;

    public DetectionOverlay(Context context) {
        super(context);
        init();
    }

    public DetectionOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint = new Paint();
        boxPaint.setColor(Color.YELLOW);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(28);
        textPaint.setStyle(Paint.Style.FILL);

        labelPaint = new Paint();
        labelPaint.setColor(Color.MAGENTA);
        labelPaint.setStyle(Paint.Style.FILL);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (detectionResults == null) return;

        for (DetectionResult detection : detectionResults) {
            canvas.drawRect(detection.boundingBox, boxPaint);
            String labelText = ImageProcessor.CLASSES[detection.classId];
            Rect textBounds = new Rect();
            textPaint.getTextBounds(labelText, 0, labelText.length(), textBounds);
            float textWidth = textBounds.width();
            float textHeight = textBounds.height();

            float labelWidth = textWidth + 2 * LABEL_PADDING;
            float labelHeight = textHeight + 2 * LABEL_PADDING;

            RectF labelRect = new RectF(
                    detection.boundingBox.left,
                    detection.boundingBox.top,
                    detection.boundingBox.left + labelWidth,
                    detection.boundingBox.top + labelHeight
            );

            canvas.drawRect(labelRect, labelPaint);
            canvas.drawText(labelText,
                    labelRect.left + LABEL_PADDING,
                    labelRect.top + LABEL_PADDING + textHeight,
                    textPaint);
        }
    }

    public void setDetections(ArrayList<DetectionResult> results) {
        detectionResults = results;
        invalidate();
    }
}