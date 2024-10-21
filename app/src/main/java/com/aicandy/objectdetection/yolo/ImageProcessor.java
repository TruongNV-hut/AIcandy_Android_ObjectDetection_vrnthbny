package com.aicandy.objectdetection.yolo;

import android.graphics.Rect;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

class DetectionResult {
    int classId;
    Float confidence;
    Rect boundingBox;

    public DetectionResult(int cls, Float output, Rect rect) {
        this.classId = cls;
        this.confidence = output;
        this.boundingBox = rect;
    }
}

public class ImageProcessor {
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    static int INPUT_WIDTH = 640;
    static int INPUT_HEIGHT = 640;

    private static int OUTPUT_ROW = 25200;
    private static int OUTPUT_COL = 85;
    private static float CONFIDENCE_THRESHOLD = 0.30f;
    private static int MAX_DETECTIONS = 15;

    static String[] CLASSES;

    static ArrayList<DetectionResult> applyNMS(ArrayList<DetectionResult> detections, int limit, float threshold) {
        Collections.sort(detections,
                new Comparator<DetectionResult>() {
                    @Override
                    public int compare(DetectionResult o1, DetectionResult o2) {
                        return o1.confidence.compareTo(o2.confidence);
                    }
                });

        ArrayList<DetectionResult> selected = new ArrayList<>();
        boolean[] active = new boolean[detections.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        boolean done = false;
        for (int i=0; i<detections.size() && !done; i++) {
            if (active[i]) {
                DetectionResult boxA = detections.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<detections.size(); j++) {
                    if (active[j]) {
                        DetectionResult boxB = detections.get(j);
                        if (calculateIOU(boxA.boundingBox, boxB.boundingBox) > threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    static float calculateIOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<DetectionResult> processOutputs(float[] outputs, float imgScaleX, float imgScaleY,
                                                     float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<DetectionResult> results = new ArrayList<>();
        for (int i = 0; i < OUTPUT_ROW; i++) {
            if (outputs[i * OUTPUT_COL + 4] > CONFIDENCE_THRESHOLD) {
                float x = outputs[i * OUTPUT_COL];
                float y = outputs[i * OUTPUT_COL + 1];
                float w = outputs[i * OUTPUT_COL + 2];
                float h = outputs[i * OUTPUT_COL + 3];

                float left = imgScaleX * (x - w/2);
                float top = imgScaleY * (y - h/2);
                float right = imgScaleX * (x + w/2);
                float bottom = imgScaleY * (y + h/2);

                float maxConf = outputs[i * OUTPUT_COL + 5];
                int cls = 0;
                for (int j = 0; j < OUTPUT_COL - 5; j++) {
                    if (outputs[i * OUTPUT_COL + 5 + j] > maxConf) {
                        maxConf = outputs[i * OUTPUT_COL + 5 + j];
                        cls = j;
                    }
                }

                Rect rect = new Rect((int)(startX + ivScaleX * left),
                        (int)(startY + top * ivScaleY),
                        (int)(startX + ivScaleX * right),
                        (int)(startY + ivScaleY * bottom));
                DetectionResult result = new DetectionResult(cls, outputs[i * OUTPUT_COL + 4], rect);
                results.add(result);
            }
        }
        return applyNMS(results, MAX_DETECTIONS, CONFIDENCE_THRESHOLD);
    }
}