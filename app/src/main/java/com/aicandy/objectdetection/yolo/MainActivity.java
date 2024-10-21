package com.aicandy.objectdetection.yolo;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int currentImageIndex = 0;
    private ImageView imageDisplay;
    private DetectionOverlay detectionOverlay;
    private Button detectButton;
    private ProgressBar loadingBar;
    private Bitmap inputImage = null;
    private Module modelModule = null;
    private float imageScaleX, imageScaleY, viewScaleX, viewScaleY, originX, originY;
    private int indexImage = 0;
    private List<String> images;

    public static String getAssetPath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private List<String> getFileFromAssetsFolder(String folderPath) {
        List<String> imageFiles = new ArrayList<>();
        AssetManager assetManager = this.getAssets();

        try {
            String[] files = assetManager.list(folderPath);
            if (files != null) {
                for (String fileName : files) {
                    if (fileName.endsWith(".png") || fileName.endsWith(".jpg")) {
                        imageFiles.add(folderPath + "/" + fileName);
                    }
                }
            } else {
                Log.d("AIcandy.vn", "The folder is empty or does not exist.");
            }
        } catch (IOException e) {
            Log.e("AIcandy.vn", "Error accessing the assets folder: ", e);
        }

        return imageFiles;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        setContentView(R.layout.activity_main);

        // Get the list of image files from the assets folder
        images = getFileFromAssetsFolder("image_test");
        loadNextImage();

        imageDisplay = findViewById(R.id.imageView);
        imageDisplay.setImageBitmap(inputImage);
        detectionOverlay = findViewById(R.id.resultView);
        detectionOverlay.setVisibility(View.INVISIBLE);

        detectButton = findViewById(R.id.detectButton);
        loadingBar = findViewById(R.id.progressBar);
        detectButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                detectButton.setEnabled(false);
                loadingBar.setVisibility(ProgressBar.VISIBLE);
                detectButton.setText(getString(R.string.run_model));

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            modelModule = LiteModuleLoader.load(MainActivity.getAssetPath(getApplicationContext(), "yolov5s.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("model_classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            ImageProcessor.CLASSES = new String[classes.size()];
            classes.toArray(ImageProcessor.CLASSES);
        } catch (IOException e) {
            Log.e("AIcandy.vn", "Error reading assets: ", e);
            finish();
        }
    }

    private void loadNextImage() {
        if (currentImageIndex >= images.size()) {
            currentImageIndex = 0;
        }
        try {
            inputImage = BitmapFactory.decodeStream(getAssets().open(images.get(currentImageIndex)));
            Log.d("AIcandy.vn", "Processing: " + images.get(currentImageIndex));
            currentImageIndex++;
        } catch (IOException e) {
            Log.e("AIcandy.vn", "Error reading image asset: ", e);
        }
    }

    @Override
    public void run() {
        Bitmap resizedImage = Bitmap.createScaledBitmap(inputImage, ImageProcessor.INPUT_WIDTH, ImageProcessor.INPUT_HEIGHT, true);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedImage, ImageProcessor.NO_MEAN_RGB, ImageProcessor.NO_STD_RGB);
        IValue[] outputTuple = modelModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();

        imageScaleX = (float)inputImage.getWidth() / ImageProcessor.INPUT_WIDTH;
        imageScaleY = (float)inputImage.getHeight() / ImageProcessor.INPUT_HEIGHT;

        viewScaleX = (inputImage.getWidth() > inputImage.getHeight() ? (float)imageDisplay.getWidth() / inputImage.getWidth() : (float)imageDisplay.getHeight() / inputImage.getHeight());
        viewScaleY = (inputImage.getHeight() > inputImage.getWidth() ? (float)imageDisplay.getHeight() / inputImage.getHeight() : (float)imageDisplay.getWidth() / inputImage.getWidth());

        originX = (imageDisplay.getWidth() - viewScaleX * inputImage.getWidth())/2;
        originY = (imageDisplay.getHeight() - viewScaleY * inputImage.getHeight())/2;

        final ArrayList<DetectionResult> results = ImageProcessor.processOutputs(outputs, imageScaleX, imageScaleY, viewScaleX, viewScaleY, originX, originY);

        runOnUiThread(() -> {
            imageDisplay.setImageBitmap(inputImage);
            detectButton.setEnabled(true);
            detectButton.setText(getString(R.string.run));
            loadingBar.setVisibility(ProgressBar.INVISIBLE);
            detectionOverlay.setDetections(results);
            detectionOverlay.invalidate();
            detectionOverlay.setVisibility(View.VISIBLE);

            // Load the next image for the next detection
            loadNextImage();
        });
    }
}