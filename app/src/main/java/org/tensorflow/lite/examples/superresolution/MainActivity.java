/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.superresolution;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.WorkerThread;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.github.chrisbanes.photoview.PhotoView;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/** A super resolution class to generate super resolution images from low resolution images * */
public class MainActivity extends AppCompatActivity {
  private static final int PICK_PHOTO = 100;
  private static final int MY_PERMISSIONS_REQUEST_CALL_PHONE2 = 101;

  static {
    System.loadLibrary("SuperResolution");
  }

  private static final String TAG = "SuperResolution";
  private static final String MODEL_NAME = "ESRGAN.tflite";
  private static final int IN_HEIGHT = 50;
  private static final int IN_WIDTH = 50;
  private static final int UPSCALE_FACTOR = 4;
  private static final int OUT_HEIGHT = IN_HEIGHT * UPSCALE_FACTOR;
  private static final int OUT_WIDTH = IN_WIDTH * UPSCALE_FACTOR;
  private static final String LR_IMG_0 = "lr-0.jpg";
  private static final String LR_IMG_1 = "lr-1.jpg";
  private static final String LR_IMG_2 = "lr-2.jpg";
  private static final String LR_IMG_3 = "lr-3.jpg";
  private static final String LR_IMG_4 = "lr-4.jpg";
  private long processingTimeMs;

  private MappedByteBuffer model;
  private long superResolutionNativeHandle = 0;
  private Bitmap selectedLRBitmap = null;
  private Bitmap srBitmap = null;
  private boolean useGPU = false;

  private ImageView lowResImageView1;
  private ImageView lowResImageView2;
  private ImageView lowResImageView3;
  private PhotoView srPhotoView;
  private TextView selectedImageTextView;
  private TextView progressTextView;
  private TextView logTextView;
  private Switch gpuSwitch;
  private UIHandler UIhandler;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    requirePremision();

    final Button superResolutionButton = findViewById(R.id.upsample_button);
    lowResImageView1 = findViewById(R.id.low_resolution_image_1);
    lowResImageView2 = findViewById(R.id.low_resolution_image_2);
    lowResImageView3 = findViewById(R.id.low_resolution_image_3);
    progressTextView = findViewById(R.id.progress_tv);
    logTextView = findViewById(R.id.log_view);

    srPhotoView = (PhotoView) findViewById(R.id.sr_view);
    PhotoView selectedPhotoView = (PhotoView) findViewById(R.id.selected_view);

    selectedImageTextView = findViewById(R.id.chosen_image_tv);
    gpuSwitch = findViewById(R.id.switch_use_gpu);

    ImageView[] lowResImageViews = {lowResImageView1, lowResImageView2, lowResImageView3};

    AssetManager assetManager = getAssets();
    try {
      InputStream inputStream1 = assetManager.open(LR_IMG_1);
      Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream1);
      lowResImageView1.setImageBitmap(bitmap1);

      InputStream inputStream2 = assetManager.open(LR_IMG_4);
      Bitmap bitmap2 = BitmapFactory.decodeStream(inputStream2);
      lowResImageView2.setImageBitmap(bitmap2);

      InputStream inputStream3 = assetManager.open(LR_IMG_0);
      Bitmap bitmap3 = BitmapFactory.decodeStream(inputStream3);
      lowResImageView3.setImageBitmap(bitmap3);
    } catch (IOException e) {
      Log.e(TAG, "Failed to open an low resolution image");
    }

    for (ImageView iv : lowResImageViews) {
      setLRImageViewListener(iv);
    }

    superResolutionButton.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View view) {
            if (selectedLRBitmap == null) {
              Toast.makeText(
                      getApplicationContext(),
                      "Please choose one low resolution image",
                      Toast.LENGTH_LONG)
                  .show();
              return;
            }

            srPhotoView.setImageDrawable(null);
            selectedPhotoView.setImageBitmap(selectedLRBitmap);
            progressTextView.setText("loading...");

            if (superResolutionNativeHandle == 0) {
                superResolutionNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked());
            } else if (useGPU != gpuSwitch.isChecked()) {
              // We need to reinitialize interpreter when execution hardware is changed
              deinit();
              superResolutionNativeHandle = initTFLiteInterpreter(gpuSwitch.isChecked());
            }
            useGPU = gpuSwitch.isChecked();
            if (superResolutionNativeHandle == 0) {
              showToast("TFLite interpreter failed to create!");
              return;
            }

            UIhandler = new UIHandler();
            new Thread(new Runnable() {
              @Override
              public void run() {
                doSuperResolution();
              }
            }).start();

          }
        });


    //从相册选择图片
    findViewById(R.id.open_button).setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        //动态申请获取访问 读写磁盘的权限
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
          ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 101);
        } else {
          //打开相册
          Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
          //Intent.ACTION_GET_CONTENT = "android.intent.action.GET_CONTENT"
          intent.setType("image/*");
          startActivityForResult(intent, PICK_PHOTO); // 打开相册
        }
      }
    });

  }

  private class UIHandler extends Handler {
    @Override
    public void handleMessage(Message msg) {
      // TODO Auto-generated method stub
      super.handleMessage(msg);
      Bundle bundle = msg.getData();
      String progress = bundle.getString("progress");

      if(progress!=null){
        progressTextView.setText(progress);
        logTextView.setText(progress);
      }
      srPhotoView.setImageBitmap(srBitmap);
    }
  }

  private void requirePremision() {
    if (ContextCompat.checkSelfPermission(this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED)
    {
      ActivityCompat.requestPermissions(this,
              new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
              MY_PERMISSIONS_REQUEST_CALL_PHONE2);
    }else {
      //权限已经被授予，在这里直接写要执行的相应方法即可
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults)
  {
    if (requestCode == MY_PERMISSIONS_REQUEST_CALL_PHONE2)
    {
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
      {

      } else
      {
        // Permission Denied
        Toast.makeText(MainActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
      }
    }
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }


  @Override
  protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    switch (requestCode) {
      case PICK_PHOTO:
        if (resultCode == RESULT_OK && null != data) { // 判断手机系统版本号
          Uri uri = data.getData();
          try {
            selectedLRBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            selectedImageTextView.setText(
                    "You opened low resolution image:  ("
                            + data.toString()
                            + ")");
          } catch (IOException e) {
            e.printStackTrace();
            showToast("Pick photo failed!");
          }
        }
        break;
      default:
        break;
    }
    super.onActivityResult(requestCode, resultCode, data);
  }


  @Override
  public void onDestroy() {
    super.onDestroy();
    deinit();
  }

  private void setLRImageViewListener(ImageView iv) {
    iv.setOnTouchListener(
        new View.OnTouchListener() {
          @Override
          public boolean onTouch(View v, MotionEvent event) {
            if (v.equals(lowResImageView1)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView1.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 1");
            } else if (v.equals(lowResImageView2)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView2.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 2");
            } else if (v.equals(lowResImageView3)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView3.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 3");
            }
            return false;
          }
        });
  }

  @WorkerThread
  public synchronized void doSuperResolution() {
    final long startTime = SystemClock.uptimeMillis();
    int progress = 0;
    int w = selectedLRBitmap.getWidth();
    int h = selectedLRBitmap.getHeight();

    int max_a = (int)Math.ceil ((float)w/IN_WIDTH);
    int max_b = (int)Math.ceil((float)h/IN_HEIGHT);

    int max_w = max_a * IN_WIDTH;
    int max_h = max_b * IN_HEIGHT;

    srBitmap = Bitmap.createBitmap(w*UPSCALE_FACTOR,h*UPSCALE_FACTOR, Bitmap.Config.ARGB_8888);
    Bitmap inputBitmap = Bitmap.createBitmap(selectedLRBitmap,0,0,w,h);

    for(int a = 0; a<max_a;a++){
      Message msg = new Message();
      if(a!=0){
        Bundle bundle = new Bundle();
        bundle.putString("progress", "progress: " +  a + "/" + max_a
                + ", need " +  (SystemClock.uptimeMillis() - startTime)/a*(max_a -a) + "ms");
        msg.setData(bundle);
      }
      MainActivity.this.UIhandler.sendMessage(msg);

//      int in_width = (max_a-a==1)?w-a*IN_WIDTH:IN_WIDTH;
      int x = (max_a-a==1)?(w-IN_WIDTH):IN_WIDTH*a;

      for(int b=0;b<max_b;b++){
//        int in_height = (max_b-b==1)?h-b*IN_HEIGHT:IN_HEIGHT;
        int y = (max_b-b==1)?h-IN_HEIGHT:IN_HEIGHT*b;
        int[] lowResRGB = new int[IN_WIDTH*IN_HEIGHT];

        inputBitmap.getPixels(
                lowResRGB, 0, IN_WIDTH,x , y, IN_WIDTH, IN_HEIGHT);

        srBitmap.setPixels(
                superResolutionFromJNI(superResolutionNativeHandle, lowResRGB)
                ,0,OUT_WIDTH
                ,UPSCALE_FACTOR*x,UPSCALE_FACTOR*y,OUT_WIDTH,OUT_HEIGHT);
        if(progress<0){
          processingTimeMs = -1;
          return ;
        }
        progress ++;
      }
    }
    processingTimeMs = SystemClock.uptimeMillis() - startTime;
    Message msg = new Message();
    Bundle bundle = new Bundle();
    bundle.putString("progress", "Inference time: " + processingTimeMs + "ms");
    msg.setData(bundle);
    MainActivity.this.UIhandler.sendMessage(msg);

  }


  @WorkerThread
  public synchronized int[] doSuperResolution(int[] lowResRGB, int w, int h) {
    return superResolutionFromJNI(superResolutionNativeHandle, lowResRGB);
  }

  private MappedByteBuffer loadModelFile() throws IOException {
    try (AssetFileDescriptor fileDescriptor =
            AssetsUtil.getAssetFileDescriptorOrCached(getApplicationContext(), MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  private void showToast(String str) {
    Toast.makeText(getApplicationContext(), str, Toast.LENGTH_LONG).show();
  }

  private long initTFLiteInterpreter(boolean useGPU) {
    try {
      model = loadModelFile();
    } catch (IOException e) {
      Log.e(TAG, "Fail to load model", e);
    }
    return initWithByteBufferFromJNI(model, useGPU);
  }

  private void deinit() {
    deinitFromJNI(superResolutionNativeHandle);
  }

  private native int[] superResolutionFromJNI(long superResolutionNativeHandle, int[] lowResRGB);

  private native long initWithByteBufferFromJNI(MappedByteBuffer modelBuffer, boolean useGPU);

  private native void deinitFromJNI(long superResolutionNativeHandle);
}
