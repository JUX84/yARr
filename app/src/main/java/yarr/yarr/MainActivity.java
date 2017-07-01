package yarr.yarr;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "yARr";
    private static final int ORIGINAL_WIDTH = 1280;
    private static final int ORIGINAL_HEIGHT = 720;

    static {
        System.loadLibrary("opencv_java3");
    }

    private List<Frame> frames;
    private int WIDTH = 960;
    private int HEIGHT = 540;
    private Mat corners_object;
    private List<KeyPoint> list_keypoints_object;
    private CameraBridgeViewBase mOpenCvCameraView;
    private ORB detector;
    private DescriptorMatcher matcher;
    private Mat object;
    private MatOfKeyPoint keypoints_object;
    private Mat descriptors_object;
    private List<Detector> threads;
    private double frameIdx;
    private double currentFrameIdx;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // int nfeatures=500,
                    // float scaleFactor=1.2f,
                    // int nlevels=8,
                    // int edgeThreshold=31,
                    // int firstLevel=0,
                    // int WTA_K=2,
                    // int scoreType=ORB::HARRIS_SCORE,
                    // int patchSize=31,
                    // int fastThreshold=20
                    detector = ORB.create(1000, 2.0f, 4, 31, 0, 2, ORB.HARRIS_SCORE, 31, 100);

                    matcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING);

                    threads = new LinkedList<>();

                    frames = new LinkedList<>();
                    frameIdx = 0;
                    currentFrameIdx = 0;

                    new Thread() {
                        public void run() {
                            URL url;
                            Bitmap bmp = null;
                            try {
                                url = new URL("http://vrcodex.com/wp-content/uploads/2014/09/stones.jpg");
                                try {
                                    bmp = BitmapFactory.decodeStream(url.openConnection().getInputStream());
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            } catch (MalformedURLException e) {
                                e.printStackTrace();
                            }
                            if (bmp != null) {
                                object = new Mat();
                                Utils.bitmapToMat(bmp, object);
                                Imgproc.cvtColor(object, object, Imgproc.COLOR_RGB2GRAY);

                                corners_object = new Mat(4, 1, CvType.CV_32FC2);

                                double[] obj_corner0 = new double[2];
                                obj_corner0[0] = 0;
                                obj_corner0[1] = 0;
                                double[] obj_corner1 = new double[2];
                                obj_corner1[0] = object.cols();
                                obj_corner1[1] = 0;
                                double[] obj_corner2 = new double[2];
                                obj_corner2[0] = object.cols();
                                obj_corner2[1] = object.rows();
                                double[] obj_corner3 = new double[2];
                                obj_corner3[0] = 0;
                                obj_corner3[1] = object.rows();

                                corners_object.put(0, 0, obj_corner0);
                                corners_object.put(1, 0, obj_corner1);
                                corners_object.put(2, 0, obj_corner2);
                                corners_object.put(3, 0, obj_corner3);

                                keypoints_object = new MatOfKeyPoint();
                                descriptors_object = new Mat();

                                detector.detectAndCompute(object, new Mat(), keypoints_object, descriptors_object);
                                list_keypoints_object = keypoints_object.toList();
                            }
                        }
                    }.start();

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            }
            if (checkSelfPermission(Manifest.permission.INTERNET) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.INTERNET}, 1);
            }
        }

        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onPause() {
        super.onPause();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat currentFrame = new Mat();
        Iterator<Detector> i = threads.iterator();
        while (i.hasNext()) {
            Detector d = i.next();
            int status = d.getStatus();
            double index = d.getIndex();
            if (index == currentFrameIdx && status > 0) {
                frames.add(new Frame(index, d.getFrame()));
                i.remove();
                break;
            }
        }
        Iterator<Frame> j = frames.iterator();
        while (j.hasNext()) {
            Frame f = j.next();
            double index = f.getIndex();
            if (index == currentFrameIdx) {
                currentFrame = f.getFrame();
                currentFrameIdx += 1;
                j.remove();
                break;
            }
        }

        Mat frame = inputFrame.rgba();

        if (detector == null || list_keypoints_object == null || list_keypoints_object.isEmpty())
            return frame;

        if (threads.size() < Runtime.getRuntime().availableProcessors()) {
            Detector d = new Detector(frameIdx, frame.clone());
            new Thread(d).start();
            threads.add(d);
        } else {
            frames.add(new Frame(frameIdx, frame.clone()));
        }

        frameIdx += 1;

        if (currentFrame.empty())
            return frame;

        return currentFrame;
    }

    private class Detector implements Runnable {
        private Mat frame;
        private Mat gray;
        private int status;
        private double index;
        private Point[] points;

        Detector(double idx, Mat mat) {
            index = idx;
            frame = mat;
            gray = frame.clone();
            long t = System.currentTimeMillis();
            Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY);
            Imgproc.resize(gray, gray, new Size(WIDTH, HEIGHT));
            Log.i(TAG, "imgproc: " + (System.currentTimeMillis() - t) + "ms");
            gray = new Mat(gray, new Rect(WIDTH / 4, HEIGHT / 6, WIDTH / 4 * 2, HEIGHT / 6 * 4));
        }

        @Override
        public void run() {
            detectRegion(gray);
            if (status == 0)
                status = 1;
        }

        void detectRegion(Mat region) {
            MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();
            Mat descriptors_scene = new Mat();

            long t = System.currentTimeMillis();
            detector.detectAndCompute(region, new Mat(), keypoints_scene, descriptors_scene);
            Log.i(TAG, "detectAndCompute: " + (System.currentTimeMillis() - t) + "ms");
            if (descriptors_scene.empty())
                return;

            MatOfDMatch matches = new MatOfDMatch();

            t = System.currentTimeMillis();
            matcher.match(descriptors_object, descriptors_scene, matches);
            Log.i(TAG, "match: " + (System.currentTimeMillis() - t) + "ms");
            if (matches.empty())
                return;

            List<DMatch> list_matches = matches.toList();
            if (list_matches.size() < 4)
                return;

            double min_dist = Double.MAX_VALUE;
            double max_dist = Double.MIN_VALUE;

            for (int i = 0; i < descriptors_scene.rows(); i++) {
                double dist = list_matches.get(i).distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }

            List<Point> points_object = new LinkedList<>();
            List<Point> points_scene = new LinkedList<>();
            List<KeyPoint> list_keypoints_scene = keypoints_scene.toList();

            for (int i = 0; i < descriptors_object.rows(); ++i) {
                if (list_matches.get(i).distance < 2 * min_dist) {
                    points_object.add(list_keypoints_object.get(list_matches.get(i).queryIdx).pt);
                    points_scene.add(list_keypoints_scene.get(list_matches.get(i).trainIdx).pt);
                }
            }

            if (points_object.size() == 0 || points_scene.size() == 0)
                return;

            MatOfPoint2f mat_points_object = new MatOfPoint2f();
            mat_points_object.fromList(points_object);
            MatOfPoint2f mat_points_scene = new MatOfPoint2f();
            mat_points_scene.fromList(points_scene);

            t = System.currentTimeMillis();
            Mat homography = Calib3d.findHomography(mat_points_object, mat_points_scene, Calib3d.RANSAC, 10);
            Log.i(TAG, "homography: " + (System.currentTimeMillis() - t) + "ms");

            if (homography.empty())
                return;

            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            t = System.currentTimeMillis();
            Core.perspectiveTransform(corners_object, scene_corners, homography);
            Log.i(TAG, "perspectiveTransform: " + (System.currentTimeMillis() - t) + "ms");

            points = new Point[4];
            points[0] = new Point(scene_corners.get(0, 0));
            points[1] = new Point(scene_corners.get(1, 0));
            points[2] = new Point(scene_corners.get(2, 0));
            points[3] = new Point(scene_corners.get(3, 0));

            for (Point p : points) {
                p.x = (p.x + WIDTH / 4) * ((double)ORIGINAL_WIDTH / (double)WIDTH);
                p.y = (p.y + HEIGHT / 6) * ((double)ORIGINAL_HEIGHT / (double)HEIGHT);
            }

            if (isRectangular(points)) {
                Scalar green = new Scalar(0, 255, 0);

                Imgproc.line(frame, points[0], points[1], green, 4);
                Imgproc.line(frame, points[1], points[2], green, 4);
                Imgproc.line(frame, points[2], points[3], green, 4);
                Imgproc.line(frame, points[3], points[0], green, 4);

                status = 2;
            }
        }

        double calculateAngle(double p1c, double cp0, double p0p1) {
            return Math.acos((p1c * p1c + cp0 * cp0 - p0p1 * p0p1) / (2 * p1c * cp0)) * (180 / Math.PI);
        }

        boolean isRectangular(Point[] p) {
            double p0p1 = Math.sqrt(Math.pow(p[1].x - p[0].x, 2) + Math.pow(p[1].y - p[0].y, 2));
            double p1p2 = Math.sqrt(Math.pow(p[2].x - p[1].x, 2) + Math.pow(p[2].y - p[1].y, 2));
            double p2p3 = Math.sqrt(Math.pow(p[3].x - p[2].x, 2) + Math.pow(p[3].y - p[2].y, 2));
            double p3p0 = Math.sqrt(Math.pow(p[0].x - p[3].x, 2) + Math.pow(p[0].y - p[3].y, 2));

            double p0p2 = Math.sqrt(Math.pow(p[2].x - p[0].x, 2) + Math.pow(p[2].y - p[0].y, 2));
            double p1p3 = Math.sqrt(Math.pow(p[3].x - p[1].x, 2) + Math.pow(p[3].y - p[1].y, 2));

            if (p0p1 < 150)
                return false;
            if (p1p2 < 100)
                return false;
            if (p2p3 < 150)
                return false;
            if (p3p0 < 100)
                return false;
            if (p0p2 < 120)
                return false;
            if (p1p3 < 120)
                return false;

            double angle = calculateAngle(p3p0, p0p1, p1p3);
            if (angle < 70 || angle > 110)
                return false;

            angle = calculateAngle(p0p1, p1p2, p0p2);
            if (angle < 70 || angle > 110)
                return false;

            angle = calculateAngle(p1p2, p2p3, p1p3);
            if (angle < 70 || angle > 110)
                return false;

            angle = calculateAngle(p2p3, p3p0, p0p2);
            return !(angle < 70 || angle > 110);
        }

        double getIndex() {
            return index;
        }

        int getStatus() {
            return status;
        }

        Mat getFrame() {
            return frame;
        }
    }

    private class Frame {
        private double index;
        private Mat frame;

        Frame(double idx, Mat mat) {
            index = idx;
            frame = mat;
        }

        double getIndex() {
            return index;
        }

        Mat getFrame() {
            return frame;
        }
    }
}
