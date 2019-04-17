#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <iostream>
#include <iomanip>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <dlib/opencv.h>

using namespace dlib;
using namespace std;
using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "Usage: ./cascadeclassifier \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id>)\n"
            "--predictior <predictor_file.dat> \n Using OpenCV version " << CV_VERSION << endl << endl;
}

static void convertAndResize(const Mat& src, Mat& gray, Mat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}

static void convertAndResize(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}

static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, Scalar(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = Scalar(255,0,0);
    Scalar fontColorNV  = Scalar(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}
// ----------------------------------------------------------------------------------------

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}



int main(int argc, char** argv)
{  
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            help();
            return -1;
        }

        if (getCudaEnabledDeviceCount() == 0)
            {
                return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;
            }

        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

            string cascadeName;
            string inputName;
            bool isInputImage = false;
            bool isInputVideo = false;

            shape_predictor sp;

            for (int i = 1; i < argc; ++i)
                {
                    if (string(argv[i]) == "--cascade")
                        cascadeName = argv[++i];
                    else if (string(argv[i]) == "--video")
                    {
                        inputName = argv[++i];
                        isInputVideo = true;
                    }
                    else if (string(argv[i]) == "--predictor")
                    {
                        deserialize(argv[++i]) >> sp;
                    }
                    else if (string(argv[i]) == "--help")
                    {
                        help();
                        return -1;
                    }
                    else if (!isInputImage)
                    {
                        inputName = argv[i];
                        isInputImage = true;
                    }
                    else
                    {
                        cout << "Unknown key: " << argv[i] << endl;
                        return -1;
                    }
                }

            Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);

               cv::CascadeClassifier cascade_cpu;
               if (!cascade_cpu.load(cascadeName))
               {
                   return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
               }

               VideoCapture capture;
               Mat image;


               if (isInputImage)
                  {
                      image = imread(inputName);
                      CV_Assert(!image.empty());
                  }
                  else if (isInputVideo)
                  {
                      capture.open(inputName);
                      CV_Assert(capture.isOpened());
                  }


               namedWindow("result", 1);

               Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
               std::vector<cv::Rect> faces;

               GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;


               bool useGPU = true;
               double scaleFactor = 1.0;
               bool findLargestObject = false;
               bool filterRects = true;
               bool helpScreen = false;

               int countf = 0;
               for (;;)
               {
                   if (isInputVideo)
                   {
                       capture >> frame;
                       if (frame.empty())
                       {
                           break;
                       }
                   }

                   (image.empty() ? frame : image).copyTo(frame_cpu);
                   frame_gpu.upload(image.empty() ? frame : image);

                   convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
                   convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);

                   TickMeter tm;
                   tm.start();

                   if (useGPU)
                   {
                       cascade_gpu->setFindLargestObject(findLargestObject);
                       cascade_gpu->setScaleFactor(1.2);
                       cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);

                       cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
                       cascade_gpu->convert(facesBuf_gpu, faces);
                   }
                   else
                   {
                       Size minSize = cascade_gpu->getClassifierSize();
                       cascade_cpu.detectMultiScale(resized_cpu, faces, 1.2,
                                                    (filterRects || findLargestObject) ? 4 : 0,
                                                    (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                                                    | CASCADE_SCALE_IMAGE,
                                                    minSize);
                   }

                   std::vector<dlib::rectangle> dets;
                   for(auto && f : faces)
                        dets.push_back(openCVRectToDlib(f));

                   dlib::array2d<unsigned char> dlibImageGray;
                   dlib::assign_image(dlibImageGray, dlib::cv_image<unsigned char>(resized_cpu));

                   //dlib::cv_image<dlib::bgr_pixel> img(resized_cpu);

                   std::vector<full_object_detection> shapes;
                   std::vector<std::vector<cv::Point>> cvShapes;
                   for (unsigned long j = 0; j < dets.size(); ++j)
                   {
                       std::vector<cv::Point> cvShape;
                       full_object_detection shape = sp(dlibImageGray, dets[j]);
                       //cout << "number of parts: "<< shape.num_parts() << endl;
                       //cout << "pixel position of first part:  " << shape.part(0) << endl;
                       //cout << "pixel position of second part: " << shape.part(1) << endl;

                       for(int i =0; i < shape.num_parts(); i++)
                       {
                           cvShape.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
                       }


                       //cvShape.push_back(cv::Point(shape.part(0).x(), shape.part(0).y()));
                       //cvShape.push_back(cv::Point(shape.part(1).x(), shape.part(1).y()));
                       //cvShape.push_back(cv::Point(shape.part(2).x(), shape.part(2).y()));
                       //cvShape.push_back(cv::Point(shape.part(3).x(), shape.part(3).y()));
                       // You get the idea, you can get all the face part locations if
                       // you want them.  Here we just store them in shapes so we can
                       // put them on the screen.
                       shapes.push_back(shape);
                       cvShapes.push_back(cvShape);
                   }



                   for (size_t i = 0; i < faces.size(); ++i)
                   {
                       cv::rectangle(resized_cpu, faces[i], Scalar(255));
                   }

                   tm.stop();
                   double detectionTime = tm.getTimeMilli();
                   double fps = 1000 / detectionTime;

                   //print detections to console
                   cout << setfill(' ') << setprecision(2);
                   cout << setw(6) << fixed << fps << " FPS, " << faces.size() << " det";
                   if ((filterRects || findLargestObject) && !faces.empty())
                   {
                       for (size_t i = 0; i < faces.size(); ++i)
                       {
                           cout << ", [" << setw(4) << faces[i].x
                                << ", " << setw(4) << faces[i].y
                                << ", " << setw(4) << faces[i].width
                                << ", " << setw(4) << faces[i].height << "]";
                       }
                   }
                   cout << endl;

                   cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);

                   for(auto && cvs : cvShapes)
                       for(uint i = 0; i < cvs.size(); i++)
                       {
                           std::cout << "but size: " << cvs.size() << std::endl;

                           uint next_i = i < cvs.size() - 1 ? i + 1 : 0;
                           cv::line(frameDisp, cvs[i], cvs[next_i], cv::Scalar(255,255,0),1);

                           cv::circle(frameDisp, cvs[i], 3, cv::Scalar(255/12 * i,255/12 * i,255), 2);

                           if(i == 0)
                               cv::circle(frameDisp, cvs[i], 3, cv::Scalar(0,0,255), 2);

                           if(i == cvs.size() - 1)
                               cv::circle(frameDisp, cvs[i], 3, cv::Scalar(255,0,0), 2);

                           //cv::line(frameDisp, s[0], s[1], cv::Scalar(255,255,0),2);
                           //cv::line(frameDisp, s[1], s[2], cv::Scalar(255,255,0),2);
                           //cv::line(frameDisp, s[2], s[3], cv::Scalar(255,255,0),2);
                           //cv::line(frameDisp, s[3], s[0], cv::Scalar(255,255,0),2);
                       }

                   displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
                   imshow("result", frameDisp);
                   std::string imgstr("filmik/img-");
                   std::string count_string;
                   count_string = std::to_string(countf);
                   int padding_size = 5 - count_string.length();
                   std::string padding_append = std::string(padding_size, '0');
                   imgstr.append(padding_append);
                   imgstr.append(std::to_string(countf));

                   imgstr += ".png";
                   imwrite(imgstr, frameDisp);
                   countf++;

                   char key = (char)waitKey(5);
                   if (key == 27)
                   {
                       break;
                   }

                   switch (key)
                   {
                   case ' ':
                       useGPU = !useGPU;
                       break;
                   case 'm':
                   case 'M':
                       findLargestObject = !findLargestObject;
                       break;
                   case 'f':
                   case 'F':
                       filterRects = !filterRects;
                       break;
                   case '1':
                       scaleFactor *= 1.05;
                       break;
                   case 'q':
                   case 'Q':
                       scaleFactor /= 1.05;
                       break;
                   case 'h':
                   case 'H':
                       helpScreen = !helpScreen;
                       break;
                   }
               }

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

