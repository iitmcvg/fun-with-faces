#include <math.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

using namespace dlib;
using namespace std;

void preprocessROI(cv::Mat& roi_eye) {
    GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
    equalizeHist( roi_eye, roi_eye );
}



/*std::string IntToString ( int number )
{
  std::ostringstream oss;
  // Works just like cout
  oss<< number;
  // Return the underlying string
  return oss.str();
}*/


void mask(cv::Mat obj_clr,std::vector<cv::Point> dest, cv::Mat frame_clr)
{
  cout<<"welcome to mask function :P \n";

cv::Mat obj;

cv::cvtColor(obj_clr,obj,CV_BGR2GRAY);

cv::Mat sobj, sobj_clr, mask, invert, bg,fg;

cv::Rect rect=cv::boundingRect(dest);

 int rx,ry,rh,rw;
 rx=rect.x;
 ry=rect.y;
 rh=rect.height;
 rw=rect.width;

//cv::rectangle(frame_clr,cv::Point(rx-rw/2,ry-rh/2),cv::Point(rx+rw/2,ry+rh/2),cv::Scalar(0, 0, 255));
//cout<<"line1 ";

 cv::resize(obj,sobj,cv::Size(rw,rh));
 //cout<<"line 1.1";
 cv::resize(obj_clr,sobj_clr,cv::Size(rw,rh));
//cout<<"line2  ";
 cv::threshold(sobj,mask,0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);
 cv::bitwise_not(mask,invert);
 //cout<<"line3  ";
 cv::RotatedRect roi=rect;
 //cv::Rect roi = cv::Rect(rx-rw/2, ry-rh/2, rw, rh);
 cv::Mat image_roi = frame_clr(roi);
 //cout<<"line4  ";

 cv::bitwise_and(image_roi, image_roi,bg,mask);  //checkout :P
 //cout<<"line5";
 cv::bitwise_and(sobj_clr, sobj_clr, fg,invert);
 //cout<<"line6";
 cv::add(fg, bg, image_roi);
 cout<<"reached end of mask :D \n" ;
}


int main(int argc, char** argv) {
    try {
        cv::VideoCapture cap(0);
        image_window win;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;
        cv::Mat frame,frame_clr, temp1, temp2, temp3, roi_l_clr, roi_r_clr, roi_l_gray, roi_r_gray;
        cv::Mat temp;
        while(!win.is_closed()) {
            cap >> frame_clr;

            cv::flip(frame_clr, frame_clr, 1);
            cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);
            cv_image<unsigned char> cimg_gray(frame);
//detect faces
            std::vector<rectangle> faces = detector(cimg_gray);

            std::vector<full_object_detection> shapes;

            for (unsigned long i = 0; i < faces.size(); ++i)
              {
               shapes.push_back(pose_model(cimg_gray, faces[i]));

               full_object_detection shape = pose_model(cimg_gray, faces[i]);

               std::vector<cv::Point> dest2;
               for(int k=36;k<=41;k++) //eye
               {
                 dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

               }

               for(int k=42;k<=47;k++) //eye
               {
                 dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

               }

                for(int k=17;k<=26;k++)    //eyebrow
                {
                dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()+10));

                }

               cv::Mat obj_clr2 =cv::imread("./res/images/specs2.png");
               mask(obj_clr2,dest2,frame_clr);


                std::vector<cv::Point> dest1;
                for(int k=31;k<=35;k++) //moustache
                {
                  dest1.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

                }

                for(int k=48;k<=54;k++)//moustache
                {
                  dest1.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

                }

                
                cv::Mat obj_clr1 = cv::imread("./res/images/moustache.jpg");


            mask(obj_clr1,dest1,frame_clr);

               std::vector<cv::Point> dest3;
              for(int k=29;k<=35;k++)    //nosetip
               {
                 dest3.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
               }
               cv::Mat obj_clr3=cv::imread("./res/images/clown_nose.jpg");
               mask(obj_clr3,dest3,frame_clr);


              // cv::circle(frame_clr,cv::Point(shape.part(30).x(), shape.part(30).y()),20,cv::Scalar(0, 0, 255),-1);
                              std::vector<full_object_detection> shapes;
                              shapes.push_back(shape);

             }
            win.clear_overlay();
            win.set_image(cv_image<bgr_pixel>(frame_clr));// cv image , rgb image
            //win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(exception& e) {
      cout << e.what() << endl;
    }
}
