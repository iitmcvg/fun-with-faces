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



std::string IntToString ( int number )
{
  std::ostringstream oss;
  // Works just like cout
  oss<< number;
  // Return the underlying string
  return oss.str();
}



int main(int argc, char** argv) {
    try {
        cv::VideoCapture cap(0);
        image_window win, win2,win3;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;
        cv::Mat frame, frame_clr, temp1, temp2, temp3, roi_l_clr, roi_r_clr, roi_l_gray, roi_r_gray;
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

               std::vector<cv::Point> dest;
               for(int k=31;k<=35;k++)
               {
                 dest.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
               }
               for(int k=48;k<=54;k++)
               {
                 dest.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
               }

               cv::Mat obj;
               obj=cv::imread("/home/pooja/projects/obj3.jpg", 0);
               cv::Mat obj_clr = cv::imread("/home/pooja/projects/obj3.jpg");
               cv::Mat sobj, sobj_clr, mask, invert, bg,fg;


               cv::Rect rect=cv::boundingRect(dest);
               //cv::rectangle(frame_clr,rect,(0,255,255),1,8,0);

                int rx,ry,rh,rw;
                rx=rect.x;
                ry=rect.y;
                rh=rect.height;
                rw=rect.width;


                cv::resize(obj,sobj,cv::Size(rw,rh));
                cv::resize(obj_clr,sobj_clr,cv::Size(rw,rh));

                cv::threshold(sobj,mask,0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);
                cv::bitwise_not(mask,invert);

                cv::Rect roi = cv::Rect(rx, ry, rw, rh);
                cv::Mat image_roi = frame_clr(roi);

                cv::bitwise_and(image_roi, image_roi,bg,mask);  //checkout :P
                cv::bitwise_and(sobj_clr, sobj_clr, fg,invert);
                cv::add(fg, bg, image_roi);
                //cv::bitwise_and(frame_clr,sobj)


      //STARTHERE    #    cv::addWeighted()
                //cout<<"coordinates"<<rx<<"and"<<ry<<endl;
                //cout<<"ht "<<rh<<"and"<<rw<<endl;
               std::vector<full_object_detection> shapes;


                        //  for (unsigned long j = 0; j < faces.size(); ++j)
                          //{

                        //      cout << "number of parts: "<< shape.num_parts() << endl;
    //for finding out the indices
                          /*    for(int k=0;k<shape.num_parts();k++)
                              {
                                  cout<<"pixel position of "<<k<<"is  "<<shape.part(k) <<endl;

                                  cv::Point pt= cv::Point(shape.part(k).x(), shape.part(k).y());


                                  std::string text = IntToString (k);
                                cv::putText(frame_clr, text, pt, CV_FONT_HERSHEY_COMPLEX, 0.5,(255, 0,0), 1,8);
                              }*/
                            //}

                              shapes.push_back(shape);
win2.set_image(cv_image<unsigned char>(mask));
win3.set_image(cv_image<unsigned char>(obj));

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
