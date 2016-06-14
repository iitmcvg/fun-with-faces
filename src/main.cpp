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
// errors: inverted color of objects, hat overlay not exact hence line 227 is commented :(

void mask2(cv::Mat obj_clr,int rx,int ry,int rw, int rh ,float angle, cv::Mat frame_clr, cv::Mat bg, cv::Mat fg)
{
//  cout<<"welcome to mask function :P main\n";
  //cv::rectangle(frame_clr,cv::Point(rx-rw/2,ry-rh/2),cv::Point(rx+rw/2,ry+rh/2),cv::Scalar(0, 0, 255));

cv::bitwise_not(obj_clr,obj_clr);

cv::Mat temp;
cv::Mat rot_mat( 2, 3, CV_32FC1 );
cv::Mat warp_mat( 2, 3, CV_32FC1 );
cv::Mat persp_mat(3,3,CV_32FC1);
cv::RotatedRect rect=cv::RotatedRect(cv::Point(rx,ry),cv::Size(rw,rh), angle);
// std::vector<cv::Point> vertices;
cv::Point2f vertices[4]; // roi corners
rect.points(vertices);
cv::Point2f corners[4]; //object corners

temp= cv::Mat::zeros(frame_clr.rows, frame_clr.cols, CV_8UC3);

//rot_mat = cv::getRotationMatrix2D( cv::Point(rx,ry), angle, 1);// whats that scale?
//cv::warpAffine( obj_clr, temp, rot_mat, frame_clr.size() );
//warp_mat = cv::getAffineTransform( corners, vertices );
//cv::warpAffine( temp, temp, warp_mat, temp.size() );
// corners[0]= cv::Point2f(0,0);
// corners[3]= cv::Point2f(obj_clr.cols-1, 0);
// corners[1]= cv::Point2f(0, obj_clr.rows-1);
// corners[2]= cv::Point2f(obj_clr.cols -1, obj_clr.rows -1);

corners[1]= cv::Point2f(0,0);
corners[2]= cv::Point2f(obj_clr.cols-1, 0);
corners[0]= cv::Point2f(0, obj_clr.rows-1);
corners[3]= cv::Point2f(obj_clr.cols -1, obj_clr.rows -1);


persp_mat=cv::getPerspectiveTransform(corners , vertices );

cv::warpPerspective(obj_clr,temp,persp_mat,temp.size());

cv::Mat temp_gray,mask,invert;
cv::cvtColor(temp,temp_gray,CV_BGR2GRAY);
cv::threshold(temp_gray,mask,0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);

cv::bitwise_not(mask,invert);

cv::bitwise_and(frame_clr, frame_clr,bg,invert);  //checkout :P
cv::bitwise_and(temp, temp, fg,mask);

cv::add(fg, bg, frame_clr);
 // DONE
}

void mask(cv::Mat obj_clr,int rx,int ry,int rw, int rh ,float angle, cv::Mat frame_clr, cv::Mat bg, cv::Mat fg)
{
//  cout<<"welcome to mask function :P main\n";
  //cv::rectangle(frame_clr,cv::Point(rx-rw/2,ry-rh/2),cv::Point(rx+rw/2,ry+rh/2),cv::Scalar(0, 0, 255));

cv::bitwise_not(obj_clr,obj_clr);
cv::Mat temp;
cv::Mat rot_mat( 2, 3, CV_32FC1 );
cv::Mat warp_mat( 2, 3, CV_32FC1 );
cv::Mat persp_mat(3,3,CV_32FC1);
cv::RotatedRect rect=cv::RotatedRect(cv::Point(rx,ry),cv::Size(rw,rh), angle);
// std::vector<cv::Point> vertices;
cv::Point2f vertices[4]; // roi corners
rect.points(vertices);
cv::Point2f corners[4]; //object corners

temp= cv::Mat::zeros(frame_clr.rows, frame_clr.cols, CV_8UC3);

//rot_mat = cv::getRotationMatrix2D( cv::Point(rx,ry), angle, 1);// whats that scale?
//cv::warpAffine( obj_clr, temp, rot_mat, frame_clr.size() );
//warp_mat = cv::getAffineTransform( corners, vertices );
//cv::warpAffine( temp, temp, warp_mat, temp.size() );
corners[0]= cv::Point2f(0,0);
corners[3]= cv::Point2f(obj_clr.cols-1, 0);
corners[1]= cv::Point2f(0, obj_clr.rows-1);
corners[2]= cv::Point2f(obj_clr.cols -1, obj_clr.rows -1);
persp_mat=cv::getPerspectiveTransform(corners , vertices );

cv::warpPerspective(obj_clr,temp,persp_mat,temp.size());

cv::Mat temp_gray,mask,invert;
cv::cvtColor(temp,temp_gray,CV_BGR2GRAY);
cv::threshold(temp_gray,mask,0,255,cv::THRESH_BINARY + cv::THRESH_OTSU);

cv::bitwise_not(mask,invert);

cv::bitwise_and(frame_clr, frame_clr,bg,invert);  //checkout :P
cv::bitwise_and(temp, temp, fg,mask);

cv::add(fg, bg, frame_clr);
//cv::equalizeHist( frame_clr, frame_clr);

 // DONE
}



int main(int argc, char** argv) {
    try {
        cv::VideoCapture cap(0);
        image_window win;
        dlib::image_window win2, win3;
        cv::Mat bg,fg;
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

              float lx=0,ly=0,rx=0,ry=0;
               std::vector<cv::Point> dest2;
               for(int k=36;k<=41;k++) //eye
               {
                 dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                 rx=rx+ shape.part(k).x();
                 ry=ry+ shape.part(k).y();
               }
                rx=rx/6;
                ry=ry/6;

               for(int k=42;k<=47;k++) //eye
               {
                 dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                 lx=lx+ shape.part(k).x();
                 ly=ly+ shape.part(k).y();
               }
                lx=lx/6; ly=ly/6;
                int cx,cy,ew,eh;
                cx=(lx+rx)/2;
                cy=(ly+ry)/2;

                float angle=(std::atan((ry-ly)/(rx-lx))*180/3.14);

            //    angle= -angle;
                //cout<<"\n again"<<angle;
                if (angle<0) angle=angle+360;
              //  float angle= ((std::acos(cos(std::atan(((-ry+ly)/(rx-lx)*180/3.14)))))*180/3.14);
                ew=(shape.part(45).x()+shape.part(16).x()-shape.part(36).x()-shape.part(0).x())/2;
                eh=(shape.part(19).y() + shape.part(24).y() - shape.part(46).y()-shape.part(41).y())/2;

                cv::Mat obj_clr2 =cv::imread("./res/images/specs2.png");
                mask(obj_clr2,cx,cy,ew,eh,angle,frame_clr,bg,fg);

                int mx=0,my=0,mh,mw;
                std::vector<cv::Point> dest1;
                for(int k=31;k<=35;k++) //moustache
                {
                  dest1.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                  mx=mx+ shape.part(k).x();
                  my=my+ shape.part(k).y();
                }
                for(int k=48;k<=54;k++)//moustache
                {
                  dest1.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                  mx=mx+ shape.part(k).x();
                  my=my+ shape.part(k).y();
                }
                  mx=mx/12;  my=my/12;
                cv::Mat obj_clr1 = cv::imread("./res/images/moustache.jpg");

                mh=(shape.part(30).y()+shape.part(33).y()-shape.part(51).y()- shape.part(62).y())/2;
                mw=shape.part(54).x() - shape.part(48).x();

               mask(obj_clr1,mx,my,mw,mh,angle,frame_clr,bg,fg);

               cv::Mat obj_clr4=cv::imread("./res/images/hat.png");
                               std::vector<cv::Point> dest4;

                               for(int k=0;k<=26;k++) //face
                               {
                                 dest4.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

                               }
                               cv::Rect face=cv::boundingRect(dest4);

                               int fh,fw,fx,fy;
                               fh=face.height;
                               fw=face.width;
                               fx=face.x;
                               fy=face.y;
                               fh=fh* 0.75;
                               //fx=fx+(fw/2)*sin(-angle*3.14/180);
                               //fy=fy-fh*sin(-angle*3.14/180)-15;
                              fx=fx+(fw/2);
                               fy=fy-fh-25;

                               //cv::Rect hatrect =cv::Rect(fx,fy-fh-30,fw,fh);    //check it out
                  //             mask2(obj_clr4,fx,fy,fw,fh,angle,frame_clr,bg,fg);

// Tasks left!! Find out why cap is upside down?
// why is the object colour inverted? go deep into the masking functions
// ADD a black maccha if reqd.
// your project is over! :D

          //     for(int k=17;k<=26;k++)    //eyebrow
          //     {
          //     dest2.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()+10));
          //
          //     }
        /*  int nx=0, ny=0, nw, nh;
               std::vector<cv::Point> dest3;
              for(int k=28;k<=35;k++)    //nosetip
               {
                 dest3.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                 nx=nx+ shape.part(k).x();
                 ny=ny+ shape.part(k).y();
               }
               nx=nx/8;
               ny=ny/8;
               nw = (shape.part(34).x()- shape.part(32).x())*1.2;
               nh = shape.part(28).y()- shape.part(33).y();
               cv::Mat obj_clr3=cv::imread("./res/images/clown_nose.jpg");
               mask(obj_clr3,nx,ny,nw,nh,angle,frame_clr,bg,fg);*/

              // cv::circle(frame_clr,cv::Point(shape.part(30).x(), shape.part(30).y()),20,cv::Scalar(0, 0, 255),-1);
                             std::vector<full_object_detection> shapes;
                              shapes.push_back(shape);

             }
            win.clear_overlay();
            win.set_image(cv_image<bgr_pixel>(frame_clr));// cv image , rgb image

            //win2.set_image(dlib::cv_image<bgr_pixel>(bg));
            //win3.set_image(dlib::cv_image<bgr_pixel>(fg));

            //win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(exception& e) {
      cout << e.what() << endl;
    }
}
