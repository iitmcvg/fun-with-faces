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

//imshow("mask",mask);
//imshow("invert",invert);
cv::bitwise_and(frame_clr, frame_clr,bg,invert);  //checkout :P
cv::bitwise_and(~temp, ~temp, fg,mask);

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


// imshow("mask",mask);
// imshow("invert",invert);
// imshow("temp_gray",temp_gray);
// imshow("temp",temp);
cv::waitKey(1);

cv::bitwise_and(frame_clr, frame_clr,bg,invert);  //checkout :P
cv::bitwise_and(~temp, ~temp, fg, mask);

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
                cv::Mat obj_clr1 = cv::imread("./res/images/3d.png");

                mh=(shape.part(30).y()+shape.part(33).y()-shape.part(51).y()- shape.part(62).y())/2;
                mw=shape.part(54).x() - shape.part(48).x();

               mask(obj_clr1,mx,my,mw,mh,angle,frame_clr,bg,fg);

               cv::Mat obj_clr4=cv::imread("./res/images/hat1.jpg");
                               std::vector<cv::Point> dest4;

                cv::RotatedRect Ellipse;
                std::vector<cv::Point> destface;
                    for(int p = 0;p <=67 ; p++)
                    {
                      destface.push_back(cv::Point(shape.part(p).x(), shape.part(p).y()));
                    }
                Ellipse = cv::fitEllipse( cv::Mat(destface));

                int xc = Ellipse.center.x;
                int yc = Ellipse.center.y;
                float b = Ellipse.size.width/ 2;
                float a = Ellipse.size.height/ 2;
                float theta = Ellipse.angle;

              //  cout<<" angle" <<theta;
                theta = theta/180*3.14;
                float fh,fw,fx,fy;
            //    fy = yc - sin(theta) * 2* a;
              //  fx = xc - cos(theta) * 2* b;
            //  fy =( yc + 2*a)*sin(theta);
            //  fx = (xc + 2*b)*cos(theta);
            //    fw = 2*b;
            //    fh = 2*a;
            //     cv::line(frame_clr, s, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
            //Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
                cv::ellipse(frame_clr, Ellipse, cv::Scalar(0,255,0), 2, 8 );
                               for(int k=0;k<=26;k++) //face
                               {
                                 dest4.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));

                               }
                               cv::Rect face=cv::boundingRect(dest4);
//fy = face.y - sin(theta)*b;
//fx = face.x + cos(theta)*a;
cv::line(frame_clr, cv::Point(xc,yc), cv::Point(fx,fy),  cv::Scalar(255,0,0),1, 8);
cv::line(frame_clr, cv::Point(shape.part(8).x(),shape.part(8).y()), cv::Point(shape.part(27).x(),shape.part(27).y()), cv::Scalar(0,255,0),1,8);

                               fh=face.height;
                               fw=face.width;
                               fx=face.x;
                               fy=face.y;
                               fh=fh* 0.75;
                               fx=fx+(fw/2)*sin(-angle*3.14/180);
                               fy=fy-fh*sin(-angle*3.14/180)-15;
                              fx=fx+(fw/2);
                              fy=fy-fh+40;
                               cv::Rect hatrect =cv::Rect(fx,fy-fh+30,fw,fh);    //check it out
                               mask2(obj_clr4,fx,fy,fw,fh,angle,frame_clr,bg,fg);


          int nx=0, ny=0, nw, nh;
               std::vector<cv::Point> dest3;
              for(int k=28;k<=35;k++)    //nosetip
               {
                 dest3.push_back(cv::Point(shape.part(k).x(),shape.part(k).y()));
                 nx=nx+ shape.part(k).x();
                 ny=ny+ shape.part(k).y();
               }
               nx=nx/8;
               ny=ny/8;
               nw = (shape.part(34).x()- shape.part(32).x());
               nh = (shape.part(28).y()- shape.part(33).y());
               cv::Mat obj_clr3=cv::imread("./res/images/clown_nose.jpg");
               mask(obj_clr3,nx,ny,nw,nh,angle,frame_clr,bg,fg);

                             std::vector<full_object_detection> shapes;
                              shapes.push_back(shape);

          cv::Mat framepic =cv::imread("./res/images/frame2.jpg");
          mask(framepic,frame_clr.cols/2, frame_clr.rows/2, frame_clr.cols,frame_clr.rows,0,frame_clr, bg, fg );

        }
            win.clear_overlay();
            win.set_image(cv_image<bgr_pixel>(frame_clr));
        }
    }

catch(exception& e) {
      cout << e.what() << endl;
    }
}
