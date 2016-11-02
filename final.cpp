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
using namespace cv;

void mask(cv::Mat obj_clr,int rx,int ry,int rw, int rh ,float angle, cv::Mat frame_clr, cv::Mat bg, cv::Mat fg)
{
  	Mat toCheckIfClipIsWhite = obj_clr.clone();
  	threshold(toCheckIfClipIsWhite, toCheckIfClipIsWhite, 240, 255, THRESH_BINARY);			//Because some useless guy may put slightly greyish white for background :P
  	bool clipIsWhite = false;
  	cout<<"CLIPISWHITE: "<<clipIsWhite<<"\n\n\n";
  	if(toCheckIfClipIsWhite.at<Vec3b>(1,0)[0] == 255 && toCheckIfClipIsWhite.at<Vec3b>(1,0)[1] == 255 && toCheckIfClipIsWhite.at<Vec3b>(1,0)[2] == 255)
  	{
  		bitwise_not(obj_clr,obj_clr);
  		clipIsWhite=true;

  	/******************************************************REMOVE THIS LINE IN THE FINAL VERSION************************/
  	}


  	Mat persp_mat(3,3,CV_32FC1);
  	RotatedRect rect=RotatedRect(Point(rx,ry),Size(rw,rh), angle);

  			Point2f vertices[4]; // roi corners
  			rect.points(vertices);
  			Point2f corners[4]; //object corners

  			corners[0]= Point2f(0,0);
  			corners[3]= Point2f(obj_clr.cols-1, 0);
  			corners[1]= Point2f(0, obj_clr.rows-1);
  			corners[2]= Point2f(obj_clr.cols -1, obj_clr.rows -1);

  			persp_mat=getPerspectiveTransform(corners , vertices );

  /************************Initializes it to white...no need to do the invert you were doing before :P***********************/

  	Mat bg = frame_clr.clone(), fg;
  	if(clipIsWhite==true)
  		fg=Mat(frame_clr.rows, frame_clr.cols, CV_8UC3, Scalar(255,255,255));
  	else
  		fg=Mat(frame_clr.rows, frame_clr.cols, CV_8UC3, Scalar(0,0,0));
  	warpPerspective(obj_clr, fg, persp_mat, fg.size());

  	namedWindow("mask", 0); imshow("mask", fg);
  	if(clipIsWhite==true)
  		subtract(bg, fg, frame_clr);
  	else		//If the clip is black
  	{
  		//Make a black hole in bg wherever there's content in the fg
  		Mat temp = fg.clone();
  		threshold(temp, temp, 2, 255, THRESH_BINARY);
  		subtract(bg, temp, bg);
  		//Finally add the two images
  		add(fg, bg, frame_clr);
  	}

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
