#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

using namespace cv;
using namespace std;


int main()
{
   
	std::cout<<"welcome to mask function :P\n";
  	//cv::rectangle(frame_clr,cv::Point(rx-rw/2,ry-rh/2),cv::Point(rx+rw/2,ry+rh/2),cv::Scalar(0, 0, 255));
	cv::Mat obj_clr= imread("clip.jpg",1);
	cv::Mat frame_clr= imread("frontface.jpg",1);
	float angle=60;	int rx,ry,rw,rh; rx=200; 	ry=200;		rw=150;		rh=150;
	
	namedWindow("input",0);
	imshow("input",frame_clr);
	namedWindow("object to be masked",0);
	imshow("object to be masked",obj_clr);
	waitKey(0);

	cv::Mat check = obj_clr.clone();
	threshold(check,check,240,255,THRESH_BINARY);
	bool ClipIsWhite = false;
	cout<<"clip is white"<<ClipIsWhite<<"\n\n";

	if(check.at<Vec3b>(1,0)[0] == 255 && check.at<Vec3b>(1,0)[1] == 255 && check.at<Vec3b>(1,0)[2] == 255 )
	{
				cv::bitwise_not(obj_clr,obj_clr);
				ClipIsWhite=true;
	}
																imshow("obj after not",obj_clr);
																waitKey(0);
	
	cv::Mat persp_mat(3,3,CV_32FC1);
	cv::RotatedRect rect=cv::RotatedRect(cv::Point(rx,ry),cv::Size(rw,rh), angle);

			cv::Point2f vertices[4]; // roi corners
			rect.points(vertices);
			cv::Point2f corners[4]; //object corners
	
			corners[1]= cv::Point2f(0,0);
			corners[2]= cv::Point2f(obj_clr.cols-1, 0);
			corners[0]= cv::Point2f(0, obj_clr.rows-1);
			corners[3]= cv::Point2f(obj_clr.cols -1, obj_clr.rows -1);

	persp_mat=cv::getPerspectiveTransform(corners , vertices );

	cv::Mat bg,fg;

	bg = frame_clr.clone();

	if(ClipIsWhite == true)
		fg = cv::Mat(frame_clr.rows, frame_clr.cols, CV_8UC3, cv::Scalar(255,255,255));
	else
		fg = cv::Mat(frame_clr.rows, frame_clr.cols, CV_8UC3, cv::Scalar(0,0,0));

	warpPerspective(obj_clr, fg, persp_mat, fg.size());

	namedWindow("mask-fg",0); 
	cv::imshow("mask-fg",fg);

	if(ClipIsWhite == true)
		subtract(bg, fg, frame_clr);
	else
	{
		//A black hole in bg wherever there is content in fg
		cv::Mat temp = fg.clone();
		threshold(temp, temp, 2, 255, THRESH_BINARY);
		subtract(bg, temp, bg);
		add(fg, bg, frame_clr);
	}
namedWindow("output",0);
cv::imshow("output",frame_clr);
cv::waitKey(0);
while((char)waitKey(0)!='q');
cout<<"\n byee! \n";
return 0;
}