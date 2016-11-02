#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main()
{

	cout<<"welcome to mask function :P\n";
  	//rectangle(frame_clr,Point(rx-rw/2,ry-rh/2),Point(rxrw/2,ryrh/2),Scalar(0, 0, 255));
	Mat obj_clr= imread("clip.jpg",1);
	Mat frame_clr= imread("frontface.jpg",1);
	float angle=60;	int rx,ry,rw,rh; rx=200; 	ry=200;		rw=150;		rh=150;

	namedWindow("input",0);
	imshow("input",frame_clr);
	namedWindow("object to be masked",0);
	imshow("object to be masked",obj_clr);
	waitKey(0);

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

	namedWindow("output",0);
	imshow("output",frame_clr);
	while((char)waitKey(0)!='q');
	cout<<"\n byee! \n";
	return 0;
}
