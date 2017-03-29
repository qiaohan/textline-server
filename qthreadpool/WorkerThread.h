/*************************************************************************
    > File Name: WorkerThread.h
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Fri 04 Nov 2016 05:44:34 PM CST
 ************************************************************************/
#ifndef _WORKERT_H_
#define _WORKERT_H_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

//class QThreadPool;

class WorkerThread
{
public:
	WorkerThread(const char *name, void * (*worker)(void *));
	int getStatus(void);
	//void run(const char *url,cv::Mat &src_mat, string &result,int flag);
	~WorkerThread(void);
private: 
	//static QThreadPool *threadpool;
	//static void* worker(void* ptr);
	pthread_t threads;
	int status; 
};

#endif
