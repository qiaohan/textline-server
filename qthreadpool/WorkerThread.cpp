/*************************************************************************
    > File Name: WorkerThread.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Fri 04 Nov 2016 05:45:47 PM CST
 ************************************************************************/

#include<iostream>
#include "WorkerThread.h"
#include "threadPool.h"
#include "ReqPackage.h"
#include "unistd.h"
using namespace std;

WorkerThread::WorkerThread(const char *name, void *(*worker) (void *))
{
	status = 1; //idle
	std::cout << "WorkerThread::WorkerThread(const char *name)<<<<<<<<<< before create" << std::endl;
	if(worker)
	{
		int ret = pthread_create(&threads, NULL, worker, NULL);
		if(0!=ret){
			cout<<"thread create failed"<<endl;
			abort();
		}
	}
	std::cout << "WorkerThread::WorkerThread(const char *name)<<<<<<<<<< after create" << std::endl;
	usleep(100);
}

int WorkerThread::getStatus(void)
{
	return status;
}

WorkerThread::~WorkerThread(void)
{
	usleep(200);
}
/*

void* WorkerThread:: worker(void* ptr)
{
	int countWork = -1;
	char buf[65536];
	//string model_file = "/home/asr/hzwenxiang/lib/ResNet-50-deploy-new.prototxt";
	//string model_file1 = "/home/asr/hzwenxiang/lib/ResNet-50-deploy-fanlaji.prototxt";
	//string trained_file = "/home/asr/hzwenxiang/lib/newfanlaji_ResNet_iter_3600.caffemodel";
	//string trained_file1 = "/home/asr/hzwenxiang/lib/fanlaji_ResNet_iter_91000.caffemodel";
	//string model_file = "/home/asr/hzwenxiang/lib/KM_fanlaji_new.prototxt";
        //string model_file1 = "/home/asr/hzwenxiang/lib/KM_fanlaji_1_new.prototxt";
        //string trained_file = "/home/asr/hzwenxiang/lib/newfanlaji_KMnet_iter_20591.caffemodel";
        //string trained_file1 = "/home/asr/hzwenxiang/lib/newfanlaji_1_KMnet_iter_8496.caffemodel";
	string model_file="/home/asr/hzwenxiang/lib/KM_fanlaji_2_new.prototxt";
    string trained_file="/home/asr/hzwenxiang/lib/newfanlaji_2_KMnet_iter_40000.caffemodel";

	string tags_file="/home/asr/hzwenxiang/lib/lofter_15.txt";
	std::vector<pair<string,int> > tags;
	std::ifstream infile(tags_file.c_str());
	string label;
	int id=0;
	while(infile >> label >> id)
	{
		tags.push_back(make_pair(label.c_str(),id));
	}
	Mat img;
	FeatureExtract extract;
	boost::shared_ptr<Net<float> > Net;
	//Nets=extract.init_net(model_file1,trained_file1);
    	Net=extract.init_net(model_file,trained_file);
	ReqPackage * req;
	while (true)
	{
		if(threadpool->popreq(req))
		{	// ?
			countWork++;
			
			std::vector<cv::Rect> dets;
			std::vector<cv::Rect> det_str;
			int load_status = 1;
			
			try
			{
				img = req->src_mat;//imread(para->imgPath);
			}
			catch (exception& ee)
			{
				load_status = 0;
				sprintf(buf, "{ \"status\": -2}");
			}

			if (load_status == 0 || img.cols <= 0)
			{
				sprintf(buf, "{ \"status\": -2}");
			}
			else
			{							
				vector<float> feature(3,0.0);
				feature=extract.predict_image(img,Net);
			}
		}
	}
	return 0;
}

*/
/*
string ReturnCheckStr(std::vector<float> &feature,std::vector<float> &feature1)
{
	string out;
	cJSON *root,*status,*result;
	root=cJSON_CreateObject();
	cJSON_AddNumberToObject(root,"status",1);
	result=cJSON_CreateArray();
	std::stringstream buf;
	buf.str(std::string());
	buf.clear();
	buf<<"meiwenzi:";
	buf<<feature1[0];
	buf<<";";
	buf<<"youwenzi:";
	buf<<feature1[1];
	buf<<";";
	buf<<"guanggao:";
	buf<<feature[0];
	buf<<";";
	buf<<"feiguanggao:";
	buf<<feature[1];
	string result_str=buf.str();
	cJSON_AddItemToObject(root,"result",cJSON_CreateString(result_str.c_str()));
	out=cJSON_Print(root);
	cJSON_Delete(root);
	return out;
}
*/
