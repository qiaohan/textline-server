#include "com_netease_mm_image_jni_OCR.h"
#include <unistd.h>
#include <iostream>
#include "jpeg2Mat.h"
#include "TextLineOCR.h"
#include "cJSON.h"
#include <stack>
#include "semaphore.h"
#include<time.h>
#include<sys/time.h>

string ReturnCheckStr(vector<int> sentence)
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
	//buf<<feature[2];
	buf<<";";
	buf<<"youwenzi:";
	//buf<<feature[1]+feature[0];
	buf<<";";
	buf<<"guanggao:";
	//buf<<feature[0];
	buf<<";";
	buf<<"feiguanggao:";
	//buf<<feature[1];
	string result_str=buf.str();
	cJSON_AddItemToObject(root,"result",cJSON_CreateString(result_str.c_str()));
	out=cJSON_Print(root);
	cJSON_Delete(root);
	return out;
}

stack< TextLineReader<float> * >  tlinereaders;
sem_t *_ReqSem;
pthread_mutex_t *_ReqLock;
/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    init
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_netease_mm_image_jni_TextDetect2_init
  (JNIEnv * env, jclass, jstring thnum){
	std::cout << "init start!" << std::endl;
	const char* pthnum = env->GetStringUTFChars(thnum, NULL);
	int threadNum = atoi(pthnum);
	threadNum = 15;
	_ReqSem = new sem_t();
	_ReqLock = new pthread_mutex_t();
	pthread_mutex_init(_ReqLock,NULL);
	sem_init(_ReqSem,0,threadNum);
	for(int i=0; i<threadNum; i++)
		tlinereaders.push( new TextLineReader<float>() );
	std::cout << "init succ!" << std::endl;
	return true;
}

/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    detect
 * Signature: ([B)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_mm_image_jni_TextDetect2_detect
  (JNIEnv * env, jclass, 
	//jbyteArray src){
	jstring impath){
	/*
	uint8_t * data = (uint8_t*)env->GetByteArrayElements(src, 0);
	int len = env->GetArrayLength(src);
	Mat src_mat = Jpeg2Mat(data,len);
	*/

		struct timeval t1,t2;
		double timeuse;
		gettimeofday(&t1,NULL);
	sem_wait(_ReqSem);
		gettimeofday(&t2,NULL);
		timeuse=(t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000;
		cout<<"get sem:"<<timeuse<<endl;
	Mat src_mat;// = cv::imread(env->GetStringUTFChars(impath,NULL));
	pthread_mutex_lock(_ReqLock);
	TextLineReader<float> * reader = tlinereaders.top();
	cout<<"pre NUM:"<<tlinereaders.size()<<endl;
	tlinereaders.pop();
	pthread_mutex_unlock(_ReqLock);
		gettimeofday(&t1,NULL);
		timeuse=(t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000;
		cout<<"get process lock:"<<-timeuse<<endl;

	vector<int> residx = reader->read(src_mat); 
	string res=ReturnCheckStr( residx );
	char *buf;
	buf = (char*)malloc((res.length() + 20)*sizeof(char));
	memset(buf, 0, (res.length() + 20)*sizeof(char));
	sprintf(buf, "%s", res.c_str());
	jstring outString = env->NewStringUTF(buf);
	free(buf);
	buf = NULL;
		gettimeofday(&t2,NULL);
		timeuse=(t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000;
		cout<<"get res:"<<timeuse<<endl;

	pthread_mutex_lock(_ReqLock);
	tlinereaders.push(reader);
	cout<<"post NUM:"<<tlinereaders.size()<<endl;
	pthread_mutex_unlock(_ReqLock);
		gettimeofday(&t1,NULL);
		timeuse=(t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/1000000;
		cout<<"get release lock:"<<-timeuse<<endl;

	sem_post(_ReqSem);
	return outString;
}

