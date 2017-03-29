#include "com_netease_mm_image_jni_OCR.h"
#include "threadPool.h"
#include "ReqPackage.h"
#include <unistd.h>
#include <iostream>
#include "jpeg2Mat.h"
#include "TextLineOCR.h"
#include "cJSON.h"

QThreadPool * ptrCThreadPool;

void* worker(void* ptr)
{
	/*
	char buf[65536];
	string tags_file="/home/asr/hzwenxiang/lib/lofter_15.txt";
	std::vector<pair<string,int> > tags;
	std::ifstream infile(tags_file.c_str());
	string label;
	int id=0;
	while(infile >> label >> id)
	{
		tags.push_back(make_pair(label.c_str(),id));
	}
	*/
	ReqPackage * req = new ReqPackage();
	TextLineReader tlinereader;
	cout<<"finish init!"<<endl;
	while (true)
	{
		if(ptrCThreadPool->popreq(req))
		{
			cout<<"start process..."<<endl;	
			vector<int> sentence = tlinereader.read(req->src_mat);
			ResPackage r;
			r.chars = sentence;
			ptrCThreadPool->pushres(req->id,r);			
			//sleep(0.1);
		}
	}
	return 0;
}

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

//Mat mm;

/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    init
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_netease_mm_image_jni_TextDetect_init
  (JNIEnv * env, jclass, jstring thnum){
	const char* pthnum = env->GetStringUTFChars(thnum, NULL);
	int threadNum = atoi(pthnum);
	ptrCThreadPool = new QThreadPool(30, 1000 ,1, worker);
	std::cout << "init succ!" << std::endl;
	return true;
}

/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    detect
 * Signature: ([B)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_mm_image_jni_TextDetect_detect
  (JNIEnv * env, jclass, jbyteArray src){
	uint8_t * data = (uint8_t*)env->GetByteArrayElements(src, 0);
	int len = env->GetArrayLength(src);
	//cout<<src<<'\t'<<len<<endl;
	/*
	vector<uchar> mm(len);
	for(int i=0;i<len;i++)
		mm[i]=data[i];
	cout<<"mat"<<endl;
	InputArray ina(mm);
	cout<<"inputarray"<<endl;
	Mat src_mat = imdecode( ina,CV_LOAD_IMAGE_COLOR);
	void * src1;
	unsigned long osize;
	int w,h;
	Jpeg2DIB_DeCompress(data,len,&src1,&osize,&w,&h);
	Mat src_mat(h,w,CV_8UC3,src1);
	//Mat src_mat = mm;
	*/
	Mat src_mat = Jpeg2Mat(data,len);
	ReqPackage * req = new ReqPackage();
	ResPackage r;
	req->src_mat = src_mat;
	req->id = (void*)req;
	ptrCThreadPool->pushreq(*req);
	while(!ptrCThreadPool->popres(req->id,&r))
		sleep(0.1);
	//sleep(5);
	cout<<"get response"<<endl;
	delete req;
	//free(src1);
	string res=ReturnCheckStr(r.chars);
	char *buf;
	buf = (char*)malloc((res.length() + 20)*sizeof(char));
	memset(buf, 0, (res.length() + 20)*sizeof(char));
	sprintf(buf, "%s", res.c_str());
	jstring outString = env->NewStringUTF(buf);
	free(buf);
	buf = NULL;
	return outString;
}

