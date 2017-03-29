
#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_
#include "WorkerThread.h"
#include "ReqPackage.h"
#include <cstring>
#include <cstdlib>
#include <queue>
#include <opencv2/opencv.hpp>
#include "pthread.h"
#include "semaphore.h"
using namespace std;
using namespace cv;


class QThreadPool
{
public:
	QThreadPool(int numthread, int queuesize,int reqsemnum, void * (*run)(void *));
	~QThreadPool();
	bool pushreq(ReqPackage req);
	bool popreq(ReqPackage * req);
	bool pushres(void* id, ResPackage res);
	bool popres(void* id, ResPackage * res);
	bool checkReqState(int reqid);
private:
	//boost::mutex _Exe_mutex;
	//boost::condition_variable _cndExe;
	int _processNum;
	int _ReqQueueSize;
	queue<WorkerThread*> _queueExe;
	queue<ReqPackage> _queueReq;
	map<void*,ResPackage> _queueState;
	int _workingNum;
	int _waitingNum;
	//pthread_rwlock_t _ReqQueueRWlock;
	pthread_mutex_t *_ReqQueueLock;
	pthread_mutex_t *_StateQueueLock;
	sem_t *_ReqLockSem;
	sem_t *_StateLockSem;
};
#endif
