/*************************************************************************
    > File Name: threadPool.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Fri 04 Nov 2016 05:40:33 PM CST
 ************************************************************************/

#include<iostream>
#include "threadPool.h"
#include "WorkerThread.h"
#include <unistd.h>

using namespace std;

QThreadPool::QThreadPool(int threadNum, int queuesize,int reqsemnum, void *(*run)(void *))
{
	if(run==NULL)
	cout<<"invalid thread function"<<endl;
	_ReqQueueLock = new pthread_mutex_t();
	_StateQueueLock = new pthread_mutex_t();
	pthread_mutex_init(_ReqQueueLock,NULL);
	pthread_mutex_init(_StateQueueLock,NULL);
	_ReqLockSem = new sem_t();
	_StateLockSem = new sem_t();
	sem_init(_ReqLockSem,0,reqsemnum);
	sem_init(_StateLockSem,0,reqsemnum);
	
	//pthread_rwlock_init(&_ReqQueueRWlock,NULL);
	/*
	 * request queue init
	*/
	
	_ReqQueueSize = queuesize;
	_ReqQueueSize = (_ReqQueueSize > 2) ? _ReqQueueSize : 2;
	_ReqQueueSize = (_ReqQueueSize < 600) ? _ReqQueueSize : 600;
	_waitingNum = 0;
	/*
	 * worker thread init
	*/
	_processNum = threadNum;
	_processNum = (_processNum > 1) ? _processNum : 1;
	_processNum = (_processNum < 150) ? _processNum : 150;

	for (int i = 0; i < _processNum; ++i)
	{
		std::cout << "QThreadPool::QThreadPool(int threadNum) before create>>>>>>>>>>>>>" << std::endl;
		WorkerThread *p = new WorkerThread("modelPath", run);
		std::cout << "QThreadPool::QThreadPool(int threadNum) after create>>>>>>>>>>>>>" << std::endl;
		_queueExe.push(p);
	}
	_workingNum = 0;
	//cout << "threadPool: inited, containing "<<_processNum<< " threads."<<endl;
	usleep(100);
}
QThreadPool::~QThreadPool()
{
	//pthread_rwlock_destroy(&_ReqQueueRWlock);
	pthread_mutex_destroy(_ReqQueueLock);
	pthread_mutex_destroy(_StateQueueLock);
	delete _ReqQueueLock;
	delete _StateQueueLock;
	sem_destroy(_ReqLockSem);
	sem_destroy(_StateLockSem);
	delete _ReqLockSem;
	delete _StateLockSem;
	while (_queueExe.size() > 0)
	{
		WorkerThread *p;
		delete _queueExe.front();
		_queueExe.pop();
	}
}

bool QThreadPool::pushreq(ReqPackage req)
{
	sem_wait(_ReqLockSem);
	//pthread_rwlock_wrlock(&_ReqQueueRWlock);
	if(_queueReq.size()>=_ReqQueueSize){
		sem_post(_ReqLockSem);	
		return false;
	}
	pthread_mutex_lock(_ReqQueueLock);
	if(_queueReq.size()>=_ReqQueueSize){
		//pthread_rwlock_unlock(&_ReqQueueRWlock);
		pthread_mutex_unlock(_ReqQueueLock);
		sem_post(_ReqLockSem);
		return false;
	}
	else{
		_queueReq.push(req);
		//_ReqState.insert(pair<ReqPackage*,bool>(req,false));
		//cout<<"waiting for fetch NUM:"<<_ReqState.size()<<endl;
		pthread_mutex_unlock(_ReqQueueLock);
		sem_post(_ReqLockSem);
		return true;
	}
}

bool QThreadPool::popreq(ReqPackage * req)
{
	static bool acc = false;
	pthread_mutex_lock(_ReqQueueLock);
	if(_queueReq.empty()){
		pthread_mutex_unlock(_ReqQueueLock);
		if(acc)
			sleep(0.2);
		else
			sleep(2);
		acc = false;
		return false;
	}
	else{
		acc = true;
		cout<<"waiting for process NUM:"<<_queueReq.size()<<endl;
		*req = _queueReq.front();
		_queueReq.pop();
		pthread_mutex_unlock(_ReqQueueLock);
		return true;
	}
}

bool QThreadPool::pushres(void* id, ResPackage res)
{
	pthread_mutex_lock(_StateQueueLock);
	_queueState.insert(pair<void*,ResPackage>(id,res));
	pthread_mutex_unlock(_StateQueueLock);
	return true;
}

bool QThreadPool::popres(void* id, ResPackage * res)
{
	sem_wait(_StateLockSem);
	pthread_mutex_lock(_StateQueueLock);
	map<void*, ResPackage>::iterator it = _queueState.find(id);
	if(it==_queueState.end()){
		sem_post(_StateLockSem);
		pthread_mutex_unlock(_StateQueueLock);
		sleep(0.1);
		return false;
	}else{
		*res = it->second;
		_queueState.erase(it);
		sem_post(_StateLockSem);
		pthread_mutex_unlock(_StateQueueLock);
		return true;
	}
}

/*
bool QThreadPool::checkReqState(ReqPackage* req)
{
	bool done;
	pthread_rwlock_rdlock(&_ReqQueueRWlock);
	auto it = _ReqState.find(req);
	if(it==_ReqState.end())
		return false;
	done = it->second;
	pthread_rwlock_unlock(&_ReqQueueRWlock);
	if(done)
	{
		pthread_rwlock_wrlock(&_ReqQueueRWlock);
		_ReqState.erase(it);
		pthread_rwlock_unlock(&_ReqQueueRWlock);
		return true;
	}else{
		return false;
	}
}
*/
