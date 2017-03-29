/*************************************************************************
    > File Name: ../modelpredict/decode.h
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Wed 29 Mar 2017 01:42:37 PM CST
 ************************************************************************/
#ifndef __DECODE_H__
#define __DECODE_H__

#include<iostream>
using namespace std;

vector<int> decode_greedy(float* logits, int width, int length){
	vector<int> res;
	//cout<<"width:"<<width<<" length:"<<length<<endl;
	for(int i=0; i<length; i++)
	{
		int max = 0;
		for(int j=0; j<width; j++)
		{
			if( *(logits+width*i+j) > *(logits+width*i+max) )
				max = j;
		}
		if( res.empty()||*res.end() != max )
			res.push_back(max);
		//cout<<i<<endl;
	}
	return res;
}

#endif
