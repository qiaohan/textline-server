/* DO NOT EDIT THIS FILE - it is machine generated */
#include "jni.h"
/* Header for class com_netease_mm_image_jni_SpamDetect */

#ifndef _Included_com_netease_mm_image_jni_SpamDetect
#define _Included_com_netease_mm_image_jni_SpamDetect
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    init
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_netease_mm_image_jni_TextDetect2_init
  (JNIEnv *, jclass, jstring);

/*
 * Class:     com_netease_mm_image_jni_SpamDetect
 * Method:    detect
 * Signature: ([B)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_mm_image_jni_TextDetect2_detect
	(JNIEnv * env, jclass jc, jstring impath);
#ifdef __cplusplus
}
#endif
#endif

