#ifndef _JPGMAT_
#define _JPGMAT_

#include "turbojpeg.h"
#include <opencv2/core/core.hpp>
#include <tuple>
#include <vector>
#include <fstream>

/**
* decode JPEG format to XXX format
* @param pJpegData
* @param JpegdataSize
* @param convrt_flag
*        "RGB" - see TJPF_RGB
*         "BGR" - see TJPF_BGR
*          ...
*  see all support format at enum TJPF
*/
std::tuple<bool,std::vector<uint8_t>,uint64_t,uint64_t,uint64_t> decodeJpeg2X(uint8_t* pJpegData,uint64_t JpegdataSize,const char* convrt_flag)
{
    assert( pJpegData != NULL );
    int width = 0,height = 0,jpegsubsamp = 0;

    tjhandle jpeg = tjInitDecompress();

    if(jpeg == nullptr)
    {
        return std::make_tuple(false, std::vector<uint8_t>(0), 0, 0, 0); 
    }

    if(tjDecompressHeader2(jpeg,pJpegData,JpegdataSize,&width,&height,&jpegsubsamp) != 0)
    {
        return std::make_tuple(false, std::vector<uint8_t>(0), 0, 0, 0);
    }

    TJPF eformat;
    if(convrt_flag == "ABGR")
        eformat = TJPF::TJPF_ABGR;
    else if(convrt_flag == "ARGB")
        eformat = TJPF::TJPF_ARGB;
    else if(convrt_flag == "BGR")
        eformat = TJPF::TJPF_BGR;
    else if(convrt_flag == "BGRA")
        eformat = TJPF::TJPF_BGRA;
    else if(convrt_flag == "BGRX")
        eformat = TJPF::TJPF_BGRX;
    else if(convrt_flag == "CMYK")
        eformat = TJPF::TJPF_CMYK;
    else if(convrt_flag == "GRAY")
        eformat = TJPF::TJPF_GRAY;
    else if(convrt_flag == "RGB")
        eformat = TJPF::TJPF_RGB;
    else if(convrt_flag == "RGBA")
        eformat = TJPF::TJPF_RGBA;
    else if(convrt_flag == "RGBX")
        eformat = TJPF::TJPF_RGBX;
    else if(convrt_flag == "XBGR")
        eformat = TJPF::TJPF_XBGR;
    else if(convrt_flag == "XRGB")
        eformat = TJPF::TJPF_XRGB;

    uint64_t pitch = tjPixelSize[eformat] * width;
    uint64_t size = pitch * height;
    std::vector<uint8_t> output(size);

    if(tjDecompress2(jpeg,pJpegData,JpegdataSize,&output.front(),width,pitch,height,eformat,0) != 0)
    {
        return std::make_tuple(false, std::vector<uint8_t>(0), 0, 0, 0);
    }

    return std::make_tuple(true, std::move(output), size, width, height);
}

std::tuple<bool,std::vector<uint8_t>,uint64_t,uint64_t,uint64_t> decodeJpeg2X(std::string filename,const char* convrt_flag)
{
    std::ifstream ifs(filename.c_str(),std::ios_base::binary | std::ios_base::in);

    if(!ifs.good())
        return std::make_tuple(false, std::vector<uint8_t>(0), 0, 0, 0);
    ifs.seekg(0,std::ios::end);
    uint64_t size = ifs.tellg();
    ifs.seekg(0,std::ios::beg);

    std::vector<char> buffer(size);
    ifs.read(&buffer.front(),size);

    return decodeJpeg2X((uint8_t*)&buffer.front(),size,convrt_flag);
}

cv::Mat Jpeg2Mat(uint8_t *jpegData, uint64_t jpegSize)
{
    auto res = decodeJpeg2X( (uint8_t*)jpegData,jpegSize,"BGR");
    bool success = false;
    std::vector<uint8_t> buff;
    int width,height,size;

    std::tie(success,buff,size,width,height) = res;
    Mat dst(height,width,CV_8UC3,(uint8_t*)&buff.front());
    return dst.clone();
}

cv::Mat Jpeg2Mat(std::string filename)
{
    auto res = decodeJpeg2X( filename ,"BGR");
    bool success = false;
    std::vector<uint8_t> buff;
    int width,height,size;

    std::tie(success,buff,size,width,height) = res;

    Mat dst(height,width,CV_8UC3,(uint8_t*)&buff.front());
    return dst.clone();
}
#endif
