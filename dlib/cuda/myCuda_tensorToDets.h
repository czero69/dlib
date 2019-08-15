#include <vector>
#pragma once

//#ifdef MYCUDATENSOR2DETS_EXPORTS
//#define MYCUDATENSOR2DETS_API __declspec(dllexport)
//#else
//#define MYCUDATENSOR2DETS_API __declspec(dllimport)
//#endif
//#ifndef MYLIB_H
//#define MYLIB_H
//extern "C" MYCUDATENSOR2DETS_API 

#define MAX_RECTS 10000

typedef struct {
    int i; // linear index in tensor spaces
    int k;
    float score;
    float dx, dy, dw, dh;
    } MyRectAndOffset;

extern "C"
namespace dlib{
namespace cuda{

struct cudaDeviceScheduleMode { enum Choice {
    Spinning,
    Yield,
    Blocking
}; };
/// this struct is for data returned from device. General purpose here is to do not copy entire bitmaps from device to host.
/// and to copy only metadata such as rects centers and regressions shifts dx,dy,dw,dh
/// theese metadata are before tensor to image (input) coordinates transform

int runTest();
int get_rects_from_device2host(std::vector<MyRectAndOffset> & mr, const float* dev_data, const long int nr_m_nc, const size_t det_win_size, double scoreThreshold, bool useBB);
int set_flag_cudaDeviceSchedule(cudaDeviceScheduleMode::Choice scheduleChoice);
}
}
//# endif
