// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to run a CNN based vehicle detector using dlib.  The
    example loads a pretrained model and uses it to find the front and rear ends
    of cars in an image.  The model used by this example was trained by the
    dnn_mmod_train_find_cars_ex.cpp example program on this dataset:
        http://dlib.net/files/data/dlib_front_and_rear_vehicles_v1.tar

    Users who are just learning about dlib's deep learning API should read
    the dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples to learn
    how the API works.  For an introduction to the object detection method you
    should read dnn_mmod_ex.cpp.

    You can also see a video of this vehicle detector running on YouTube:
        https://www.youtube.com/watch?v=OHbJ7HhbG74
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>


#include "myUtils.h"

using namespace std;
using namespace dlib;



// shrinked model5:
template <long num_filters, typename SUBNET> using myCon2 = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<bn_con<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<4>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    // net_type net;
    shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, the file also includes a separately trained shape_predictor.  To see
    // a generic example of how to train those refer to train_shape_predictor_ex.cpp.

    if (argc != 4)
    {
        cout << "   ./dnn_test_detector detector.dat train-images-test.xml test-images-test.xml" << endl;
        cout << endl;
        return 0;
    }

    const std::string detectorPath = argv[1];
    const std::string training_filepath = argv[2];
    const std::string testing_filepath = argv[3];

    std::vector<matrix<dlib::rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> boxes_train, boxes_test;

    std::vector<std::string> parts_list_train;
    std::vector<std::string> parts_list_test;

    //std::vector<std::vector<std::pair<dlib::mmod_rect, std::vector<dlib::point>>>> boxesPartsTrain, boxesPartsTest;

    // loads images to RAM, downsacles them, then loads images to RAM to fit all available space
    imageLoader myImgLoader(training_filepath, testing_filepath, 30, 45, 0.9, 0.40, 2592/2, 2048/2, 2);

    // init load
    myImgLoader(images_train, images_test, boxes_train, boxes_test, parts_list_train, parts_list_test);

    mmod_options options(boxes_train, 60, 15);
    options.overlaps_ignore = test_box_overlap(0.5, 0.95);
    options.use_bounding_box_regression = true;

    net_type net(options);
    deserialize(detectorPath) >> net;

    cout << "num training images: "<< images_train.size() << endl;
    cout << "training results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);

    cout << "num testing images: "<< images_test.size() << endl;
    cout << "testing results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);
    upsample_image_dataset<pyramid_down<2>>(images_test, boxes_test, 1200*1200);
    cout << "testing upsampled results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);

    return 0;
}
catch(image_load_error& e)
{
    cout << e.what() << endl;
    cout << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The correct model file can be obtained from: http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2" << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
