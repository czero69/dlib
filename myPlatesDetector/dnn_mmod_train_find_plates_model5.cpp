// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a CNN based object detector using dlib's 
    loss_mmod loss layer.  This loss layer implements the Max-Margin Object
    Detection loss as described in the paper:
        Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046).
    This is the same loss used by the popular SVM+HOG object detector in dlib
    (see fhog_object_detector_ex.cpp) except here we replace the HOG features
    with a CNN and train the entire detector end-to-end.  This allows us to make
    much more powerful detectors.

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.  You should also read the introductory DNN+MMOD
    example dnn_mmod_ex.cpp as well before proceeding.
    

    This example is essentially a more complex version of dnn_mmod_ex.cpp.  In it we train
    a detector that finds the rear ends of motor vehicles.  I will also discuss some
    aspects of data preparation useful when training this kind of detector.  
    
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

#include <dlib/gui_widgets.h> // just for testing

#include "myUtils.h"
#include "dlib/data_io/image_dataset_metadata.h"

#include <utility>

using namespace std;
using namespace dlib;

bool DEBUG_IMSHOW = 0;


// shrinked model5:
template <long num_filters, typename SUBNET> using myCon2 = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<bn_con<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<4>>>>>>;


// ----------------------------------------------------------------------------------------

int ignore_overlapped_boxes(
    std::vector<mmod_rect>& boxes,
    const test_box_overlap& overlaps
)
/*!
    ensures
        - Whenever two rectangles in boxes overlap, according to overlaps(), we set the
          smallest box to ignore.
        - returns the number of newly ignored boxes.
!*/
{
    int num_ignored = 0;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i].ignore)
            continue;
        for (size_t j = i+1; j < boxes.size(); ++j)
        {
            if (boxes[j].ignore)
                continue;
            if (overlaps(boxes[i], boxes[j]))
            {
                ++num_ignored;
                if(boxes[i].rect.area() < boxes[j].rect.area())
                    boxes[i].ignore = true;
                else
                    boxes[j].ignore = true;
            }
        }
    }
    return num_ignored;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{    
    if (argc != 3)
    {
        cout << "   ./dnn_mmod_train_plates_classic-dlib-model training.xml testing.xml" << endl;
        cout << endl;
        cout << "It takes about a day to finish if run on a high end GPU like a 1080ti." << endl;
        cout << endl;
        return 0;
    }

    const std::string training_filepath = argv[1];
    const std::string testing_filepath = argv[2];


    std::vector<matrix<dlib::rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> boxes_train, boxes_test;
    std::vector<std::string> parts_list_train;
    std::vector<std::string> parts_list_test;

    //std::vector<std::vector<std::pair<dlib::mmod_rect, std::vector<dlib::point>>>> boxesPartsTrain, boxesPartsTest;

    // loads images to RAM, downsacles them, then loads images to RAM to fit all available space
    imageLoader myImgLoader(training_filepath, testing_filepath, 32, 45, 0.9, 0.1, 2592, 2048, 2);

    // init load
    myImgLoader(images_train, images_test, boxes_train, boxes_test, parts_list_train, parts_list_test);

    if(DEBUG_IMSHOW)
    {
    image_window win;
    win.set_image(images_train[0]);

    //for(auto && b : boxes_train[0])
    for(auto && b : boxes_train[0])
    {
        win.add_overlay(b, rgb_pixel(255,0,0));
        std::cout << "b.ignore: " << b.ignore << std::endl;
    }
    cout << "Hit enter to end program" << endl;
    cin.get();
    }

    /*
    if(DEBUG_IMSHOW)
    {
    image_window win, win2;

    for(int i = 0; i < faces_train.size(); i++)
    {
        win.clear_overlay();
        for(auto && b : faces_train[i])
        {
        win.set_image(images_train[i]);
        win.add_overlay(b.get_rect(), rgb_pixel(255,255,255));
        win.add_overlay(dlib::rectangle(b.part(0).x(),b.part(0).y(),b.part(0).x() + 2, b.part(0).y() + 2), rgb_pixel(255,255,255));
        win.add_overlay(dlib::rectangle(b.part(1).x(),b.part(1).y(),b.part(1).x() + 2, b.part(1).y() + 2), rgb_pixel(255,255,255));
        win.add_overlay(dlib::rectangle(b.part(2).x(),b.part(2).y(),b.part(2).x() + 2, b.part(2).y() + 2), rgb_pixel(255,255,255));
        win.add_overlay(dlib::rectangle(b.part(3).x(),b.part(3).y(),b.part(3).x() + 2, b.part(3).y() + 2), rgb_pixel(255,255,255));
        }
        cout << "id: " << i << "Hit enter ..." << endl;
        cin.get();
    }
    for(int i = 0; i < faces_train.size(); i++)
    {
        win2.clear_overlay();
        for(auto && b : faces_test[i])
        {
            win2.set_image(images_test[i]);
            win2.add_overlay(b.get_rect(), rgb_pixel(255,255,255));
            win2.add_overlay(dlib::rectangle(b.part(0).x(),b.part(0).y(),b.part(0).x() + 2, b.part(0).y() + 2), rgb_pixel(255,255,255));
            win2.add_overlay(dlib::rectangle(b.part(1).x(),b.part(1).y(),b.part(1).x() + 2, b.part(1).y() + 2), rgb_pixel(255,255,255));
            win2.add_overlay(dlib::rectangle(b.part(2).x(),b.part(2).y(),b.part(2).x() + 2, b.part(2).y() + 2), rgb_pixel(255,255,255));
            win2.add_overlay(dlib::rectangle(b.part(3).x(),b.part(3).y(),b.part(3).x() + 2, b.part(3).y() + 2), rgb_pixel(255,255,255));
        }
        cout << "id: " << i << "Hit enter ..." << endl;
        cin.get();
    }
    }*/


    int num_overlapped_ignored_test = 0;
    for (auto& v : boxes_test)
        num_overlapped_ignored_test += ignore_overlapped_boxes(v, test_box_overlap(0.50, 0.95));

    int num_overlapped_ignored = 0;
    int num_additional_ignored = 0;
    for (auto& v : boxes_train)
    {
        num_overlapped_ignored += ignore_overlapped_boxes(v, test_box_overlap(0.50, 0.95));
        for (auto& bb : v)
        {
            if (bb.rect.width() < 60 && bb.rect.height() < 15)
            {
                if (!bb.ignore)
                {
                    bb.ignore = true;
                    ++num_additional_ignored;
                }
            }

        }
    }

    // When modifying a dataset like this, it's a really good idea to print a log of how
    // many boxes you ignored.  It's easy to accidentally ignore a huge block of data, so
    // you should always look and see that things are doing what you expect.
    cout << "num_overlapped_ignored: "<< num_overlapped_ignored << endl;
    cout << "num_additional_ignored: "<< num_additional_ignored << endl;
    cout << "num_overlapped_ignored_test: "<< num_overlapped_ignored_test << endl;


    cout << "num training images: " << images_train.size() << endl;
    cout << "num testing images: " << images_test.size() << endl;


    // Our vehicle detection dataset has basically 3 different types of boxes.  Square
    // boxes, tall and skinny boxes (e.g. semi trucks), and short and wide boxes (e.g.
    // sedans).  Here we are telling the MMOD algorithm that a vehicle is recognizable as
    // long as the longest box side is at least 70 pixels long and the shortest box side is
    // at least 30 pixels long.  mmod_options will use these parameters to decide how large
    // each of the sliding windows needs to be so as to be able to detect all the vehicles.
    // Since our dataset has basically these 3 different aspect ratios, it will decide to
    // use 3 different sliding windows.  This means the final con layer in the network will
    // have 3 filters, one for each of these aspect ratios. 
    //
    // Another thing to consider when setting the sliding window size is the "stride" of
    // your network.  The network we defined above downsamples the image by a factor of 8x
    // in the first few layers.  So when the sliding windows are scanning the image, they
    // are stepping over it with a stride of 8 pixels.  If you set the sliding window size
    // too small then the stride will become an issue.  For instance, if you set the
    // sliding window size to 4 pixels, then it means a 4x4 window will be moved by 8
    // pixels at a time when scanning. This is obviously a problem since 75% of the image
    // won't even be visited by the sliding window.  So you need to set the window size to
    // be big enough relative to the stride of your network.  In our case, the windows are
    // at least 30 pixels in length, so being moved by 8 pixel steps is fine. 
    mmod_options options(boxes_train, 60, 15);


    // This setting is very important and dataset specific.  The vehicle detection dataset
    // contains boxes that are marked as "ignore", as we discussed above.  Some of them are
    // ignored because we set ignore to true in the above code.  However, the xml files
    // also contained a lot of ignore boxes.  Some of them are large boxes that encompass
    // large parts of an image and the intention is to have everything inside those boxes
    // be ignored.  Therefore, we need to tell the MMOD algorithm to do that, which we do
    // by setting options.overlaps_ignore appropriately.  
    // 
    // But first, we need to understand exactly what this option does.  The MMOD loss
    // is essentially counting the number of false alarms + missed detections produced by
    // the detector for each image.  During training, the code is running the detector on
    // each image in a mini-batch and looking at its output and counting the number of
    // mistakes.  The optimizer tries to find parameters settings that minimize the number
    // of detector mistakes.
    // 
    // This overlaps_ignore option allows you to tell the loss that some outputs from the
    // detector should be totally ignored, as if they never happened.  In particular, if a
    // detection overlaps a box in the training data with ignore==true then that detection
    // is ignored.  This overlap is determined by calling
    // options.overlaps_ignore(the_detection, the_ignored_training_box).  If it returns
    // true then that detection is ignored.
    // 
    // You should read the documentation for test_box_overlap, the class type for
    // overlaps_ignore for full details.  However, the gist is that the default behavior is
    // to only consider boxes as overlapping if their intersection over union is > 0.5.
    // However, the dlib vehicle detection dataset contains large boxes that are meant to
    // mask out large areas of an image.  So intersection over union isn't an appropriate
    // way to measure "overlaps with box" in this case.  We want any box that is contained
    // inside one of these big regions to be ignored, even if the detection box is really
    // small.  So we set overlaps_ignore to behave that way with this line.
    options.overlaps_ignore = test_box_overlap(0.5, 0.95);
    options.use_bounding_box_regression = true;

    net_type net(options);

    // The final layer of the network must be a con layer that contains 
    // options.detector_windows.size() filters.  This is because these final filters are
    // what perform the final "sliding window" detection in the network.  For the dlib
    // vehicle dataset, there will be 3 sliding window detectors, so we will be setting
    // num_filters to 3 here.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size() * 5);


    dnn_trainer<net_type> trainer(net,sgd(0.0001,0.9));
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();


    // While training, we are going to use early stopping.  That is, we will be checking
    // how good the detector is performing on our test data and when it stops getting
    // better on the test data we will drop the learning rate.  We will keep doing that
    // until the learning rate is less than 1e-4.   These two settings tell the trainer to
    // do that.  Essentially, we are setting the first argument to infinity, and only the
    // test iterations without progress threshold will matter.  In particular, it says that
    // once we observe 1000 testing mini-batches where the test loss clearly isn't
    // decreasing we will lower the learning rate.
    trainer.set_iterations_without_progress_threshold(50000);
    trainer.set_test_iterations_without_progress_threshold(1000);

    const string sync_filename = "mmod_plates_model5_T3_sync";
    trainer.set_synchronization_file(sync_filename, std::chrono::minutes(5));

    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels;
    //std::vector<std::vector<std::pair<mmod_rect, std::vector<dlib::point>>>> mini_batch_labels;
    random_cropper cropper;
    cropper.set_seed(time(0));
    cropper.set_chip_dims(350, 350);
    cropper.set_randomly_flip(false);
    // Usually you want to give the cropper whatever min sizes you passed to the
    // mmod_options constructor, or very slightly smaller sizes, which is what we do here.
    cropper.set_min_object_size(59,14);
    cropper.set_max_rotation_degrees(2);
    dlib::rand rnd;

    // Log the training parameters to the console
    cout << trainer << cropper << endl;

    int cnt = 1;
    // Run the trainer until the learning rate gets small.

    // load new images in roundRobin fashion every 5k iterations
    // this is due to ram limitations
    const int CNT_RELOAD_IMAGES = 10000;

    if(myImgLoader.checkRamIsNotEnough())
    {
        std::cout << "NOTE: Ram mem is not enough, images will be reloading in roundRobin fashion;" << std::endl;
    }
    else {
        std::cout << "Ram is enough for all images, no need for round robin loading;" << std::endl;
    }

    while(trainer.get_learning_rate() >= 1e-4)
    {
        // Every 30 mini-batches we do a testing mini-batch.
        //std::cout << "cnt : " << cnt << std::endl;
        if(cnt % CNT_RELOAD_IMAGES == 0)
            if(myImgLoader.checkRamIsNotEnough())
            {
                std::cout << "reloading images " << std::endl;
                myImgLoader(images_train, images_test, boxes_train , boxes_test, parts_list_train, parts_list_test);
            }

        if (cnt%30 != 0 || images_test.size() == 0)
        {
            //cropper(27, images_train, boxes_train, mini_batch_samples, mini_batch_labels);
            cropper(57, images_train, boxes_train, mini_batch_samples, mini_batch_labels);
            // We can also randomly jitter the colors and that often helps a detector
            // generalize better to new images.
            for (auto&& img : mini_batch_samples)
                disturb_colors(img, rnd);

            // It's a good idea to, at least once, put code here that displays the images
            // and boxes the random cropper is generating.  You should look at them and
            // think about if the output makes sense for your problem.  Most of the time
            // it will be fine, but sometimes you will realize that the pattern of cropping
            // isn't really appropriate for your problem and you will need to make some
            // change to how the mini-batches are being generated.  Maybe you will tweak
            // some of the cropper's settings, or write your own entirely separate code to
            // create mini-batches.  But either way, if you don't look you will never know.
            // An easy way to do this is to create a dlib::image_window to display the
            // images and boxes.

            trainer.train_one_step(mini_batch_samples, mini_batch_labels);
        }
        else
        {
            //cropper(27, images_test, boxes_test, mini_batch_samples, mini_batch_labels);
            cropper(57, images_test, boxes_test, mini_batch_samples, mini_batch_labels);
            // We can also randomly jitter the colors and that often helps a detector
            // generalize better to new images.
            for (auto&& img : mini_batch_samples)
                disturb_colors(img, rnd);

            trainer.test_one_step(mini_batch_samples, mini_batch_labels);
        }
        ++cnt;
    }
    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("mmod_plates_model5_detector_T3.dat") << net;


    // It's a really good idea to print the training parameters.  This is because you will
    // invariably be running multiple rounds of training and should be logging the output
    // to a file.  This print statement will include many of the training parameters in
    // your log.
    cout << trainer << cropper << endl;

    cout << "\nsync_filename: " << sync_filename << endl;
    cout << "num training images: "<< images_train.size() << endl;
    cout << "training results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);
    // Upsampling the data will allow the detector to find smaller cars.  Recall that 
    // we configured it to use a sliding window nominally 70 pixels in size.  So upsampling
    // here will let it find things nominally 35 pixels in size.  Although we include a
    // limit of 1800*1800 here which means "don't upsample an image if it's already larger
    // than 1800*1800".  We do this so we don't run out of RAM, which is a concern because
    // some of the images in the dlib vehicle dataset are really high resolution.
    upsample_image_dataset<pyramid_down<2>>(images_train, boxes_train, 1800*1800);
    cout << "training upsampled results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);


    cout << "num testing images: "<< images_test.size() << endl;
    cout << "testing results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);
    upsample_image_dataset<pyramid_down<2>>(images_test, boxes_test, 1200*1200);
    cout << "testing upsampled results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);

    /*
        This program takes many hours to execute on a high end GPU.  It took about a day to
        train on a NVIDIA 1080ti.  The resulting model file is available at
            http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
        It should be noted that this file on dlib.net has a dlib::shape_predictor appended
        onto the end of it (see dnn_mmod_find_cars_ex.cpp for an example of its use).  This
        explains why the model file on dlib.net is larger than the
        mmod_rear_end_vehicle_detector.dat output by this program.

        You can see some videos of this vehicle detector running on YouTube:
            https://www.youtube.com/watch?v=4B3bzmxMAZU
            https://www.youtube.com/watch?v=bP2SUo5vSlc

        Also, the training and testing accuracies were:
            num training images: 2217
            training results: 0.990738 0.736431 0.736073 
            training upsampled results: 0.986837 0.937694 0.936912 
            num testing images: 135
            testing results: 0.988827 0.471372 0.470806 
            testing upsampled results: 0.987879 0.651132 0.650399 
    */

    return 0;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




