// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to run a CNN based vehicle detector using dlib.  The
    example loads a pretrained model and uses it to find the rear ends of cars in
    an image.  We will also visualize some of the detector's processing steps by
    plotting various intermediate images on the screen.  Viewing these can help
    you understand how the detector works.
    
    The model used by this example was trained by the dnn_mmod_train_find_cars_ex.cpp 
    example.  Also, since this is a CNN, you really should use a GPU to get the
    best execution speed.  For instance, when run on a NVIDIA 1080ti, this detector 
    runs at 98fps when run on the provided test image.  That's more than an order 
    of magnitude faster than when run on the CPU.

    Users who are just learning about dlib's deep learning API should read
    the dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples to learn
    how the API works.  For an introduction to the object detection method you
    should read dnn_mmod_ex.cpp.

    You can also see some videos of this vehicle detector running on YouTube:
        https://www.youtube.com/watch?v=4B3bzmxMAZU
        https://www.youtube.com/watch?v=bP2SUo5vSlc
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

#include <chrono>  // for high_resolution_clock

#include <dlib/cuda/myCuda_tensorToDets.h>

using namespace std;
using namespace dlib;



// The rear view vehicle detector network
/*template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;*/


template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;

template <long num_filters, typename SUBNET> using rcon5_1_div4  = relu<affine<con<num_filters / 4,5,1,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcon1_5_div4  = relu<affine<con<num_filters / 4,1,5,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcon1_1_div4  = relu<affine<con<num_filters / 4,1,1,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcon5_5_div4_str2  = relu<affine<con<num_filters / 4,5,5,2,2,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcon5_5_div4  = relu<affine<con<num_filters,5,5,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcon1_1  = relu<affine<con<num_filters,1,1,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcoord_con1_1  = relu<affine<coord_con<num_filters,1,1,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcoord_con1_1_div4  = relu<affine<coord_con<num_filters / 4,1,1,1,1,SUBNET>>>;
template <long num_filters, typename SUBNET> using rcoord_con5_5_div4  = relu<affine<coord_con<num_filters,5,5,1,1,SUBNET>>>;

template <long num_filters, typename SUBNET> using rfireBlock5 = rcon1_1<num_filters, rcon1_5_div4<num_filters, rcon5_1_div4<num_filters, rcon1_1_div4<num_filters, SUBNET>>>>;
template <typename SUBNET> using _55_rfireBlock5  = rfireBlock5<55, SUBNET>;

template <long num_filters, typename SUBNET> using rfireBlock5_v2 = rcon1_1<num_filters, rcon5_5_div4_str2<num_filters, rcon1_1_div4<num_filters, SUBNET>>>;
template <long num_filters, typename SUBNET> using r_coord_fireBlock5_v2 = rcon1_1<num_filters, rcon5_5_div4_str2<num_filters, rcoord_con1_1_div4<num_filters, SUBNET>>>;
template <typename SUBNET> using _55_rfireBlock5_v2  = rcon5_1_div4<55, rcon1_1<55, SUBNET>>;
template <typename SUBNET> using _55_rfireBlock5_v3  = rcon5_5_div4<55, rcon1_1<55, SUBNET>>;
template <typename SUBNET> using _55_coord_rfireBlock5_v3  = rcoord_con5_5_div4<55, rcon1_1<55, SUBNET>>;

template <typename SUBNET> using _32part_downsampler_fb = max_pool<5,5, 2, 2, rfireBlock5<32, SUBNET>>;
template <typename SUBNET> using _32part_downsampler_fb_v2 = rfireBlock5_v2<32, SUBNET>;
template <typename SUBNET> using _32part_coord_downsampler_fb_v2 = r_coord_fireBlock5_v2<32, SUBNET>;


template <typename SUBNET> using downsampler2  = _32part_downsampler_fb<_32part_downsampler_fb<relu<affine<con5d<16,SUBNET>>>>>;
template <typename SUBNET> using downsampler2_v2  = _32part_downsampler_fb_v2<_32part_downsampler_fb_v2<relu<affine<con5d<16,SUBNET>>>>>;
template <typename SUBNET> using coord_downsampler2_v2  = _32part_downsampler_fb_v2<_32part_coord_downsampler_fb_v2<relu<affine<con5d<16,SUBNET>>>>>;

//using net_type = loss_mmod<con<1,9,9,1,1,rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<4>>>>>>;
//using net_type = loss_mmod<con<1,9,9,1,1,_55_rfireBlock5<downsampler2<input_rgb_image_pyramid<pyramid_down<4>>>>>>;

//using net_type = loss_mmod<con<1,9,9,1,1,_55_rfireBlock5_v2<downsampler2_v2<input_rgb_image_pyramid<pyramid_down<4>>>>>>;

// not yet rdy//using net_type = loss_mmod<con<1,9,9,1,1,_55_rfireBlock5_v2<downsampler2_v2<rcon1_1< 1 , input_rgb_image_pyramid<pyramid_down<4>>>>>>>;

// experiment phase here:

// model 7 (t2)

// model 7-v2
//using net_type = loss_mmod<concat2<tag5, tag4, tag5<coord_con<28,9,9,1,1, skip3<tag4<con<7,9,9,1,1,tag3<_55_rfireBlock5_v2<coord_downsampler2_v2<rcon1_1< 1 , input_rgb_image_pyramid<pyramid_down<4>>>>>>>>>>>>>;

// model7v3
using net_type = loss_mmod<concat2<tag5, tag4, tag5<con<28,9,9,1,1, skip3<tag4<con<7,9,9,1,1,tag3<_55_coord_rfireBlock5_v3<multiply<coord_downsampler2_v2<input_rgb_image_pyramid<pyramid_down<4>>>>>>>>>>>>>;

//model8
template <long num_filters, typename SUBNET> using con1_1  = affine<con<num_filters,1,1,1,1,SUBNET>>;
template <long num_filters, typename SUBNET> using rcon5_5_str2  = relu<affine<con<num_filters,5,5,2,2,SUBNET>>>;


template <long num_filters, typename SUBNET> using rskip_coord_fireBlock5_v5 = relu<add_prev1<con1_1<num_filters, rcon5_5_str2<num_filters/2, rcoord_con1_1<num_filters/2, tag1<SUBNET>>>>>>;
template <long num_filters, typename SUBNET> using rskip_down_fireBlock5_v5 = relu<add_prev2<avg_pool<2,2,2,2,skip1<tag2<con1_1<num_filters, rcon5_5_str2<num_filters/2, rcon1_1<num_filters/2, tag1<SUBNET>>>>>>>>>;
template <long num_filters, typename SUBNET> using rskip_down_coord_fireBlock5_v5 = relu<add_prev7<avg_pool<2,2,2,2,skip6<tag7<con1_1<num_filters, rcon5_5_str2<num_filters/2, rcoord_con1_1<num_filters/2, tag6<SUBNET>>>>>>>>>;

template <typename SUBNET> using skip_coord_downsampler2_v3  = rskip_down_fireBlock5_v5<32, rskip_down_coord_fireBlock5_v5<32, relu<affine<con5d<16,SUBNET>>>>>;


//using net_type = loss_mmod<concat2<tag4, tag5, tag5<con<28,9,9,1,1, skip3<tag4<con<7,9,9,1,1,tag3<rskip_coord_fireBlock5_v5<55, skip_coord_downsampler2_v3<input_rgb_image_pyramid<pyramid_down<4>>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    // trying options ---------------------------------------

    cuda::cudaDeviceScheduleMode::Choice myMode = cuda::cudaDeviceScheduleMode::Choice::Blocking;
    cuda::set_flag_cudaDeviceSchedule(myMode);
    mmod_options options;


    options.use_bounding_box_regression = false;

    // net_type net;
    net_type net(options);

    // The final layer of the network must be a con layer that contains
    // options.detector_windows.size() filters.  This is because these final filters are
    // what perform the final "sliding window" detection in the network.  For the dlib
    // vehicle dataset, there will be 3 sliding window detectors, so we will be setting
    // num_filters to 3 here.

    //net.subnet().layer_details().set_num_filters(options.detector_windows.size() * 5);

    ///    ------------------------------------------------
    //shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, the file also includes a separately trained shape_predictor.  To see
    // a generic example of how to train those refer to train_shape_predictor_ex.cpp.

    if (argc != 3)
    {
        cout << "   ./dnn_mmod_find_plates detector.dat img.jpg" << endl;
        cout << endl;
        return 0;
    }

    const std::string detectorPath = argv[1];
    const std::string imgPath = argv[2];


    deserialize(detectorPath) >> net;

    //net.subnet().layer_details().set_num_filters(7);

    std::cout << " net.subnet().get_output().k() :" <<  net.subnet().get_output().k() << std::endl;
    std::cout << " net.subnet().layer_details().get_layer_params().k() :" <<  net.subnet().layer_details().get_layer_params().k() << std::endl;
    // net.subnet().get_output().k() ;
    //net.subnet().layer_details().set_num_filters(7);

    matrix<rgb_pixel> img;
    load_image(img, imgPath);

    std::vector<matrix<rgb_pixel>> img_vec(8);

    for(int i = 0; i < 8; i++)
        load_image(img_vec[i], imgPath);

    // WARNING - INPUT IMAGE IS SCALED HERE - BE AWARE (!)
    const int nominal_width = 10 * 2592;
    const int nominal_height = 10 * 2048;

        if(img.nc() >= nominal_width/2 || img.nr() >= nominal_height/2)
        {
            resize_image(0.5, img); // it makes swap, so RAM will go down from now

            for(int i = 0; i < 8; i++)
                resize_image(0.5,  img_vec[i]);
        }

    image_window win;
    win.set_image(img);

    // Run the detector on the image and show us the output.


    for (auto&& d : net.process(img, -0.9))
    {
        // We use a shape_predictor to refine the exact shape and location of the detection
        // box.  This shape_predictor is trained to simply output the 4 corner points of
        // the box.  So all we do is make a rectangle that tightly contains those 4 points
        // and that rectangle is our refined detection position.
        //auto fd = sp(img,d);
        rectangle rect;
        //for (unsigned long j = 0; j < fd.num_parts(); ++j)
         //   rect += fd.part(j);
        win.add_overlay(d.rect, rgb_pixel(255,0,0));
    }

    //time measuring

    // sigle

    // Record start time

    // warmup
    std::cout << "-------warmup ---------" << std::endl;
    for(int i = 0; i < 5; i++)
        net(img);

    // this is how you can set probability threshold for detector
    // or for batches process_batch
    //net.process(img, -0.5);

    std::cout << "aw net.subnet().get_output().k() :" <<  net.subnet().get_output().k() << std::endl;
    std::cout << "aw net.subnet().layer_details().get_layer_params().k() :" <<  net.subnet().layer_details().get_layer_params().k() << std::endl;

    // Record start time

    std::cout << "-------END warmup ---------" << std::endl << std::endl;

    /*
    std::cout << "-------BATCH TEST ---------" << std::endl;

    net.clean();
    auto startBatch = std::chrono::high_resolution_clock::now();

    // Portion of code to be timed

    std::vector<std::vector<dlib::mmod_rect>> v_ret;

    const int BATCH_TEST_COUNT = 2;

    for(int i = 0; i < BATCH_TEST_COUNT; i++)
        v_ret = net(img_vec);
    // Record end time
    auto finishBatch = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedBatch = finishBatch - startBatch;
    std::cout << "Elapsed time per 8x Batch img: " << elapsedBatch.count()/(BATCH_TEST_COUNT*img_vec.size()) << " s\n";

    std::cout << "-------END BATCH TEST ---------" << std::endl << std::endl;
    */
    //record start time

    std::cout << "-------SINGLE TEST ---------" << std::endl;
    net.clean(); // --

    auto start = std::chrono::high_resolution_clock::now();

    const int NORMAL_TEST_COUNT = 100;

    // Portion of code to be timed
    for(int i = 0; i < NORMAL_TEST_COUNT; i++)
        net.process(img, -0.5); //net(img);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time per img: " << elapsed.count()/NORMAL_TEST_COUNT << " s\n";

    std::cout << "-------END SINGLE TEST ---------" << std::endl;

    //std::vector<matrix<rgb_pixel>> imgVec;

    //exit(0);
    cout << "Hit enter to view the intermediate processing steps" << endl;
    cin.get();



    // czy możemy zrobić regresję partsów zamiast regresję boxa ?
    // czy skip architecture da poprawę jakościową
    // sprawdź czy aktualna box regresjon jest w ogóle używana?

    // Now let's look at how the detector works.  The high level processing steps look like:
    //   1. Create an image pyramid and pack the pyramid into one big image.  We call this
    //      image the "tiled pyramid".
    //   2. Run the tiled pyramid image through the CNN.  The CNN outputs a new image where
    //      bright pixels in the output image indicate the presence of cars.  
    //   3. Find pixels in the CNN's output image with a value > 0.  Those locations are your
    //      preliminary car detections.  
    //   4. Perform non-maximum suppression on the preliminary detections to produce the
    //      final output.
    //
    // We will be plotting the images from steps 1 and 2 so you can visualize what's
    // happening.  For the CNN's output image, we will use the jet colormap so that "bright"
    // outputs, i.e. pixels with big values, appear in red and "dim" outputs appear as a
    // cold blue color.  To do this we pick a range of CNN output values for the color
    // mapping.  The specific values don't matter.  They are just selected to give a nice
    // looking output image.
    const float lower = -2.5;
    const float upper = 0.0;
    cout << "jet color mapping range:  lower="<< lower << "  upper="<< upper << endl;



    //
    net.process(img, -0.9);

    // Create a tiled pyramid image and display it on the screen. 
    std::vector<rectangle> rects;
    matrix<rgb_pixel> tiled_img;
    // Get the type of pyramid the CNN used
    using pyramid_type = std::remove_reference<decltype(input_layer(net))>::type::pyramid_type;
    // And tell create_tiled_pyramid to create the pyramid using that pyramid type.
    create_tiled_pyramid<pyramid_type>(img, tiled_img, rects, 
                                       input_layer(net).get_pyramid_padding(), 
                                       input_layer(net).get_pyramid_outer_padding());
    image_window winpyr(tiled_img, "Tiled pyramid");



    // This CNN detector represents a sliding window detector with 3 sliding windows.  Each
    // of the 3 windows has a different aspect ratio, allowing it to find vehicles which
    // are either tall and skinny, squarish, or short and wide.  The aspect ratio of a
    // detection is determined by which channel in the output image triggers the detection.
    // Here we are just going to max pool the channels together to get one final image for
    // our display.  In this image, a pixel will be bright if any of the sliding window
    // detectors thinks there is a car at that location.
    cout << "Number of channels in final tensor image: " << net.subnet().get_output().k() << endl;
    matrix<float> network_output = image_plane(net.subnet().get_output(),0,0);
    matrix<float> network_output_huntingForRegression_k5 = image_plane(net.subnet().get_output(),0,7);
    for (long k = 1; k < net.subnet().get_output().k() / 5 ; k += 1)
        network_output = max_pointwise(network_output, image_plane(net.subnet().get_output(),0,k));

    // We will also upsample the CNN's output image.  The CNN we defined has an 8x
    // downsampling layer at the beginning. In the code below we are going to overlay this
    // CNN output image on top of the raw input image.  To make that look nice it helps to
    // upsample the CNN output image back to the same resolution as the input image, which
    // we do here.
    const double network_output_scale = img.nc()/(double)network_output.nc();
    resize_image(network_output_scale, network_output);


    matrix<float> showingMatrix_5k = (network_output_huntingForRegression_k5/ 30);
    resize_image(network_output_scale, showingMatrix_5k);
    // Display the network's output as a color image.   
    image_window win_output_5k(showingMatrix_5k, "showingMatrix_5k");
    image_window win_output(jet(network_output, upper, lower), "Output tensor from the network");


    // Also, overlay network_output on top of the tiled image pyramid and display it.
    for (long r = 0; r < tiled_img.nr(); ++r)
    {
        for (long c = 0; c < tiled_img.nc(); ++c)
        {
            dpoint tmp(c,r);
            tmp = input_tensor_to_output_tensor(net, tmp);
            tmp = point(network_output_scale*tmp);
            if (get_rect(network_output).contains(tmp))
            {
                float val = network_output(tmp.y(),tmp.x());
                // alpha blend the network output pixel with the RGB image to make our
                // overlay.
                rgb_alpha_pixel p;
                assign_pixel(p , colormap_jet(val,lower,upper));
                p.alpha = 120;
                assign_pixel(tiled_img(r,c), p);
            }
        }
    }
    // If you look at this image you can see that the vehicles have bright red blobs on
    // them.  That's the CNN saying "there is a car here!".  You will also notice there is
    // a certain scale at which it finds cars.  They have to be not too big or too small,
    // which is why we have an image pyramid.  The pyramid allows us to find cars of all
    // scales.
    image_window win_pyr_overlay(tiled_img, "Detection scores on image pyramid");




    // Finally, we can collapse the pyramid back into the original image.  The CNN doesn't
    // actually do this step, since it's enough to threshold the tiled pyramid image to get
    // the detections.  However, it makes a nice visualization and clearly indicates that
    // the detector is firing for all the cars.
    matrix<float> collapsed(img.nr(), img.nc());
    resizable_tensor input_tensor;
    input_layer(net).to_tensor(&img, &img+1, input_tensor);
    for (long r = 0; r < collapsed.nr(); ++r)
    {
        for (long c = 0; c < collapsed.nc(); ++c)
        {
            // Loop over a bunch of scale values and look up what part of network_output
            // corresponds to the point(c,r) in the original image, then take the max
            // detection score over all the scales and save it at pixel point(c,r).
            float max_score = -1e30;
            for (double scale = 1; scale > 0.2; scale *= 5.0/6.0)
            {
                // Map from input image coordinates to tiled pyramid coordinates.
                dpoint tmp = center(input_layer(net).image_space_to_tensor_space(input_tensor,scale, drectangle(dpoint(c,r))));
                // Now map from pyramid coordinates to network_output coordinates.
                tmp = point(network_output_scale*input_tensor_to_output_tensor(net, tmp));

                if (get_rect(network_output).contains(tmp))
                {
                    float val = network_output(tmp.y(),tmp.x());
                    if (val > max_score)
                        max_score = val;
                }
            }

            collapsed(r,c) = max_score;

            // Also blend the scores into the original input image so we can view it as
            // an overlay on the cars.
            rgb_alpha_pixel p;
            assign_pixel(p , colormap_jet(max_score,lower,upper));
            p.alpha = 120;
            assign_pixel(img(r,c), p);
        }
    }

    image_window win_collapsed(jet(collapsed, upper, lower), "Collapsed output tensor from the network");
    image_window win_img_and_sal(img, "Collapsed detection scores on raw image");


    cout << "Hit enter to end program" << endl;
    cin.get();
}
catch(image_load_error& e)
{
    cout << e.what() << endl;
    cout << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The correct model file can be obtained from: http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2" << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




