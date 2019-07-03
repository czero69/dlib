





#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

#include <chrono>

using namespace std;
using namespace dlib;

/*
using net_type = loss_multiclass_log<
                            fc<10,
                            relu<fc<84,
                            relu<fc<120,
                            max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                            max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                            input<matrix<unsigned char>>
                            >>>>>>>>>>>>;
*/

template <
    int N,
    template <typename> class BN,
    int stride,
    typename SUBNET
    >
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
    >
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <
    template <int,template<typename>class,int,typename> class block,
    int N,
    template<typename>class BN,
    typename SUBNET
    >
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

//template <typename SUBNET> using res       = relu<residual<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares      = relu<residual<block,8,affine,SUBNET>>;
//template <typename SUBNET> using res_down  = relu<residual_down<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = relu<residual_down<block,8,affine,SUBNET>>;

using net_type = loss_multiclass_log<fc<17,
                                avg_pool_everything<
                                ares<ares<ares<ares_down<
                                repeat<1,ares,
                                ares_down<
                                ares<
                                input<matrix<unsigned char>>
                                >>>>>>>>>>;

int main(int argc, char** argv) try
{
    net_type net;
    //shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, the file also includes a separately trained shape_predictor.  To see
    // a generic example of how to train those refer to train_shape_predictor_ex.cpp.

    if (argc != 3)
    {
        cout << "   ./dnn_co.. detector.dat img.jpg" << endl;
        cout << endl;
        return 0;
    }

    const std::string detectorPath = argv[1];
    const std::string imgPath = argv[2];


    /*
     *
     * set fc<16, to all countries below count
BY: 0
CZ: 1
D: 2
E: 3
F: 4
GB: 5
H: 6
I: 7
LT: 8
LV: 9
MC: 10
NL: 11
PL: 12
RO: 13
RUS: 14
SK: 15
UA: 16

    */

    deserialize(detectorPath) >> net;

    std::cout << "deserialization complete! " << std::endl;

    // test if not auto in serialization
    //int trainingLabelsCount = 16;
    //net.subnet().layer_details().set_num_outputs(trainingLabelsCount);


    dlib::matrix<unsigned char> imgTmp(15*4,60*4);

    matrix<unsigned char> img;
    load_image(img, imgPath);
    resize_image(img, imgTmp);
    std::swap(img, imgTmp);


    std::vector<matrix<unsigned char>> img_vec(8);

    for(int i = 0; i < 8; i++)
    {
        dlib::matrix<unsigned char> imgTmp(15*4,60*4);
        load_image(img_vec[i], imgPath);
        resize_image(img_vec[i], imgTmp);
        std::swap(img_vec[i], imgTmp);
    }

    /*const int nominal_width = 2592;
    const int nominal_height = 2048;

        if(img.nc() >= nominal_width/2 || img.nr() >= nominal_height/2)
        {
            resize_image(0.5, img); // it makes swap, so RAM will go down from now

            for(int i = 0; i < 8; i++)
                resize_image(0.5,  img_vec[i]);
        }*/

    image_window win;
    win.set_image(img);

    //time measuring
    // sigle
    // Record start time

    unsigned long predicted_label;
    std::vector<float> probs;
    // warmup
    for(int i = 0; i < 50; i++)
        predicted_label = net(img);

    // Record start time

    auto startBatch = std::chrono::high_resolution_clock::now();

    // Portion of code to be timed

    for(int i = 0; i < 25; i++)
         net(img_vec);

    // Record end time
    auto finishBatch = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedBatch = finishBatch - startBatch;
    std::cout << "Elapsed time per 8x Batch img: " << elapsedBatch.count()/(25*8) << " s\n";

    //record start time

    auto start = std::chrono::high_resolution_clock::now();

    // Portion of code to be timed
    for(int i = 0; i < 100; i++)
        net(img);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time per img: " << elapsed.count()/100 << " s\n";

    //std::vector<matrix<rgb_pixel>> imgVec;
    std:cout << "predicted label: " << predicted_label << std::endl;

    // Make a network with softmax as the final layer.  We don't have to do this
        // if we just want to output the single best prediction, since the anet_type
        // already does this.  But if we instead want to get the probability of each
        // class as output we need to replace the last layer of the network with a
        // softmax layer, which we do as follows:
    softmax<net_type::subnet_type> snet;
    snet.subnet() = net.subnet();

    snet(img);

    auto & probList = snet(img);
    for(auto & p : probList)
    std::cout << "prob: " << p << std::endl;


}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

