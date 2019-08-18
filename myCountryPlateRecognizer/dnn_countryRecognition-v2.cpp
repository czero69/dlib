// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will train the venerable LeNet convolutional
    neural network to recognize hand written digits.  The network will take as
    input a small image and classify it as one of the 10 numeric digits between
    0 and 9.

    The specific network we will run is from the paper
        LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE 86.11 (1998): 2278-2324.
    except that we replace the sigmoid non-linearities with rectified linear units.

    These tools will use CUDA and cuDNN to drastically accelerate network
    training and testing.  CMake should automatically find them if they are
    installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <myPlatesDetector/myUtils.h>
#include <dlib/gui_widgets.h>
#include <random>

using namespace std;
using namespace dlib;

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


template <typename SUBNET> using res       = relu<residual<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares      = relu<residual<block,8,affine,SUBNET>>;
template <typename SUBNET> using res_down  = relu<residual_down<block,8,bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = relu<residual_down<block,8,affine,SUBNET>>;

const unsigned long number_of_classes = 10;
using net_type = loss_multiclass_log<fc<number_of_classes,
                            avg_pool_everything<
                            res<res<res<res_down<
                            repeat<1,res, // repeat this layer x1 times
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;

// And finally, let's define a residual network building block that uses
// parametric ReLU units instead of regular ReLU.
template <typename SUBNET>
using pres  = prelu<add_prev1<bn_con<con<8,3,3,1,1,prelu<bn_con<con<8,3,3,1,1,tag1<SUBNET>>>>>>>>;



template <
    typename array_type
    >
void copyVecFast(const std::vector<array_type>& original, std::vector<array_type>& output, int begin =0, int end=0)
{
  assert(end - begin > 0);
  assert(end < original.size());

  output.clear();
  output.reserve(end - begin);
  copy(original.begin() + begin ,original.begin() + end,back_inserter(output));
}

int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.
    if (argc != 3)
    {
        cout << "./dnn_countryRecognition path-to-trainingDir path-to-testingDir" << endl;
        return 1;
    }

    const std::string dirnameTrain = argv[1];
    const std::string dirnameTest = argv[2];

    const std::string dirnameTrain2 = "/media/kamil/DATA/lprDataset/treningCountryRecognZabytkowe";

    std::cout << "dirnameTrain: " << dirnameTrain << ", " << "dirnameTest"  <<  dirnameTest  << std::endl;


    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;


    int trainingLabelsCount = 0, testingLabelsCount = 0;

    trainingLabelsCount += load_image_datasetCountryPlate_subset(training_images, training_labels, dirnameTrain, dlib::datasets_utils::TRAIN, 10);
    //trainingLabelsCount += load_image_datasetCountryPlate_subset(training_images, training_labels, dirnameTrain, dlib::datasets_utils::TEST, 10); // add test images to training now,
    testingLabelsCount += load_image_datasetCountryPlate_subset(testing_images, testing_labels, dirnameTest, dlib::datasets_utils::TEST,  10);

    assert(trainingLabelsCount == testingLabelsCount);

    std::cout << "trainingLabelsCount: " << trainingLabelsCount <<  std::endl;

    auto seed = unsigned ( std::time(0) );

    // @TODo
    // this is not correct way, no guarantee random will be tha same and can be platfrom specyfic
    // for the sake of easy implementation I leave it here

    std::srand ( seed );
    std::random_shuffle ( training_images.begin(), training_images.end() );
    std::srand ( seed );
    std::random_shuffle ( training_labels.begin(), training_labels.end() );
    std::srand ( seed );
    std::random_shuffle ( testing_images.begin(), testing_images.end() );
    std::srand ( seed );
    std::random_shuffle ( testing_labels.begin(), testing_labels.end() );


    bool DEBUG_IMSHOW = 0;

    if(DEBUG_IMSHOW)
        for(int i =0; i < testing_images.size(); i++)
        {
        dlib::image_window win;
        win.set_image(testing_images[i]);
        std::cout << "label is: " << testing_labels[i] << std::endl;

        cout << "Hit enter ..." << endl;
        cin.get();
        }

    //load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    // Now let's define the LeNet.  Broadly speaking, there are 3 parts to a network
    // definition.  The loss layer, a bunch of computational layers, and then an input
    // layer.  You can see these components in the network definition below.
    //
    // The input layer here says the network expects to be given matrix<unsigned char>
    // objects as input.  In general, you can use any dlib image or matrix type here, or
    // even define your own types by creating custom input layers.
    //
    // Then the middle layers define the computation the network will do to transform the
    // input into whatever we want.  Here we run the image through multiple convolutions,
    // ReLU units, max pooling operations, and then finally a fully connected layer that
    // converts the whole thing into just 10 numbers.
    //
    // Finally, the loss layer defines the relationship between the network outputs, our 10
    // numbers, and the labels in our dataset.  Since we selected loss_multiclass_log it
    // means we want to do multiclass classification with our network.   Moreover, the
    // number of network outputs (i.e. 10) is the number of possible labels.  Whichever
    // network output is largest is the predicted label.  So for example, if the first
    // network output is largest then the predicted digit is 0, if the last network output
    // is largest then the predicted digit is 9.
    /*using net_type = loss_multiclass_log<
                                fc<10,
                                relu<fc<84,
                                relu<fc<120,
                                max_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                input<matrix<unsigned char>>
                                >>>>>>>>>>>>;*/

    /*using net_type = loss_multiclass_log<
                                fc<10,
                                relu<fc<84,
                                relu<fc<120,
                                relu<con<16,20,20,1,5,
                                //avg_pool<2,2,2,2,relu<con<16,5,5,1,1,
                                avg_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                input<matrix<unsigned char>>
                                >>>>>>>>>>>;*/
    // This net_type defines the entire network architecture.  For example, the block
    // relu<fc<84,SUBNET>> means we take the output from the subnetwork, pass it through a
    // fully connected layer with 84 outputs, then apply ReLU.  Similarly, a block of
    // max_pool<2,2,2,2,relu<con<16,5,5,1,1,SUBNET>>> means we apply 16 convolutions with a
    // 5x5 filter size and 1x1 stride to the output of a subnetwork, then apply ReLU, then
    // perform max pooling with a 2x2 window and 2x2 stride.


    // So with that out of the way, we can make a network instance.
    net_type net;

    // Make a network with softmax as the final layer.  We don't have to do this
        // if we just want to output the single best prediction, since the anet_type
        // already does this.  But if we instead want to get the probability of each
        // class as output we need to replace the last layer of the network with a
        // softmax layer, which we do as follows:
    //softmax<net_type::subnet_type> snet;
    //snet.subnet() = net.subnet();

    // And then train it using the MNIST data.  The code below uses mini-batch stochastic
    // gradient descent with an initial learning rate of 0.01 to accomplish this.
    //dnn_trainer<net_type> trainer(net);
    dnn_trainer<net_type,adam> trainer(net,adam(0.0005, 0.9, 0.999));
    //dnn_trainer<net_type> trainer(net,sgd(0.0001,0.9));
    //dnn_trainer<net_type, adam> trainer(net,adam(0.0005, 0.9, 0.999));
    trainer.set_learning_rate(0.001);
    //trainer.set_min_learning_rate(0.0000001);
    //trainer.set_mini_batch_size(128);
    trainer.set_iterations_without_progress_threshold(10000);
    trainer.set_test_iterations_without_progress_threshold(1300);
    trainer.be_verbose();

    net.subnet().layer_details().set_num_outputs(trainingLabelsCount);

    // Since DNN training can take a long time, we can ask the trainer to save its state to
    // a file named "mnist_sync" every 20 seconds.  This way, if we kill this program and
    // start it again it will begin where it left off rather than restarting the training
    // from scratch.  This is because, when the program restarts, this call to
    // set_synchronization_file() will automatically reload the settings from mnist_sync if
    // the file exists.
    trainer.set_synchronization_file("co_rec_v9_sync", std::chrono::seconds(20));
    trainer.set_iterations_without_progress_threshold(10000);
    trainer.set_test_iterations_without_progress_threshold(1000);
    // Finally, this line begins training.  By default, it runs SGD with our specified
    // learning rate until the loss stops decreasing.  Then it reduces the learning rate by
    // a factor of 10 and continues running until the loss stops decreasing again.  It will
    // keep doing this until the learning rate has dropped below the min learning rate
    // defined above or the maximum number of epochs as been executed (defaulted to 10000).

    // it was default;
    //trainer.train(training_images, training_labels);

    std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long>         mini_batch_labels;

    cout << trainer << endl;
    int cnt = 1;
    const int MINIBATCH_SIZE = 128;

    dlib::rand rnd;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr_training(0, training_images.size() - MINIBATCH_SIZE); // define the range
    std::uniform_int_distribution<> distr_testing(0, testing_images.size() - MINIBATCH_SIZE);


    while(trainer.get_learning_rate() >= 1e-5)
    {
        if (cnt%30 != 0 || testing_images.size() == 0)
        {
            //cropper(27, images_train, boxes_train, mini_batch_samples, mini_batch_labels);
            //cropper(57, training_images, mini_batch_samples);
            int begin = distr_training(eng);
            //std::cout << "begin: " << begin << std::endl;
            mini_batch_samples.clear();
            mini_batch_labels.clear();
            copyVecFast(training_images, mini_batch_samples, begin + 0, begin + MINIBATCH_SIZE);
            copyVecFast(training_labels, mini_batch_labels, begin + 0, begin + MINIBATCH_SIZE);

            // for some reason works much worse
            /*while(mini_batch_samples.size() < 128)
                    {
                        auto idx = rnd.get_random_32bit_number()%training_images.size();
                        mini_batch_samples.push_back(training_images[idx]);
                        mini_batch_labels.push_back(training_labels[idx]);
                    }*/

            //parallel_for(mini_batch_samples.size(), mini_batch_samples.size() + minibatch_size, [&](long i){
            //
            // });
            // We can also randomly jitter the colors and that often helps a detector
            // generalize better to new images.
            for (auto&& img : mini_batch_samples)
                disturb_colors(img, rnd);

            //@DEBUG SHOW ME IMAGES PLS
            if(DEBUG_IMSHOW)
                for(int i =0; i < mini_batch_samples.size(); i++)
                {
                dlib::image_window win;
                win.set_image(mini_batch_samples[i]);
                std::cout << "label is: " << mini_batch_labels[i] << std::endl;

                cout << "Hit enter ..." << endl;
                cin.get();
                }

            trainer.train_one_step(mini_batch_samples, mini_batch_labels);
        }
        else
        {
            mini_batch_samples.clear();
            mini_batch_labels.clear();
            //cropper(27, images_test, boxes_test, mini_batch_samples, mini_batch_labels);
            int begin = distr_testing(eng);
            copyVecFast(testing_images, mini_batch_samples, begin + 0, begin + MINIBATCH_SIZE);
            copyVecFast(testing_labels, mini_batch_labels, begin + 0, begin + MINIBATCH_SIZE);

            // We can also randomly jitter the colors and that often helps a detector
            // generalize better to new images.
            for (auto&& img : mini_batch_samples)
                disturb_colors(img, rnd);

            trainer.test_one_step(mini_batch_samples, mini_batch_labels);
        }
        ++cnt;

    }

    // When you call train_one_step(), the trainer will do its processing in a
        // separate thread.  This allows the main thread to work on loading data
        // while the trainer is busy executing the mini-batches in parallel.
        // However, this also means we need to wait for any mini-batches that are
        // still executing to stop before we mess with the net object.  Calling
        // get_net() performs the necessary synchronization.
        trainer.get_net();


    // At this point our net object should have learned how to classify MNIST images.  But
    // before we try it out let's save it to disk.  Note that, since the trainer has been
    // running images through the network, net will have a bunch of state in it related to
    // the last batch of images it processed (e.g. outputs from each layer).  Since we
    // don't care about saving that kind of stuff to disk we can tell the network to forget
    // about that kind of transient data so that our file will be smaller.  We do this by
    // "cleaning" the network before saving it.
    net.clean();
    serialize("co_rec_v9.dat") << net;
    // Now if we later wanted to recall the network from disk we can simply say:
    // deserialize("mnist_network.dat") >> net;


    // Now let's run the training images through the network.  This statement runs all the
    // images through it and asks the loss layer to convert the network's raw output into
    // labels.  In our case, these labels are the numbers between 0 and 9.

    //const unsigned long number_of_classes2 = trainingLabelsCount;

    using test_net_type = loss_multiclass_log<fc<number_of_classes,
                                    avg_pool_everything<
                                    ares<ares<ares<ares_down<
                                    repeat<1,ares,
                                    ares_down<
                                    ares<
                                    input<matrix<unsigned char>>
                                    >>>>>>>>>>;
        // Then we can simply assign our trained net to our testing net.
        test_net_type tnet = net;


    std::vector<unsigned long> predicted_labels = tnet(training_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
    for (size_t i = 0; i < training_images.size(); ++i)
    {
        if (predicted_labels[i] == training_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    cout << "training num_right: " << num_right << endl;
    cout << "training num_wrong: " << num_wrong << endl;
    cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

    // Let's also see if the network can correctly classify the testing images.  Since
    // MNIST is an easy dataset, we should see at least 99% accuracy.
    predicted_labels = tnet(testing_images);
    num_right = 0;
    num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        if (predicted_labels[i] == testing_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    cout << "testing num_right: " << num_right << endl;
    cout << "testing num_wrong: " << num_wrong << endl;
    cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;


    // Finally, you can also save network parameters to XML files if you want to do
    // something with the network in another tool.  For example, you could use dlib's
    // tools/convert_dlib_nets_to_caffe to convert the network to a caffe model.
    net_to_xml(tnet, "tnetlenet.xml");
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

