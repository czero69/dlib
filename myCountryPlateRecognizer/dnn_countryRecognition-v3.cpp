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

std::vector <std::string> countries = {
  "BY",
  "CH",
  "CZ",
  "D",
  "E",
  "EST",
  "F",
  "GB",
  "H",
  "I",
  "KZ",
  "LT",
  "LV",
  "MC",
  "NL",
  "PL",
  "RO",
  "RUS",
  "S",
  "SK",
  "UA"
};
std::vector <std::string> isMultilineVec = {"0", "1"};

std::map<std::string, std::vector<std::string> > labels = {{"country", countries}, {"isMultiline", isMultilineVec}};
loss_multimulticlass_log_ myloss(labels);

const int number_of_classes = 23;

using net_type = loss_multimulticlass_log<fc<number_of_classes,
                            avg_pool_everything<
                            res<res<res<res_down<
                            repeat<1,res, // repeat this layer 9 times//
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;



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


std::map<int, std::string> lpCountryCodes = {
  { 0, "BY" },
  { 1, "CH"},
  { 2, "CZ" },
  { 3, "D" },
  { 4, "E" },
  { 5, "EST"},
  { 6, "F" },
  { 7, "GB" },
  { 8, "H" },
  { 9, "I" },
  { 10, "KZ" },
  { 11, "LT" },
  { 12, "LV" },
  { 13, "MC" },
  { 14, "NL" },
  { 15, "PL" },
  { 16, "RO" },
  { 17, "RUS" },
  { 18, "S"},
  { 19, "SK" },
  { 20, "UA" }
};


template <
    typename array_type
    >
int load_images (
    array_type& images,
    std::vector<std::string> & country_labels,
        const std::string& dirname,
        datasets_utils::dataType trainOrTest,
        int skipStep = 0,
        int from_frame=0, int to_frame=std::numeric_limits<int>::max()
)
{
    //images.clear();
    //object_locations.clear();

    using namespace dlib::image_dataset_metadata;

    // Set the current directory to be the one that contains the
    // metadata file. We do this because the file might contain
    // file paths which are relative to this folder.
    locally_change_current_dir chdir(directory(dirname));
    //std::vector<char[256]> filename_list;
    std::vector<std::pair<std::vector<std::string>, std::string>> filenames_vec;

    std::cout << dirname << std::endl;

    DIR *dir, *dir_inside;
    struct dirent *ent, *ent_inside;
    // unsigned int class_id = 0;

    if ((dir = opendir (dirname.c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL) {
        printf ("%s, %u\n", ent->d_name, ent->d_type);
        //filename_list.push_back(ent->d_name);
        if( strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 )
        {
          std::vector<std::string> newVec;
          filenames_vec.push_back(std::make_pair(newVec, std::string(ent->d_name)));
          if((dir_inside = opendir ((dirname + "/" + std::string(ent->d_name) +  "/" + toString(trainOrTest)).c_str())) != NULL)
          {
            /* print all the files and directories within directory */
            while ((ent_inside = readdir (dir_inside)) != NULL)
            {
              if( strcmp(ent_inside->d_name, ".") != 0 && strcmp(ent_inside->d_name, "..") != 0 &&
                (checkExtension(std::string(ent_inside->d_name), "png") ||  checkExtension(std::string(ent_inside->d_name), "jpg")))
              {
                //printf ("%s, %u\n", ent_inside->d_name, ent_inside->d_type);
                filenames_vec.back().first.push_back(std::string(ent_inside->d_name));
              }
              //std::string(ent->d_name) + "/ok/" + std::string(ent_inside->d_name)
            }
            closedir (dir_inside);
          }
          else
          {
            /* could not open directory */
            perror ("empty inner dir, no files to load");
            return -1;
          }
        }
      }
    closedir (dir);
    }
    else
    {
      /* could not open directory */
      perror ("empty dir, no files to load");
      return -1;
    }

    const std::string delimiter = "-";

    typedef typename array_type::value_type image_type;
    image_type img;
    //dlib::set_image_size(imgTmp, 15, 60);
    std::vector<mmod_rect> rects;
    //for (unsigned long i = 0; i < data.images.size(); ++i)
    for(auto && outdir : filenames_vec)
        for(unsigned long i = std::max(int(0),int(from_frame)); i < std::min(int(outdir.first.size()),int(to_frame)); i += 1 + skipStep)
        {
            std::string fname = dirname + "/" + outdir.second + "/" + toString(trainOrTest) + "/" + outdir.first[i];
            //std::cout << "file to load: " << fname << std::endl;
            load_image(img, fname);
            image_type imgTmp(15*2,60*2);
            resize_image(img, imgTmp);
            images.push_back(std::move(imgTmp));
            std::string token = getTokenFromString(outdir.first[i], delimiter, 2);
            country_labels.push_back(token);
        }

    return lpCountryCodes.size();
}



int main(int argc, char** argv) try
{
  //if (argc != 3)
  //{
    //  cout << "./dnn_countryRecognition path-to-trainingDir path-to-testingDir" << endl;
    //  return 1;
  //}

  //const std::string dirnameTrain = argv[1];
  //const std::string dirnameTest = argv[2];

  const std::string dirnameTrainMulti = "/home/ecv/Pictures/LPR_Countries";
  const std::string dirnameTestMulti = "/home/ecv/Pictures/LPR_Countries";
  const std::string dirnameTrainSingle = "/home/ecv/data/singleline";
  const std::string dirnameTestSingle = "/home/ecv/data/singleline";


  std::cout << "Multi: dirnameTrain: " << dirnameTrainMulti << ", " << "dirnameTest"  <<  dirnameTestMulti  << std::endl;
  std::cout << "Single: dirnameTrain: " << dirnameTrainSingle << ", " << "dirnameTest"  <<  dirnameTestSingle  << std::endl;


  std::vector<matrix<unsigned char>> training_images;
  std::vector<std::string>         country_training_labels;
  std::vector<matrix<unsigned char>> testing_images;
  std::vector<std::string>         country_testing_labels;
  std::vector<std::map<std::string, std::string> > training_labels;
  std::vector<std::map<std::string, std::string> > testing_labels;


  int trainingLabelsCount = 0, testingLabelsCount = 0;

//----------------------------------------------------------
//LABELS (isMultiline):
// 0 - single
// 1 - multi
//----------------------------------------------------------

  trainingLabelsCount = load_images(training_images, country_training_labels, dirnameTrainMulti, dlib::datasets_utils::TRAIN, 5);
  testingLabelsCount = load_images(testing_images, country_testing_labels, dirnameTestMulti, dlib::datasets_utils::TEST,  5);

  int trainMultiSize = country_training_labels.size();
  int testMultiSize = country_testing_labels.size();

  std::map<std::string, std::string> tempMap;
  for(int i = 0; i < trainMultiSize; ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", country_training_labels[i]));
    tempMap.insert(make_pair("isMultiline", "1"));
    training_labels.push_back(tempMap);
  }

  for(int i = 0; i < testMultiSize; ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", country_testing_labels[i]));
    tempMap.insert(make_pair("isMultiline", "1"));
    testing_labels.push_back(tempMap);
  }


  cout<<"TestMulti: "<<testing_images.size()<<" "<<testing_labels.size()<<endl;
  cout<<"TrainMulti: "<<training_images.size()<<" "<<training_labels.size()<<endl;


  trainingLabelsCount = load_images(training_images, country_training_labels, dirnameTrainSingle, dlib::datasets_utils::TRAIN, 10);
  testingLabelsCount = load_images(testing_images, country_testing_labels, dirnameTestSingle, dlib::datasets_utils::TEST, 10);

  for(int i = trainMultiSize; i < country_training_labels.size(); ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", country_training_labels[i]));
    tempMap.insert(make_pair("isMultiline", "0"));
    training_labels.push_back(tempMap);
  }

  for(int i = testMultiSize; i < country_testing_labels.size(); ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", country_testing_labels[i]));
    tempMap.insert(make_pair("isMultiline", "0"));
    testing_labels.push_back(tempMap);
  }


  cout<<"TestAll: "<<testing_images.size()<<" "<<testing_labels.size()<<endl;
  cout<<"TrainAll: "<<training_images.size()<<" "<<training_labels.size()<<endl;


//--------------------------------------------------------------------------------------------------------------------
  assert(trainingLabelsCount == testingLabelsCount);

  std::cout << "trainingLabelsCount: " << trainingLabelsCount <<  std::endl;

  auto seed = unsigned ( std::time(0) );


  std::vector<int> numbers;
  std::vector<matrix<unsigned char>> temp_training_images = training_images;
  std::vector<matrix<unsigned char>> temp_testing_images = testing_images;
  std::vector<std::map<std::string, std::string> > temp_training_labels = training_labels;
  std::vector<std::map<std::string, std::string> > temp_testing_labels = testing_labels;

  for(int i = 0; i < training_images.size(); ++i)
    numbers.push_back(i);

  std::srand ( seed );
  std::random_shuffle ( numbers.begin(), numbers.end() );

  for(int i = 0; i < numbers.size(); ++i)
  {
    training_images[i] = temp_training_images[numbers[i]];
    training_labels[i] = temp_training_labels[numbers[i]];
  }

  numbers.clear();
  for(int i = 0; i < testing_images.size(); ++i)
    numbers.push_back(i);
  std::srand ( seed );
  std::random_shuffle ( numbers.begin(), numbers.end() );

  for(int i = 0; i < numbers.size(); ++i)
  {
    testing_images[i] = temp_testing_images[numbers[i]];
    testing_labels[i] = temp_testing_labels[numbers[i]];
  }
  temp_testing_images.clear();
  temp_testing_labels.clear();
  temp_training_images.clear();
  temp_training_labels.clear();


  bool DEBUG_IMSHOW = 0;

  if(DEBUG_IMSHOW)
      for(int i =0; i < testing_images.size(); i++)
      {
          dlib::image_window win;
          win.set_image(testing_images[i]);
          std::cout << "label is: " << testing_labels[i]["country"] << "_" << testing_labels[i]["isMultiline"] << std::endl;

          cout << "Hit enter ..." << endl;
          cin.get();

          dlib::image_window win2;
          win2.set_image(training_images[i]);
          std::cout << "label is: " << training_labels[i]["country"] << "_" << training_labels[i]["isMultiline"] << std::endl;

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
  net_type net(myloss);

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

  //net.subnet().layer_details().set_num_outputs(trainingLabelsCount);

  // Since DNN training can take a long time, we can ask the trainer to save its state to
  // a file named "mnist_sync" every 20 seconds.  This way, if we kill this program and
  // start it again it will begin where it left off rather than restarting the training
  // from scratch.  This is because, when the program restarts, this call to
  // set_synchronization_file() will automatically reload the settings from mnist_sync if
  // the file exists.
  trainer.set_synchronization_file(("full_corec_isMulti_multiSkip5_singleSkip10_sync"), std::chrono::seconds(20));
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
  std::vector<std::map<std::string, std::string> > mini_batch_labels;

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
          ////////
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
              std::cout << "label is: " << mini_batch_labels[i]["country"] <<"_" << mini_batch_labels[i]["isMultiline"] << std::endl;

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
  serialize("full_corec_isMulti_multiSkip5_singleSkip10.dat") << net;
  // Now if we later wanted to recall the network from disk we can simply say:
  // deserialize("mnist_network.dat") >> net;


  // Now let's run the training images through the network.  This statement runs all the
  // images through it and asks the loss layer to convert the network's raw output into
  // labels.  In our case, these labels are the numbers between 0 and 9.

  //const unsigned long number_of_classes2 = trainingLabelsCount;

  using test_net_type = loss_multimulticlass_log<fc<number_of_classes,
                                  avg_pool_everything<
                                  ares<ares<ares<ares_down<
                                  repeat<1,ares,
                                  ares_down<
                                  ares<
                                  input<matrix<unsigned char>>
                                  >>>>>>>>>>;
      // Then we can simply assign our trained net to our testing net.
      test_net_type tnet = net;


  std::vector<std::map<std::string, dlib::loss_multimulticlass_log_::classifier_output> > raw_predicted_labels = tnet(training_images);
  std::vector<std::map<std::string, std::string> > predicted_labels;


  for(size_t i = 0; i < raw_predicted_labels.size(); ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", raw_predicted_labels[i]["country"]));
    tempMap.insert(make_pair("isMultiline", raw_predicted_labels[i]["isMultiline"]));
    predicted_labels.push_back(tempMap);
  }
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
  raw_predicted_labels = tnet(testing_images);
  predicted_labels.clear();
  for(size_t i = 0; i < raw_predicted_labels.size(); ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", raw_predicted_labels[i]["country"]));
    tempMap.insert(make_pair("isMultiline", raw_predicted_labels[i]["isMultiline"]));
    predicted_labels.push_back(tempMap);
  }


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
