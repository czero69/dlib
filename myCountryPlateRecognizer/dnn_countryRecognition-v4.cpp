#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <myPlatesDetector/myUtils.h>
#include <dlib/gui_widgets.h>
#include <random>
#include <utility>


//-----------------------------------------------------------------Config start

#define DEBUG_IMSHOW 0
#define SKIP_MULTILINE 1000
#define SKIP_SINGLELINE 1000
#define SYNC_FILENAME "test_sync"
#define NET_FILENAME "test.dat"

const int MINIBATCH_SIZE = 128;

const std::string dirnameTrainMulti = "/home/ecv/Pictures/LPR_Countries";
const std::string dirnameTestMulti = "/home/ecv/Pictures/LPR_Countries";
const std::string dirnameTrainSingle = "/home/ecv/projects/data/singleline";
const std::string dirnameTestSingle = "/home/ecv/projects/data/singleline";

std::vector <std::string> countries = {
  "A",
  "AL",
  "B",
  "BG",
  "BR",
  "BY",
  "CH",
  "CZ",
  "D",
  "DK",
  "E",
  "EC",
  "EST",
  "F",
  "FIN",
  "GB",
  "GR",
  "H",
  "HR",
  "I",
  "KZ",
  "L",
  "LT",
  "LV",
  "MC",
  "MD",
  "MNE",
  "N",
  "NL",
  "P",
  "PE",
  "PL",
  "RA",
  "RO",
  "RUS",
  "S",
  "SK",
  "UA",
  "USA"
};

//------------------------------------------------------------------Config end

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


std::vector <std::string> isMultilineVec = {"0", "1"};

std::map<std::string, std::vector<std::string> > labels = {{"country", countries}, {"isMultiline", isMultilineVec}};
loss_multimulticlass_log_ myloss(labels);

using net_type = loss_multimulticlass_log<fc<1,
                            avg_pool_everything<
                            res<res<res<res_down<
                            repeat<1,res, // repeat this layer 9 times//
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;



template <typename array_type>
void copyVecFast(const std::vector<array_type>& original, std::vector<array_type>& output, int begin =0, int end=0)
{
  assert(end - begin > 0);
  assert(end < original.size());

  output.clear();
  output.reserve(end - begin);
  copy(original.begin() + begin ,original.begin() + end,back_inserter(output));

}

template <typename array_type>
int load_images (array_type& images, std::vector<std::string> & country_labels,
                 const std::string& dirname, datasets_utils::dataType trainOrTest,
                 int skipStep = 0, int from_frame=0, int to_frame=std::numeric_limits<int>::max())
{
    using namespace dlib::image_dataset_metadata;

    // Set the current directory to be the one that contains the
    // metadata file. We do this because the file might contain
    // file paths which are relative to this folder.
    locally_change_current_dir chdir(directory(dirname));
    std::vector<std::pair<std::vector<std::string>, std::string>> filenames_vec;

    std::cout << dirname << std::endl;

    DIR *dir, *dir_inside;
    struct dirent *ent, *ent_inside;

    if ((dir = opendir (dirname.c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL) {
        printf ("%s, %u\n", ent->d_name, ent->d_type);
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
                filenames_vec.back().first.push_back(std::string(ent_inside->d_name));
              }
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
    std::vector<mmod_rect> rects;
    for(auto && outdir : filenames_vec)
        for(unsigned long i = std::max(int(0),int(from_frame)); i < std::min(int(outdir.first.size()),int(to_frame)); i += 1 + skipStep)
        {
            std::string fname = dirname + "/" + outdir.second + "/" + toString(trainOrTest) + "/" + outdir.first[i];
            images.push_back(fname);
            std::string token = getTokenFromString(outdir.first[i], delimiter, 2);
            country_labels.push_back(token);
        }

    return countries.size();
}



int main(int argc, char** argv) try
{
  std::cout << "Multi: dirnameTrain: " << dirnameTrainMulti << ", " << "dirnameTest"  <<  dirnameTestMulti  << std::endl;
  std::cout << "Single: dirnameTrain: " << dirnameTrainSingle << ", " << "dirnameTest"  <<  dirnameTestSingle  << std::endl;


  std::vector<std::string> training_images_filenames;
  std::vector<std::string>         country_training_labels;
  std::vector<std::string> testing_images_filenames;
  std::vector<std::string>         country_testing_labels;
  std::vector<std::map<std::string, std::string> > training_labels;
  std::vector<std::map<std::string, std::string> > testing_labels;


  int trainingLabelsCount = 0, testingLabelsCount = 0;

//----------------------------------------------------------
//LABELS (isMultiline):
// 0 - single
// 1 - multi
//----------------------------------------------------------

  trainingLabelsCount = load_images(training_images_filenames, country_training_labels, dirnameTrainMulti, dlib::datasets_utils::TRAIN, SKIP_MULTILINE);
  testingLabelsCount = load_images(testing_images_filenames, country_testing_labels, dirnameTestMulti, dlib::datasets_utils::TEST,  SKIP_MULTILINE);

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


  cout<<"TestMulti: "<<testing_images_filenames.size()<<" "<<testing_labels.size()<<endl;
  cout<<"TrainMulti: "<<training_images_filenames.size()<<" "<<training_labels.size()<<endl;


  trainingLabelsCount = load_images(training_images_filenames, country_training_labels, dirnameTrainSingle, dlib::datasets_utils::TRAIN, SKIP_SINGLELINE);
  testingLabelsCount = load_images(testing_images_filenames, country_testing_labels, dirnameTestSingle, dlib::datasets_utils::TEST, SKIP_SINGLELINE);

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


  cout<<"TestAll: "<<testing_images_filenames.size()<<" "<<testing_labels.size()<<endl;
  cout<<"TrainAll: "<<training_images_filenames.size()<<" "<<training_labels.size()<<endl;

  assert(trainingLabelsCount == testingLabelsCount);

  std::cout << "trainingLabelsCount: " << trainingLabelsCount <<  std::endl;

//---------------------------------------------SHUFFLE IMAGES---------------------------------------------------

  auto seed = unsigned ( std::time(0) );

  std::vector<int> numbers;
  std::vector<std::string> temp_training_images_filenames = training_images_filenames;
  std::vector<std::string> temp_testing_images_filenames = testing_images_filenames;
  std::vector<std::map<std::string, std::string> > temp_training_labels = training_labels;
  std::vector<std::map<std::string, std::string> > temp_testing_labels = testing_labels;

  for(int i = 0; i < training_images_filenames.size(); ++i)
    numbers.push_back(i);

  std::srand ( seed );
  std::random_shuffle ( numbers.begin(), numbers.end() );

  for(int i = 0; i < numbers.size(); ++i)
  {
    training_images_filenames[i] = temp_training_images_filenames[numbers[i]];
    training_labels[i] = temp_training_labels[numbers[i]];
  }

  numbers.clear();
  for(int i = 0; i < testing_images_filenames.size(); ++i)
    numbers.push_back(i);
  std::srand ( seed );
  std::random_shuffle ( numbers.begin(), numbers.end() );

  for(int i = 0; i < numbers.size(); ++i)
  {
    testing_images_filenames[i] = temp_testing_images_filenames[numbers[i]];
    testing_labels[i] = temp_testing_labels[numbers[i]];
  }
  temp_testing_images_filenames.clear();
  temp_testing_labels.clear();
  temp_training_images_filenames.clear();
  temp_training_labels.clear();

  int num_of_training_samples = training_images_filenames.size();
  int num_of_testing_samples = testing_images_filenames.size();

//----------------------------------------LOADING & TRAINING--------------------------------------

  net_type net(myloss);
  net.subnet().layer_details().set_num_outputs(countries.size() + 2);

  dnn_trainer<net_type,adam> trainer(net,adam(0.0005, 0.9, 0.999));
  trainer.set_learning_rate(0.001);
  trainer.be_verbose();

  trainer.set_synchronization_file((SYNC_FILENAME), std::chrono::seconds(20));
  trainer.set_iterations_without_progress_threshold(10000);
  trainer.set_test_iterations_without_progress_threshold(1000);

  std::vector<matrix<unsigned char>> mini_batch_samples;
  std::vector<std::map<std::string, std::string> > mini_batch_labels;

  cout << trainer << endl;

  dlib::pipe<std::pair<std::string, std::map<std::string, std::string> > > load_training_data(num_of_training_samples);
  dlib::pipe<std::pair<std::string, std::map<std::string, std::string> > > load_testing_data(num_of_testing_samples);

  for(uint i = 0; i < num_of_training_samples; ++i)
  {
      load_training_data.enqueue(make_pair(training_images_filenames[i], training_labels[i]));
  }

  for(uint i = 0; i < num_of_testing_samples; ++i)
  {
      load_testing_data.enqueue(make_pair(testing_images_filenames[i], testing_labels[i]));
  }


  dlib::pipe<std::pair<matrix<unsigned char>, std::map<std::string, std::string> > > training_data(2*MINIBATCH_SIZE);
  dlib::pipe<std::pair<matrix<unsigned char>, std::map<std::string, std::string> > > testing_data(num_of_testing_samples);

  auto f = [&load_training_data, &training_data]()
  {
      matrix<unsigned char> img;
      std::pair<std::string, std::map<std::string, std::string> > load;
      while(load_training_data.is_enabled() && training_data.is_enabled())
      {
          load_training_data.dequeue(load);
          load_image(img, load.first);
          matrix<unsigned char> imgTmp(15*2,60*2);
          resize_image(img, imgTmp);
          training_data.enqueue(make_pair(imgTmp, load.second));
      }
  };

  std::thread data_loader1([f](){ f(); });
  std::thread data_loader2([f](){ f(); });
  std::thread data_loader3([f](){ f(); });
  std::thread data_loader4([f](){ f(); });

  auto g = [&load_testing_data, &testing_data]()
  {
      matrix<unsigned char> img;
      std::pair<std::string, std::map<std::string, std::string> > load;
      while(load_testing_data.is_enabled() && testing_data.is_enabled())
      {
          load_testing_data.dequeue(load);
          load_image(img, load.first);
          matrix<unsigned char> imgTmp(15*2,60*2);
          resize_image(img, imgTmp);
          testing_data.enqueue(make_pair(imgTmp, load.second));
      }
  };

  std::thread data_loader5([g](){ g(); });
  std::thread data_loader6([g](){ g(); });

  training_labels.clear();
  testing_labels.clear();
  std::vector<matrix<unsigned char> > training_images;
  std::vector<matrix<unsigned char> > testing_images;

  for(int i = 0; i < MINIBATCH_SIZE; ++i)
  {
      std::pair<matrix<unsigned char>, std::map<std::string, std::string> > load;
      training_data.dequeue(load);
      training_images.push_back(load.first);
      training_labels.push_back(load.second);
      testing_data.dequeue(load);
      testing_images.push_back(load.first);
      testing_labels.push_back(load.second);
  }

  int cnt = 1;
  dlib::rand rnd;
  srand(time(NULL));
  std::pair<matrix<unsigned char>, std::map<std::string, std::string> > load;

  while(trainer.get_learning_rate() >= 1e-5)
  {
      if (cnt%30 != 0 || num_of_testing_samples == 0)
      {
          if(training_data.size() != 0)
          {
              int n = std::min((int)training_data.size(), MINIBATCH_SIZE);
              for(int i = 0; i < n; ++i)
              {
                  training_data.dequeue(load);
                  training_images.push_back(load.first);
                  training_labels.push_back(load.second);
              }
          }
          int begin = std::rand()%(training_images.size()-MINIBATCH_SIZE+1);
          mini_batch_samples.clear();
          mini_batch_labels.clear();
          copyVecFast(training_images, mini_batch_samples, begin + 0, begin + MINIBATCH_SIZE);
          copyVecFast(training_labels, mini_batch_labels, begin + 0, begin + MINIBATCH_SIZE);

          for (auto&& img : mini_batch_samples)
              disturb_colors(img, rnd);

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
          if(testing_data.size() != 0)
          {
              int n = std::min((int)testing_data.size(), MINIBATCH_SIZE);
              for(int i = 0; i < n; ++i)
              {
                  testing_data.dequeue(load);
                  testing_images.push_back(load.first);
                  testing_labels.push_back(load.second);
              }
          }
          mini_batch_samples.clear();
          mini_batch_labels.clear();
          int begin = std::rand()%(testing_images.size()-MINIBATCH_SIZE+1);
          copyVecFast(testing_images, mini_batch_samples, begin + 0, begin + MINIBATCH_SIZE);
          copyVecFast(testing_labels, mini_batch_labels, begin + 0, begin + MINIBATCH_SIZE);

          for (auto&& img : mini_batch_samples)
              disturb_colors(img, rnd);

          trainer.test_one_step(mini_batch_samples, mini_batch_labels);
      }
      ++cnt;

  }

  matrix<unsigned char> temp_img;
  std::pair<std::string, std::map<std::string, std::string> > temp_load;
  while(load_testing_data.size() > 0)
  {
      load_testing_data.dequeue(temp_load);
      load_image(temp_img, temp_load.first);
      matrix<unsigned char> imgTmp(15*2,60*2);
      resize_image(temp_img, imgTmp);
      testing_data.enqueue(make_pair(imgTmp, temp_load.second));
  }

  while(testing_data.size() > 0)
  {
      testing_data.dequeue(load);
      testing_images.push_back(load.first);
      testing_labels.push_back(load.second);
  }

  load_training_data.disable();
  load_testing_data.disable();
  training_data.disable();
  testing_data.disable();
  data_loader1.join();
  data_loader2.join();
  data_loader3.join();
  data_loader4.join();
  data_loader5.join();
  data_loader6.join();


  trainer.get_net();

  net.clean();
  serialize(NET_FILENAME) << net;


//---------------------------------------------TESTING-----------------------------------

  using test_net_type = loss_multimulticlass_log<fc<1,
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
  int multi_country_right = 0;
  int multi_multiline_right = 0;
  int single_country_right = 0;
  int single_multiline_right = 0;
  int multi_country_wrong = 0;
  int multi_multiline_wrong = 0;
  int single_country_wrong = 0;
  int single_multiline_wrong = 0;

  for(size_t i = 0; i < training_images.size(); ++i)
  {
      if(training_labels[i]["isMultiline"] == "1")
      {
          if(predicted_labels[i]["isMultiline"] == training_labels[i]["isMultiline"])
              multi_multiline_right++;
          else
              multi_multiline_wrong++;

          if(predicted_labels[i]["country"] == training_labels[i]["country"])
              multi_country_right++;
          else
              multi_country_wrong++;
      }
      else
      {
          if(predicted_labels[i]["isMultiline"] == training_labels[i]["isMultiline"])
              single_multiline_right++;
          else
              single_multiline_wrong++;

          if(predicted_labels[i]["country"] == training_labels[i]["country"])
              single_country_right++;
          else
              single_country_wrong++;
      }
  }
  cout<<endl<<"Training set:"<<endl;
  cout<<"multi_country_right: "<<multi_country_right<<endl;
  cout<<"multi_country_wrong: "<<multi_country_wrong<<endl;
  cout<<"multi_multiline_right: "<<multi_multiline_right<<endl;
  cout<<"multi_multiline_wrong: "<<multi_multiline_wrong<<endl;
  cout<<"single_country_right: "<<single_country_right<<endl;
  cout<<"single_country_wrong: "<<single_country_wrong<<endl;
  cout<<"single_multiline_right: "<<single_multiline_right<<endl;
  cout<<"single_multiline_wrong: "<<single_multiline_wrong<<endl;

  cout<<endl<<"Multiline:"<<endl;
  cout<<"Country accuracy: "<<multi_country_right/(double)(multi_country_right+multi_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<multi_multiline_right/(double)(multi_multiline_right+multi_multiline_wrong)<<endl;
  cout<<"Singleline:"<<endl;
  cout<<"Country accuracy: "<<single_country_right/(double)(single_country_right+single_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<single_multiline_right/(double)(single_multiline_right+single_multiline_wrong)<<endl;
  cout<<"Total:"<<endl;
  cout<<"Country accuracy: "<<(multi_country_right + single_country_right)/(double)(single_country_right
                              +single_country_wrong+multi_country_right+multi_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<(multi_multiline_right+single_multiline_right)/(double)(single_multiline_right
                              +single_multiline_wrong+multi_multiline_right+multi_multiline_wrong)<<endl;


  raw_predicted_labels = tnet(testing_images);
  predicted_labels.clear();
  for(size_t i = 0; i < raw_predicted_labels.size(); ++i)
  {
    tempMap.clear();
    tempMap.insert(make_pair("country", raw_predicted_labels[i]["country"]));
    tempMap.insert(make_pair("isMultiline", raw_predicted_labels[i]["isMultiline"]));
    predicted_labels.push_back(tempMap);
  }


  multi_country_right = 0;
  multi_multiline_right = 0;
  single_country_right = 0;
  single_multiline_right = 0;
  multi_country_wrong = 0;
  multi_multiline_wrong = 0;
  single_country_wrong = 0;
  single_multiline_wrong = 0;
  for (size_t i = 0; i < testing_images.size(); ++i)
  {
      if(testing_labels[i]["isMultiline"] == "1")
      {
          if(predicted_labels[i]["isMultiline"] == testing_labels[i]["isMultiline"])
              multi_multiline_right++;
          else
              multi_multiline_wrong++;

          if(predicted_labels[i]["country"] == testing_labels[i]["country"])
              multi_country_right++;
          else
              multi_country_wrong++;
      }
      else
      {
          if(predicted_labels[i]["isMultiline"] == testing_labels[i]["isMultiline"])
              single_multiline_right++;
          else
              single_multiline_wrong++;

          if(predicted_labels[i]["country"] == testing_labels[i]["country"])
              single_country_right++;
          else
              single_country_wrong++;
      }
  }
  cout<<endl<<"Testing set:"<<endl;
  cout<<"multi_country_right: "<<multi_country_right<<endl;
  cout<<"multi_country_wrong: "<<multi_country_wrong<<endl;
  cout<<"multi_multiline_right: "<<multi_multiline_right<<endl;
  cout<<"multi_multiline_wrong: "<<multi_multiline_wrong<<endl;
  cout<<"single_country_right: "<<single_country_right<<endl;
  cout<<"single_country_wrong: "<<single_country_wrong<<endl;
  cout<<"single_multiline_right: "<<single_multiline_right<<endl;
  cout<<"single_multiline_wrong: "<<single_multiline_wrong<<endl;

  cout<<endl<<"Multiline:"<<endl;
  cout<<"Country accuracy: "<<multi_country_right/(double)(multi_country_right+multi_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<multi_multiline_right/(double)(multi_multiline_right+multi_multiline_wrong)<<endl;
  cout<<"Singleline:"<<endl;
  cout<<"Country accuracy: "<<single_country_right/(double)(single_country_right+single_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<single_multiline_right/(double)(single_multiline_right+single_multiline_wrong)<<endl;
  cout<<"Total:"<<endl;
  cout<<"Country accuracy: "<<(multi_country_right + single_country_right)/(double)(single_country_right
                              +single_country_wrong+multi_country_right+multi_country_wrong)<<endl;
  cout<<"Multiline accuracy: "<<(multi_multiline_right+single_multiline_right)/(double)(single_multiline_right
                              +single_multiline_wrong+multi_multiline_right+multi_multiline_wrong)<<endl;
}
catch(std::exception& e)
{
  cout << e.what() << endl;
}
