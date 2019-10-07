#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <myPlatesDetector/myUtils.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
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
  "A",
  "AL",
  "B",
  "BG",
  "BR",
  "BY",
  "CH",
  "CZ",
  "D",
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
  "N",
  "NL",
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
std::vector <std::string> isMultilineVec = {"0", "1"};

std::map<std::string, std::vector<std::string> > labels = {{"country", countries}, {"isMultiline", isMultilineVec}};
loss_multimulticlass_log_ myloss(labels);

const int number_of_classes = 1;

using net_type = loss_multimulticlass_log<fc<number_of_classes,
                            avg_pool_everything<
                            res<res<res<res_down<
                            repeat<1,res, // repeat this layer 9 times//
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;

net_type net(myloss);

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
  { 0, "A" },
  { 1, "AL"},
  { 2, "B" },
  { 3, "BG" },
  { 4, "BR" },
  { 5, "BY" },
  { 6, "CH" },
  { 7, "CZ"},
  { 8, "D" },
  { 9, "E" },
  { 10, "EC" },
  { 11, "EST" },
  { 12, "F" },
  { 13, "FIN" },
  { 14, "GB" },
  { 15, "GR" },
  { 16, "H" },
  { 17, "HR"},
  { 18, "I" },
  { 19, "KZ" },
  { 20, "L" },
  { 21, "LT" },
  { 22, "LV"},
  { 23, "MC" },
  { 24, "N" },
  { 25, "NL" },
  { 26, "PE" },
  { 27, "PL" },
  { 28, "RA" },
  { 29, "RO" },
  { 30, "RUS" },
  { 31, "S" },
  { 32, "SK" },
  { 33, "UA" },
  { 34, "USA" }
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
  if(argc < 2)
  {
    cout<<"No input"<<endl;
    return 0;
  }
  deserialize(argv[1]) >> net;

  bool WRONG_IMSHOW = 0;
  bool DEBUG_IMSHOW = 0;
  if(argc > 2)
    WRONG_IMSHOW = (argv[2][0] == '1');
  if(argc > 3)
    DEBUG_IMSHOW = (argv[3][0] == '1');

  const std::string dirnameTrainMulti = "/home/ecv/Pictures/LPR_Countries";
  const std::string dirnameTestMulti = "/home/ecv/Pictures/LPR_Countries";
  const std::string dirnameTrainSingle = "/home/ecv/projects/data/singleline";
  const std::string dirnameTestSingle = "/home/ecv/projects/data/singleline";


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

  trainingLabelsCount = load_images(training_images, country_training_labels, dirnameTrainMulti, dlib::datasets_utils::TRAIN, 499);
  testingLabelsCount = load_images(testing_images, country_testing_labels, dirnameTestMulti, dlib::datasets_utils::TEST,  499);

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


  trainingLabelsCount = load_images(training_images, country_training_labels, dirnameTrainSingle, dlib::datasets_utils::TRAIN, 499);
  testingLabelsCount = load_images(testing_images, country_testing_labels, dirnameTestSingle, dlib::datasets_utils::TEST, 499);

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

  //timespec startTime;
  //alpr::getTimeMonotonic(&startTime);
  std::vector<std::map<std::string, dlib::loss_multimulticlass_log_::classifier_output> > raw_predicted_labels = tnet(training_images);
  timespec endTime;
  //alpr::getTimeMonotonic(&endTime);
  //cout<<"Training set avg time: "<<alpr::diffclock(startTime, endTime)/training_images.size()<<endl;
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
  int multi_right = 0;
  int multi_wrong = 0;
  int country_right = 0;
  int country_wrong = 0;
  // And then let's see if it classified them correctly.
  for (size_t i = 0; i < training_images.size(); ++i)
  {
      if (predicted_labels[i] == training_labels[i])
          ++num_right;
      else
      {
          if(WRONG_IMSHOW)
          {
            dlib::image_window win;
            win.set_image(training_images[i]);
            std::cout << "Label is: " << training_labels[i]["country"] << "_" << training_labels[i]["isMultiline"] << std::endl;
            std::cout << "Predicted label is: " << predicted_labels[i]["country"] << "_" << predicted_labels[i]["isMultiline"] << std::endl;

            cout << "Hit enter ..." << endl;
            cin.get();
          }
          ++num_wrong;
      }
      if(predicted_labels[i]["country"] == training_labels[i]["country"])
        ++country_right;
      else
        ++country_wrong;
      if(predicted_labels[i]["isMultiline"] == training_labels[i]["isMultiline"])
        ++multi_right;
      else
        ++multi_wrong;
  }
  cout << "training num_right: " << num_right << endl;
  cout << "training num_wrong: " << num_wrong << endl;
  cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
  cout << "multiline accuracy: " << multi_right/(double)(multi_right+multi_wrong) << endl;
  cout << "country accuracy:   " << country_right/(double)(country_right+country_wrong) << endl;

  //alpr::getTimeMonotonic(&startTime);
  raw_predicted_labels = tnet(testing_images);
  //alpr::getTimeMonotonic(&endTime);
  //cout<<"Testing avg time: "<<alpr::diffclock(startTime, endTime)/testing_images.size()<<endl;


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
  multi_right = 0;
  multi_wrong = 0;
  country_right = 0;
  country_wrong = 0;
  for (size_t i = 0; i < testing_images.size(); ++i)
  {
      if (predicted_labels[i] == testing_labels[i])
          ++num_right;
      else
      {
          if(WRONG_IMSHOW)
          {
              dlib::image_window win;
              win.set_image(testing_images[i]);
              std::cout << "Label is: " << testing_labels[i]["country"] << "_" << testing_labels[i]["isMultiline"] << std::endl;
              std::cout << "Predicted label is: " << predicted_labels[i]["country"] << "_" << predicted_labels[i]["isMultiline"] << std::endl;

              cout << "Hit enter ..." << endl;
              cin.get();
          }
          ++num_wrong;
      }
      if(predicted_labels[i]["country"] == testing_labels[i]["country"])
        country_right++;
      else
        country_wrong++;
      if(predicted_labels[i]["isMultiline"] == testing_labels[i]["isMultiline"])
        multi_right++;
      else
        multi_wrong++;
  }
  cout << "testing num_right: " << num_right << endl;
  cout << "testing num_wrong: " << num_wrong << endl;
  cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
  cout << "multiline accuracy:" << multi_right/(double)(multi_right+multi_wrong) << endl;
  cout << "country accuracy:  " << country_right/(double)(country_right+country_wrong) << endl;
}
catch(std::exception& e)
{
  cout << e.what() << endl;
}
