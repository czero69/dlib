#ifndef DLIB_LOAD_IMAGE_DaTASET_Subset_Hh_
#define DLIB_LOAD_IMAGE_DaTASET_Subset_Hh_


#include "dlib/data_io/load_image_dataset_abstract.h"
#include "dlib/misc_api.h"
#include "dlib/dir_nav.h"
#include "dlib/image_io.h"
#include "dlib/array.h"
#include <vector>
#include "dlib/geometry.h"
#include "dlib/data_io/image_dataset_metadata.h"
#include <string>
#include <set>
#include "dlib/image_processing/full_object_detection.h"
#include <utility>
#include <limits>
#include "dlib/image_transforms/image_pyramid.h"

#include "dlib/data_io/load_image_dataset.h"

#include <map>
//#include "dlib/data_io/load_image_dataset_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline size_t num_non_ignored_boxes_subset (const std::vector<mmod_rect>& rects)
        {
            size_t cnt = 0;
            for (auto& b : rects)
            {
                if (!b.ignore)
                    cnt++;
            }
            return cnt;
        }

        inline size_t num_non_ignored_boxes (const std::vector<std::pair<mmod_rect, std::vector<dlib::point>>>& rects)
        {
            size_t cnt = 0;
            for (auto& b : rects)
            {
                if (!b.first.ignore)
                    cnt++;
            }
            return cnt;
        }
    }

    int image_dataset_get_dataset_size (
        const image_dataset_file& source
    )
    {
        using namespace dlib::image_dataset_metadata;
        dataset data;
        load_image_dataset_metadata(data, source.get_filename());
        return data.images.size();
    }

    // no parts version
    template <
        typename array_type
        >
    void load_image_dataset_subset (
        array_type& images,
        std::vector<std::vector<dlib::mmod_rect>> & object_locations,
        const image_dataset_file& source,
            unsigned long from_frame, unsigned long to_frame
    )
    {
        //images.clear();
        //object_locations.clear();

        using namespace dlib::image_dataset_metadata;
        dataset data;
        load_image_dataset_metadata(data, source.get_filename());

        // Set the current directory to be the one that contains the
        // metadata file. We do this because the file might contain
        // file paths which are relative to this folder.
        locally_change_current_dir chdir(get_parent_directory(file(source.get_filename())));

        typedef typename array_type::value_type image_type;
        image_type img;
        std::vector<mmod_rect> rects;
        //for (unsigned long i = 0; i < data.images.size(); ++i)
        for(unsigned long i = std::max(int(0),int(from_frame)); i < std::min(int(data.images.size()),int(to_frame)); ++i)
        {
            double min_rect_size = std::numeric_limits<double>::infinity();
            rects.clear();
            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                if (source.should_load_box(data.images[i].boxes[j]))
                {
                    if (data.images[i].boxes[j].ignore)
                    {
                        rects.push_back(ignored_mmod_rect(data.images[i].boxes[j].rect));
                    }
                    else
                    {
                        rects.push_back(mmod_rect(data.images[i].boxes[j].rect));
                        min_rect_size = std::min<double>(min_rect_size, rects.back().rect.area());
                    }
                    rects.back().label = data.images[i].boxes[j].label;
                }
            }

            if (!source.should_skip_empty_images() || impl::num_non_ignored_boxes(rects) != 0)
            {
                load_image(img, data.images[i].filename);
                if (rects.size() != 0)
                {
                    // if shrinking the image would still result in the smallest box being
                    // bigger than the box area threshold then shrink the image.
                    while(min_rect_size/2/2 > source.box_area_thresh())
                    {
                        pyramid_down<2> pyr;
                        pyr(img);
                        min_rect_size *= (1.0/2.0)*(1.0/2.0);
                        for (auto&& r : rects)
                        {
                            r.rect = pyr.rect_down(r.rect);
                        }
                    }
                    while(min_rect_size*(2.0/3.0)*(2.0/3.0) > source.box_area_thresh())
                    {
                        pyramid_down<3> pyr;
                        pyr(img);
                        min_rect_size *= (2.0/3.0)*(2.0/3.0);
                        for (auto&& r : rects)
                            r.rect = pyr.rect_down(r.rect);
                        for (auto&& r : rects)
                        {
                            r.rect = pyr.rect_down(r.rect);
                        }
                    }
                }
                images.push_back(std::move(img));
                object_locations.push_back(std::move(rects));
            }
        }
    }

    // version with parts
    template <
        typename array_type
        >
    void load_image_dataset_subset (
        array_type& images,
        std::vector<std::vector<std::pair<dlib::mmod_rect, std::vector<dlib::point>>>> & object_locations,
        const image_dataset_file& source,
            unsigned long from_frame, unsigned long to_frame
    )
    {
        //images.clear();
        //object_locations.clear();

        using namespace dlib::image_dataset_metadata;
        dataset data;
        load_image_dataset_metadata(data, source.get_filename());

        // Set the current directory to be the one that contains the
        // metadata file. We do this because the file might contain
        // file paths which are relative to this folder.
        locally_change_current_dir chdir(get_parent_directory(file(source.get_filename())));

        typedef typename array_type::value_type image_type;
        image_type img;
        std::vector<std::pair<mmod_rect, std::vector<point>>> rects;
        //for (unsigned long i = 0; i < data.images.size(); ++i)
        for(unsigned long i = std::max(int(0),int(from_frame)); i < std::min(int(data.images.size()),int(to_frame)); ++i)
        {
            double min_rect_size = std::numeric_limits<double>::infinity();
            rects.clear();
            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                if (source.should_load_box(data.images[i].boxes[j]))
                {
                    if (data.images[i].boxes[j].ignore)
                    {
                        std::vector<point> emptyVec;
                        rects.push_back(std::make_pair(ignored_mmod_rect(data.images[i].boxes[j].rect), emptyVec));
                    }
                    else
                    {
                        std::map<std::string,point>& parts = data.images[i].boxes[j].parts;
                        std::map<std::string,point>::iterator it;

                        std::vector<point> points;

                        it = parts.find("0");
                          if (it != parts.end())
                              points.push_back(it->second);
                          else
                          {
                              std::cerr << "error part 0 not found, skipping this sample ..." << std::endl;
                              continue;
                          }
                        it = parts.find("1");
                           if (it != parts.end())
                                points.push_back(it->second);
                           else
                           {
                                std::cerr << "error part 1 not found, skipping this sample ..." << std::endl;
                                continue;
                           }
                         it = parts.find("2");
                              if (it != parts.end())
                                   points.push_back(it->second);
                              else
                              {
                                   std::cerr << "error part 2 not found, skipping this sample ..." << std::endl;
                                   continue;
                              }
                          it = parts.find("3");
                            if (it != parts.end())
                            {
                                points.push_back(it->second);
                            }
                            else
                               {
                                      std::cerr << "error part 3 not found, skipping this sample ..." << std::endl;
                                      continue;
                               }

                        rects.push_back(std::make_pair(mmod_rect(data.images[i].boxes[j].rect), points));
                        min_rect_size = std::min<double>(min_rect_size, rects.back().first.rect.area());
                    }
                    rects.back().first.label = data.images[i].boxes[j].label;
                }
            }

            if (!source.should_skip_empty_images() || impl::num_non_ignored_boxes(rects) != 0)
            {
                load_image(img, data.images[i].filename);
                if (rects.size() != 0)
                {
                    // if shrinking the image would still result in the smallest box being
                    // bigger than the box area threshold then shrink the image.
                    while(min_rect_size/2/2 > source.box_area_thresh())
                    {
                        pyramid_down<2> pyr;
                        pyr(img);
                        min_rect_size *= (1.0/2.0)*(1.0/2.0);
                        for (auto&& r : rects)
                        {
                            r.first.rect = pyr.rect_down(r.first.rect);
                            for(int i = 0; i < r.second.size(); i++)
                            {
                                r.second[i] = dlib::point(r.second[i].x()/2, r.second[i].y()/2);
                            }
                        }
                    }
                    while(min_rect_size*(2.0/3.0)*(2.0/3.0) > source.box_area_thresh())
                    {
                        pyramid_down<3> pyr;
                        pyr(img);
                        min_rect_size *= (2.0/3.0)*(2.0/3.0);
                        for (auto&& r : rects)
                            r.first.rect = pyr.rect_down(r.first.rect);
                        for (auto&& r : rects)
                        {
                            r.first.rect = pyr.rect_down(r.first.rect);
                            for(int i = 0; i < r.second.size(); i++)
                            {
                                r.second[i] = dlib::point(r.second[i].x()*2/3, r.second[i].y()*2/3);
                            }
                        }
                    }
                }
                images.push_back(std::move(img));
                object_locations.push_back(std::move(rects));
            }
        }
    }

// ----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------

    template <
        typename array_type
        >
    std::vector<std::vector<rectangle> > load_image_dataset_subset (
        array_type& images,
        std::vector<std::vector<full_object_detection> >& object_locations,
        const image_dataset_file& source,
        std::vector<std::string>& parts_list,
        int fromFrame,
        int toFrame
    )
    {
        std::vector<std::vector<rectangle> > ignored_rects;

        typedef typename array_type::value_type image_type;
        parts_list.clear();
        //images.clear();
        //object_locations.clear();

        using namespace dlib::image_dataset_metadata;
        dataset data;
        load_image_dataset_metadata(data, source.get_filename()); // @Todo can be moved out.

        if(int(fromFrame) == int(data.images.size()) || int(fromFrame) == int(toFrame))
        {
            return ignored_rects;
        }

        // Set the current directory to be the one that contains the
        // metadata file. We do this because the file might contain
        // file paths which are relative to this folder.
        locally_change_current_dir chdir(get_parent_directory(file(source.get_filename())));


        std::set<std::string> all_parts;

        // find out what parts are being used in the dataset.  Store results in all_parts.
        for (unsigned long i = std::max(int(0),int(fromFrame)); i < std::min(int(data.images.size()),int(toFrame)); ++i)
        {
            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                if (data.images[i].boxes[j].ignore || data.images[i].boxes[j].rect.width() < 12 || data.images[i].boxes[j].rect.height() < 12)
                {
                    continue;
                }
                else
                {
                    if (source.should_load_box(data.images[i].boxes[j]))
                    {
                        const std::map<std::string,point>& parts = data.images[i].boxes[j].parts;
                        std::map<std::string,point>::const_iterator itr;

                        for (itr = parts.begin(); itr != parts.end(); ++itr)
                        {
                            // what is this iterator first ?
                            //std::cout << "itr->first: " << itr->first << std::endl;
                            all_parts.insert(itr->first);
                        }
                    }
                }
            }
        }

        // make a mapping between part names and the integers [0, all_parts.size())
        std::map<std::string,int> parts_idx;

        // what is this iterator first ?
        //std::cout << "all_parts.size(): " << all_parts.size() << std::endl;
        for (std::set<std::string>::iterator i = all_parts.begin(); i != all_parts.end(); ++i)
        {
            parts_idx[*i] = parts_list.size();
            parts_list.push_back(*i);
        }
        //std::cout << "parts_list.size(): " << parts_list.size() << std::endl;
        //std::cout << "parts_idx.size(): " << parts_idx.size() << std::endl;

        std::vector<rectangle> ignored;
        image_type img;
        std::vector<full_object_detection> object_dets;
        for (unsigned long i = std::max(int(0),int(fromFrame)); i < std::min(int(data.images.size()),int(toFrame)); ++i)
        {
            double min_rect_size = std::numeric_limits<double>::infinity();
            object_dets.clear();
            ignored.clear();
            for (unsigned long j = 0; j < data.images[i].boxes.size(); ++j)
            {
                if (source.should_load_box(data.images[i].boxes[j]))
                {
                    if (data.images[i].boxes[j].ignore || data.images[i].boxes[j].rect.width() < 12 || data.images[i].boxes[j].rect.height() < 12)
                    {
                        ignored.push_back(data.images[i].boxes[j].rect);
                    }
                    else
                    {
                        // what is this iterator first ?
                        //std::cout << "parts_idx.size(): " << parts_idx.size() << std::endl;
                        std::vector<point> partlist(parts_idx.size(), OBJECT_PART_NOT_PRESENT);

                        // populate partlist with all the parts present in this box.
                        const std::map<std::string,point>& parts = data.images[i].boxes[j].parts;
                        std::map<std::string,point>::const_iterator itr;
                        for (itr = parts.begin(); itr != parts.end(); ++itr)
                        {
                            // what is this iterator first ?
                            //std::cout << "itr->first: " << itr->first << std::endl;
                            //std::cout << "parts_idx[itr->first]: " << parts_idx[itr->first] << std::endl;
                            partlist[parts_idx[itr->first]] = itr->second;
                        }

                        object_dets.push_back(full_object_detection(data.images[i].boxes[j].rect, partlist));
                        min_rect_size = std::min<double>(min_rect_size, object_dets.back().get_rect().area());
                    }
                }
            }

            if (!source.should_skip_empty_images() || object_dets.size() != 0)
            {
                load_image(img, data.images[i].filename);
                if (object_dets.size() != 0)
                {
                    // if shrinking the image would still result in the smallest box being
                    // bigger than the box area threshold then shrink the image.
                    while(min_rect_size/2/2 > source.box_area_thresh())
                    {
                        pyramid_down<2> pyr;
                        pyr(img);
                        min_rect_size *= (1.0/2.0)*(1.0/2.0);
                        for (auto&& r : object_dets)
                        {
                            r.get_rect() = pyr.rect_down(r.get_rect());
                            for (unsigned long k = 0; k < r.num_parts(); ++k)
                                r.part(k) = pyr.point_down(r.part(k));
                        }
                        for (auto&& r : ignored)
                        {
                            r = pyr.rect_down(r);
                        }
                    }
                    while(min_rect_size*(2.0/3.0)*(2.0/3.0) > source.box_area_thresh())
                    {
                        pyramid_down<3> pyr;
                        pyr(img);
                        min_rect_size *= (2.0/3.0)*(2.0/3.0);
                        for (auto&& r : object_dets)
                        {
                            r.get_rect() = pyr.rect_down(r.get_rect());
                            for (unsigned long k = 0; k < r.num_parts(); ++k)
                                r.part(k) = pyr.point_down(r.part(k));
                        }
                        for (auto&& r : ignored)
                        {
                            r = pyr.rect_down(r);
                        }
                    }
                }
                images.push_back(img);
                object_locations.push_back(object_dets);
                ignored_rects.push_back(ignored);
            }
        }


        return ignored_rects;
    }


// ----------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------

       /* template <
            typename array_type
            >
        std::vector<std::vector<rectangle> > load_image_dataset_subset (
            array_type& images,
            std::vector<std::vector<full_object_detection> >& object_locations,
            const image_dataset_file& imageDatasetFile,
            std::vector<std::string> & parts_list,
            int fromFrame,
            int toFrame
        )
        {
            //std::vector<std::string> parts_list;
            return load_image_dataset_subset(images, object_locations, imageDatasetFile, parts_list, fromFrame, toFrame);
        }*/


// ----------------------------------------------------------------------------------------

    class imageLoader
    {
    public:

        imageLoader(std::string training_filepath, std::string testing_filepath, int HOST_RAM_GB = 16, int IMAGE_SIZEOF_MB = 15, float train_testing_ratio = 0.75, float max_RAM_usage_percent = 0.2,
                   int nominal_width = 2592, int nominal_height = 2048, int resizeImageFactor = 2) : HOST_RAM_GB(HOST_RAM_GB), IMAGE_SIZEOF_MB(IMAGE_SIZEOF_MB),
            train_testing_ratio(train_testing_ratio), max_RAM_usage_percent(max_RAM_usage_percent), nominal_width(nominal_width), nominal_height(nominal_height), imageDatasetFileTraining(training_filepath),
            imageDatasetFileTesting(testing_filepath), resizeImageFactor(resizeImageFactor)
        {
            max_RAM_MB = (HOST_RAM_GB * 1024) * max_RAM_usage_percent;
            max_img_train_ramMB = max_RAM_MB  * train_testing_ratio;
            max_img_test_ramMB = max_RAM_MB * (1 - train_testing_ratio);

            max_img_RAM_read_batch = ((HOST_RAM_GB * 1024) / IMAGE_SIZEOF_MB) * max_RAM_usage_percent;
            max_img_RAM_batch_train = max_img_RAM_read_batch * train_testing_ratio;
            max_img_RAM_batch_test =  max_img_RAM_read_batch * (1 - train_testing_ratio);

            free_ramspace_img_MB = max_RAM_MB;

            imageLoadedCount_train = 0;
            imageLoadedCount_test = 0;
            iterationBatchLoading = 0;

            pointerImgTrainStart = 0;
            pointerImgTestStart = 0;
            trainingSetSize = image_dataset_get_dataset_size(imageDatasetFileTraining);
            testingSetSize = image_dataset_get_dataset_size(imageDatasetFileTesting);

            notEnoughRamForAllImages = max_img_RAM_read_batch < trainingSetSize+testingSetSize ? true : false;

            std::cout << "trainingSetSize: " << trainingSetSize << std::endl;
            std::cout << "testingSetSize: " << testingSetSize << std::endl;
            std::cout << "max_img_ramMB: "<< max_RAM_MB << std::endl;
            std::cout << "max_img_ram_batch: "<< max_img_RAM_read_batch << std::endl;
        }

        template <
            typename array_type
            >
        void operator() (
            std::vector<dlib::matrix<array_type>> & images_train,
            std::vector<dlib::matrix<array_type>> & images_test,
            std::vector<std::vector<mmod_rect>> & boxes_train,
            std::vector<std::vector<mmod_rect>> & boxes_test,
            std::vector<std::string> & parts_list_train,
            std::vector<std::string> & parts_list_test
        )
        {
            // manual cleaning now needed
            parts_list_train.clear();
            parts_list_test.clear();
            images_train.clear();
            images_test.clear();
            boxes_train.clear();
            boxes_test.clear();

            imageLoadedCount_train = 0;
            imageLoadedCount_test = 0;
            free_ramspace_img_MB = max_RAM_MB;
            iterationBatchLoading = 0;

            int tmpTrainPointer = pointerImgTrainStart;
            int tmpTestPointer = pointerImgTestStart;

            while((imageLoadedCount_train < trainingSetSize || imageLoadedCount_test < testingSetSize) &&
                  (free_ramspace_img_MB > 0 && ((free_ramspace_img_MB * train_testing_ratio) / IMAGE_SIZEOF_MB) >= 1))
            {
                std::cout << "iterationBatchLoading: " << iterationBatchLoading << std::endl;
                int trainToLoadCount = (free_ramspace_img_MB * train_testing_ratio) / IMAGE_SIZEOF_MB ;
                int testToLoadCount = (free_ramspace_img_MB * (1 - train_testing_ratio)) / IMAGE_SIZEOF_MB;

                std::cout << "train to load count: " << trainToLoadCount << std::endl;
                std::cout << "test to load count: " << testToLoadCount << std::endl;
                //std::cout << "sizeof ..:arrrayType:.. " <<  std::to_string(sizeof(array_type)) << std::endl;
                std::cout << "loading train from: " << tmpTrainPointer << ", to: " << tmpTrainPointer + trainToLoadCount << std::endl;
                std::cout << "loading test from: " << tmpTestPointer << ", to: " << tmpTestPointer + testToLoadCount << std::endl;
                // it is ok when tmpTrainPointer + trainToLoadCount exceeds the size trainingSetSize, load_image_dataset_subset will not load more than max size
                load_image_dataset_subset(images_train, boxes_train, imageDatasetFileTraining, tmpTrainPointer, tmpTrainPointer + trainToLoadCount);
                load_image_dataset_subset(images_test,  boxes_test,  imageDatasetFileTesting, tmpTestPointer , tmpTestPointer + testToLoadCount);

                //resize down, it will free some memory
                unsigned long loadedSizeCurrentBatch = 0;
                for(int i = imageLoadedCount_train; i < images_train.size();i++)
                {
                    if(images_train[i].nc() >= nominal_width || images_train[i].nr() >= nominal_height)
                    {
                        resize_image(1.0/float(resizeImageFactor), images_train[i]); // it makes swap, so RAM will go down from now
                        for(auto && b : boxes_train[i])
                        {
                            // re-create rect
                            b.rect.set_left(b.rect.left() / resizeImageFactor);
                            b.rect.set_top(b.rect.top() / resizeImageFactor);
                            b.rect.set_right(b.rect.right() / resizeImageFactor);
                            b.rect.set_bottom(b.rect.bottom() / resizeImageFactor);
                        }
                    }
                    loadedSizeCurrentBatch += (unsigned long)(images_train[i].nc()) * (unsigned long)images_train[i].nr() * (unsigned long)(sizeof( array_type));

                }
                for(int i = imageLoadedCount_test; i < images_test.size();i++)
                {
                    if(images_test[i].nc() >= nominal_width || images_test[i].nr() >= nominal_height)
                    {
                        resize_image(1.0/float(resizeImageFactor), images_test[i]);
                        for(auto && b : boxes_test[i])
                        {
                            // re-create rect
                            b.rect.set_left(b.rect.left() / resizeImageFactor);
                            b.rect.set_top(b.rect.top() / resizeImageFactor);
                            b.rect.set_right(b.rect.right() / resizeImageFactor);
                            b.rect.set_bottom(b.rect.bottom() / resizeImageFactor);
                        }
                    }

                    loadedSizeCurrentBatch += (unsigned long)images_test[i].nc() * (unsigned long)images_test[i].nr() * (unsigned long)(sizeof( array_type ));
                }

                imageLoadedCount_train = images_train.size();
                imageLoadedCount_test = images_test.size();

                free_ramspace_img_MB = free_ramspace_img_MB - (double(double(loadedSizeCurrentBatch)/1024.0))/1024.0;

                std::cout << "imageLoadedCount_train: " << imageLoadedCount_train << std::endl;
                std::cout << "imageLoadedCount_test: " << imageLoadedCount_test << std::endl;
                std::cout << "loaded batch size in MB: " << (double(double(loadedSizeCurrentBatch)/1024.0))/1024.0 << std::endl;
                std::cout << "left free RAM space in MB from declared space: " << free_ramspace_img_MB << std::endl;
                std::cout << "---- end step " << std::endl;

                iterationBatchLoading++;

                tmpTrainPointer = (pointerImgTrainStart + imageLoadedCount_train) % trainingSetSize;
                tmpTestPointer = (pointerImgTestStart + imageLoadedCount_test) % testingSetSize;
            }

            // check once again, does all images loaded into RAM mem
            if(imageLoadedCount_train == trainingSetSize && imageLoadedCount_test == testingSetSize)
            {
                notEnoughRamForAllImages = false;
            }
            else
                notEnoughRamForAllImages = true;

            pointerImgTrainStart = tmpTrainPointer;
            pointerImgTestStart = tmpTestPointer;

            // |--------|---|
            // |--|---------|

        }



        template <
            typename array_type
            >
        void operator() (
            std::vector<dlib::matrix<array_type>> & images_train,
            std::vector<dlib::matrix<array_type>> & images_test,
            std::vector<std::vector<std::pair<mmod_rect, std::vector<dlib::point>>>> & boxesPartsTrain,
            std::vector<std::vector<std::pair<mmod_rect, std::vector<dlib::point>>>> & boxesPartsTest,
            std::vector<std::string> & parts_list_train,
            std::vector<std::string> & parts_list_test
        )
        {
            // manual cleaning now needed
            parts_list_train.clear();
            parts_list_test.clear();
            images_train.clear();
            images_test.clear();
            boxesPartsTrain.clear();
            boxesPartsTest.clear();

            imageLoadedCount_train = 0;
            imageLoadedCount_test = 0;
            free_ramspace_img_MB = max_RAM_MB;
            iterationBatchLoading = 0;

            int tmpTrainPointer = pointerImgTrainStart;
            int tmpTestPointer = pointerImgTestStart;

            while((imageLoadedCount_train < trainingSetSize || imageLoadedCount_test < testingSetSize) &&
                  (free_ramspace_img_MB > 0 && ((free_ramspace_img_MB * train_testing_ratio) / IMAGE_SIZEOF_MB) >= 1))
            {
                std::cout << "iterationBatchLoading: " << iterationBatchLoading << std::endl;
                int trainToLoadCount = (free_ramspace_img_MB * train_testing_ratio) / IMAGE_SIZEOF_MB ;
                int testToLoadCount = (free_ramspace_img_MB * (1 - train_testing_ratio)) / IMAGE_SIZEOF_MB;

                std::cout << "train to load count: " << trainToLoadCount << std::endl;
                std::cout << "test to load count: " << testToLoadCount << std::endl;

                std::cout << "loading train from: " << tmpTrainPointer << ", to: " << tmpTrainPointer + trainToLoadCount << std::endl;
                std::cout << "loading test from: " << tmpTestPointer << ", to: " << tmpTestPointer + trainToLoadCount << std::endl;
                // it is ok when tmpTrainPointer + trainToLoadCount exceeds the size trainingSetSize, load_image_dataset_subset will not load more than max size
                load_image_dataset_subset(images_train, boxesPartsTrain, imageDatasetFileTraining, tmpTrainPointer, tmpTrainPointer + trainToLoadCount);
                load_image_dataset_subset(images_test,  boxesPartsTest,  imageDatasetFileTesting, tmpTestPointer , tmpTestPointer + testToLoadCount);

                //resize down, it will free some memory
                unsigned long loadedSizeCurrentBatch = 0;
                for(int i = imageLoadedCount_train; i < images_train.size();i++)
                {
                    if(images_train[i].nc() >= nominal_width || images_train[i].nr() >= nominal_height)
                    {
                        resize_image(1.0/float(resizeImageFactor), images_train[i]); // it makes swap, so RAM will go down from now
                        for(auto && b : boxesPartsTrain[i])
                        {
                            // re-create rect
                            b.first.rect.set_left(b.first.rect.left() / resizeImageFactor);
                            b.first.rect.set_top(b.first.rect.top() / resizeImageFactor);
                            b.first.rect.set_right(b.first.rect.right() / resizeImageFactor);
                            b.first.rect.set_bottom(b.first.rect.bottom() / resizeImageFactor);
                            for(int i = 0; i < b.second.size(); i++)
                            {
                                b.second[i] = dlib::point(b.second[i].x()/resizeImageFactor, b.second[i].y()/resizeImageFactor);
                            }
                        }
                    }

                    loadedSizeCurrentBatch += (unsigned long)(images_train[i].nc()) * (unsigned long)images_train[i].nr() * (unsigned long)(sizeof( array_type));

                }
                for(int i = imageLoadedCount_test; i < images_test.size();i++)
                {
                    if(images_test[i].nc() >= nominal_width || images_test[i].nr() >= nominal_height)
                    {
                        resize_image(1.0/float(resizeImageFactor), images_test[i]);
                        for(auto && b : boxesPartsTest[i])
                        {
                            // re-create rect
                            b.first.rect.set_left(b.first.rect.left() / resizeImageFactor);
                            b.first.rect.set_top(b.first.rect.top() / resizeImageFactor);
                            b.first.rect.set_right(b.first.rect.right() / resizeImageFactor);
                            b.first.rect.set_bottom(b.first.rect.bottom() / resizeImageFactor);
                            for(int i = 0; i < b.second.size(); i++)
                            {
                                b.second[i] = dlib::point(b.second[i].x()/resizeImageFactor, b.second[i].y()/resizeImageFactor);
                            }
                        }
                    }

                    loadedSizeCurrentBatch += (unsigned long)images_test[i].nc() * (unsigned long)images_test[i].nr() * (unsigned long)(sizeof( array_type ));
                }

                imageLoadedCount_train = images_train.size();
                imageLoadedCount_test = images_test.size();

                free_ramspace_img_MB = free_ramspace_img_MB - (double(double(loadedSizeCurrentBatch)/1024.0))/1024.0;

                std::cout << "imageLoadedCount_train: " << imageLoadedCount_train << std::endl;
                std::cout << "imageLoadedCount_test: " << imageLoadedCount_test << std::endl;
                std::cout << "loaded batch size in MB: " << (double(double(loadedSizeCurrentBatch)/1024.0))/1024.0 << std::endl;
                std::cout << "left free RAM space in MB from declared space: " << free_ramspace_img_MB << std::endl;
                std::cout << "---- end step " << std::endl;

                iterationBatchLoading++;

                tmpTrainPointer = (pointerImgTrainStart + imageLoadedCount_train) % trainingSetSize;
                tmpTestPointer = (pointerImgTestStart + imageLoadedCount_test) % testingSetSize;
            }

            // check once again, does all images loaded into RAM mem
            if(imageLoadedCount_train == trainingSetSize && imageLoadedCount_test == testingSetSize)
            {
                notEnoughRamForAllImages = false;
            }
            else
                notEnoughRamForAllImages = true;

            pointerImgTrainStart = tmpTrainPointer;
            pointerImgTestStart = tmpTestPointer;

            // |--------|---|
            // |--|---------|

        }


        bool checkRamIsNotEnough(){
            return notEnoughRamForAllImages;
        }


    private:

    int resizeImageFactor;

    int HOST_RAM_GB;
    int IMAGE_SIZEOF_MB;
    float train_testing_ratio;
    float max_RAM_usage_percent;

    unsigned long max_RAM_MB;
    unsigned long max_img_train_ramMB;
    unsigned long max_img_test_ramMB;

    unsigned long max_img_RAM_read_batch;
    unsigned long max_img_RAM_batch_train;
    unsigned long max_img_RAM_batch_test;

    image_dataset_file imageDatasetFileTraining;
    image_dataset_file imageDatasetFileTesting;

    int trainingSetSize;
    int testingSetSize;

    float free_ramspace_img_MB;
    int imageLoadedCount_train,  imageLoadedCount_test;

    int iterationBatchLoading;

    const int nominal_width;
    const int nominal_height;

    int pointerImgTrainStart;
    int pointerImgTestStart;

    bool notEnoughRamForAllImages;

    };


}

// ----------------------------------------------------------------------------------------
#endif // DLIB_LOAD_IMAGE_DaTASET_Hh_
