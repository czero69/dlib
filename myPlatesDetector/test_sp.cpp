// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to use dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014

    In particular, we will train a face landmarking model based on a small dataset
    and then evaluate it.  If you want to visualize the output of the trained
    model on some images then you can run the face_landmark_detection_ex.cpp
    example program with sp.dat as the input model.

    It should also be noted that this kind of model, while often used for face
    landmarking, is quite general and can be used for a variety of shape
    prediction tasks.  But here we demonstrate it only on a simple face
    landmarking task.
*/


#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>
#include "myUtils.h"

#include <dlib/gui_widgets.h> // just for testing

#include <chrono>  // for high_resolution_clock

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);
/*!
    ensures
        - returns an object D such that:
            - D[i][j] == the distance, in pixels, between the eyes for the face represented
              by objects[i][j].
!*/

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        // In this example we are going to train a shape_predictor based on the
        // small faces dataset in the examples/faces directory.  So the first
        // thing we do is load that dataset.  This means you need to supply the
        // path to this faces folder as a command line argument so we will know
        // where it is.
        if (argc != 4)
        {
            cout << "   ./test_sp detector training.xml testing.xml" << endl;
            cout << endl;
            return 0;
        }

        const std::string detector_path = argv[1];
        const std::string training_filepath = argv[2];
        const std::string testing_filepath = argv[3];

        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;
        std::vector<std::string> parts_list_train;
        std::vector<std::string> parts_list_test;

        int resizeImageFactor = 2;

        const int max_ramGB = 32;
        const int img_sizeofMB = 5;
        const float train_testing_ratio = 0.75;
        const float max_ram_capacity_percentage = 0.5;

        unsigned long max_img_ramMB = (max_ramGB * 1024) * max_ram_capacity_percentage;
        unsigned long max_img_train_ramMB = max_img_ramMB  * train_testing_ratio;
        unsigned long max_img_test_ramMB = max_img_ramMB * (1 - train_testing_ratio);

        unsigned long max_img_ram_batch = ((max_ramGB * 1024) / img_sizeofMB) * max_ram_capacity_percentage;
        unsigned long max_img_ram_batch_train = max_img_ram_batch * train_testing_ratio;
        unsigned long max_img_ram_batch_test =  max_img_ram_batch * (1 - train_testing_ratio);

        image_dataset_file imageDatasetFileTraining = image_dataset_file(training_filepath);
        image_dataset_file imageDatasetFileTesting = image_dataset_file(testing_filepath);

        int trainingSetSize = image_dataset_get_dataset_size(imageDatasetFileTraining);
        int testingSetSize = image_dataset_get_dataset_size(imageDatasetFileTesting);

        std::cout << "trainingSetSize: " << trainingSetSize << std::endl;
        std::cout << "testingSetSize: " << testingSetSize << std::endl;
        std::cout << "max_img_ramMB: "<< max_img_ramMB << std::endl;
        std::cout << "max_img_ram_batch: "<< max_img_ram_batch << std::endl;

        // policy load max_ram_capacity_percentage of ram, downscale, calculate new max capacity, load again, dowscale again etc.

        float free_ramspace_img_MB;
        int imageLoadedCount_train = 0,  imageLoadedCount_test = 0;

        free_ramspace_img_MB = max_img_ramMB;

        int iterationBatchLoading = 0;

        // manual cleaning now needed
        parts_list_train.clear();
        parts_list_test.clear();
        images_train.clear();
        faces_train.clear();

        const int nominal_width = 2592;
        const int nominal_height = 2048;

        dlib::image_dataset_metadata::dataset dataTrain;
        dlib::image_dataset_metadata::dataset dataTest;

        load_image_dataset_metadata(dataTrain, imageDatasetFileTraining.get_filename());
        load_image_dataset_metadata(dataTest, imageDatasetFileTraining.get_filename());

        // remove * 100 here, only for fast test purposes
        while((imageLoadedCount_train * 1 < trainingSetSize || imageLoadedCount_test * 1 < testingSetSize) &&
              (free_ramspace_img_MB > 0 && ((free_ramspace_img_MB * train_testing_ratio) / img_sizeofMB) >= 1))
        {

        std::cout << "iterationBatchLoading: " << iterationBatchLoading << std::endl;

        load_image_dataset_subset(images_train, dataTrain,  faces_train, imageDatasetFileTraining, parts_list_train, imageLoadedCount_train, imageLoadedCount_train + (free_ramspace_img_MB * train_testing_ratio) / img_sizeofMB, 50); // ram inefficient
        load_image_dataset_subset(images_test, dataTest,  faces_test,  imageDatasetFileTesting, parts_list_test, imageLoadedCount_test, imageLoadedCount_test + (free_ramspace_img_MB * (1 - train_testing_ratio)) / img_sizeofMB, 50);

        std::cout << "imgtrain[0] nc, nr : " << images_train[0].nc() << ", " << images_train[0].nr() << std::endl;
        // downsample image that is larger than half of the nominal widt/height

        unsigned long loadedSizeCurrentBatch = 0;

        for(int i = imageLoadedCount_train; i < images_train.size();i++)
        {
            if(images_train[i].nc() >= nominal_width/2 || images_train[i].nr() >= nominal_height/2)
            {
                resize_image(1.0/float(resizeImageFactor), images_train[i]); // it makes swap, so RAM will go down from now
                for(auto && b : faces_train[i])
                {
                    // re-create rect
                    dlib::rectangle & oldRect =  b.get_rect();
                    dlib::rectangle newRect(oldRect.left() / resizeImageFactor, oldRect.top()/resizeImageFactor , oldRect.right()/resizeImageFactor, oldRect.bottom()/resizeImageFactor);
                    std::vector<dlib::point> newParts;
                    for(int i = 0; i < b.num_parts(); i++)
                        newParts.push_back(dlib::point(b.part(i).x()/resizeImageFactor,b.part(i).y()/resizeImageFactor));
                    b = dlib::full_object_detection(newRect, newParts);
                }
            }

            loadedSizeCurrentBatch += (unsigned long)(images_train[i].nc()) * (unsigned long)images_train[i].nr() * (unsigned long)(sizeof( unsigned char ));

        }
        for(int i = imageLoadedCount_test; i < images_test.size();i++)
        {
            if(images_test[i].nc() >= nominal_width/2 || images_test[i].nr() >= nominal_height/2)
            {
                resize_image(1.0/float(resizeImageFactor), images_test[i]);
                for(auto && b : faces_test[i])
                {
                    // re-create rect
                    dlib::rectangle & oldRect =  b.get_rect();
                    dlib::rectangle newRect(oldRect.left() / resizeImageFactor, oldRect.top()/resizeImageFactor , oldRect.right()/resizeImageFactor, oldRect.bottom()/resizeImageFactor);
                    std::vector<dlib::point> newParts;
                    for(int i = 0; i < b.num_parts(); i++)
                        newParts.push_back(dlib::point(b.part(i).x()/resizeImageFactor,b.part(i).y()/resizeImageFactor));
                    b = dlib::full_object_detection(newRect, newParts);
                }
            }

            loadedSizeCurrentBatch += (unsigned long)images_test[i].nc() * (unsigned long)images_test[i].nr() * (unsigned long)(sizeof( unsigned char ));
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

        }

        bool DEBUG_IMSHOW = 0;

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
        }


        shape_predictor sp;
        deserialize(detector_path) >> sp;

        int totalTrainBoxesCount = 0;
        for(auto && f : faces_train)
            totalTrainBoxesCount += f.size();

        auto startBatch = std::chrono::high_resolution_clock::now();

        cout << "mean training error: "<<
            test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

        auto finishBatch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedBatch = finishBatch - startBatch;
        std::cout << "Elapsed time per training sample: " << elapsedBatch.count()/(totalTrainBoxesCount) << " s\n";

        // The real test is to see how well it does on data it wasn't trained
        // on.  We trained it on a very small dataset so the accuracy is not
        // extremely high, but it's still doing quite good.  Moreover, if you
        // train it on one of the large face landmarking datasets you will
        // obtain state-of-the-art results, as shown in the Kazemi paper.
        cout << "mean testing error:  "<<
            test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around
    // the eye.
    for (unsigned long i = 0; i < 6; ++i)
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the right eye by averaging the points around
    // the eye.
    cnt = 0;
    for (unsigned long i = 15; i >= 10; --i)
    {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

