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
        if (argc != 3)
        {
            cout << "Give the path to the examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./train_shape_predictor_ex training.xml testing.xml" << endl;
            cout << endl;
            return 0;
        }
        const std::string training_filepath = argv[1];
        const std::string testing_filepath = argv[2];
        // The faces directory contains a training dataset and a separate
        // testing dataset.  The training data consists of 4 images, each
        // annotated with rectangles that bound each human face along with 68
        // face landmarks on each face.  The idea is to use this training data
        // to learn to identify the position of landmarks on human faces in new
        // images. 
        // 
        // Once you have trained a shape_predictor it is always important to
        // test it on data it wasn't trained on.  Therefore, we will also load
        // a separate testing set of 5 images.  Once we have a shape_predictor 
        // created from the training data we will see how well it works by
        // running it on the testing images. 
        // 
        // So here we create the variables that will hold our dataset.
        // images_train will hold the 4 training images and faces_train holds
        // the locations and poses of each face in the training images.  So for
        // example, the image images_train[0] has the faces given by the
        // full_object_detections in faces_train[0].
        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;
        std::vector<std::string> parts_list_train;
        std::vector<std::string> parts_list_test;

        // Now we load the data.  These XML files list the images in each
        // dataset and also contain the positions of the face boxes and
        // landmarks (called parts in the XML file).  Obviously you can use any
        // kind of input format you like so long as you store the data into
        // images_train and faces_train.  But for convenience dlib comes with
        // tools for creating and loading XML image dataset files.  Here you see
        // how to load the data.  To create the XML files you can use the imglab
        // tool which can be found in the tools/imglab folder.  It is a simple
        // graphical tool for labeling objects in images.  To see how to use it
        // read the tools/imglab/README.txt file.


        int resizeImageFactor = 2;

        const int max_ramGB = 16;
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

        //dlib::array<array2d<unsigned char> > images_train_ram_batch, images_test_ram_batch;
        //std::vector<std::vector<full_object_detection> > faces_train_ram_batch, faces_test_ram_batch;

        free_ramspace_img_MB = max_img_ramMB;

        int iterationBatchLoading = 0;

        // manual cleaning now needed
        parts_list_train.clear();
        parts_list_test.clear();
        images_train.clear();
        faces_train.clear();

        const int nominal_width = 2592;
        const int nominal_height = 2048;

        while((imageLoadedCount_train < trainingSetSize || imageLoadedCount_test < testingSetSize) &&
              (free_ramspace_img_MB > 0 && ((free_ramspace_img_MB * train_testing_ratio) / img_sizeofMB) >= 1))
        {

        std::cout << "iterationBatchLoading: " << iterationBatchLoading << std::endl;

        load_image_dataset_subset(images_train, faces_train, imageDatasetFileTraining, parts_list_train, imageLoadedCount_train, imageLoadedCount_train + (free_ramspace_img_MB * train_testing_ratio) / img_sizeofMB); // ram inefficient
        load_image_dataset_subset(images_test,  faces_test,  imageDatasetFileTesting, parts_list_test, imageLoadedCount_test, imageLoadedCount_test + (free_ramspace_img_MB * (1 - train_testing_ratio)) / img_sizeofMB);

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


        // Now make the object responsible for training the model.  
        shape_predictor_trainer trainer;
        // This algorithm has a bunch of parameters you can mess with.  The
        // documentation for the shape_predictor_trainer explains all of them.
        // You should also read Kazemi's paper which explains all the parameters
        // in great detail.  However, here I'm just setting three of them
        // differently than their default values.  I'm doing this because we
        // have a very small dataset.  In particular, setting the oversampling
        // to a high amount (300) effectively boosts the training set size, so
        // that helps this example.
        trainer.set_oversampling_amount(100);
        // I'm also reducing the capacity of the model by explicitly increasing
        // the regularization (making nu smaller) and by using trees with
        // smaller depths.  
        trainer.set_nu(0.35);
        trainer.set_tree_depth(6);

        //trainer.set_nu(0.05);
        //trainer.set_tree_depth(2);

        // some parts of training process can be parallelized.
        // Trainer will use this count of threads when possible
        trainer.set_num_threads(2);

        // Tell the trainer to print status messages to the console so we can
        // see how long the training will take.
        trainer.be_verbose();

        // Now finally generate the shape model
        shape_predictor sp = trainer.train(images_train, faces_train);

        // Now that we have a model we can test it.  This function measures the
        // average distance between a face landmark output by the
        // shape_predictor and where it should be according to the truth data.
        // Note that there is an optional 4th argument that lets us rescale the
        // distances.  Here we are causing the output to scale each face's
        // distances by the interocular distance, as is customary when
        // evaluating face landmarking systems.
        cout << "mean training error: "<< 
            test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

        // The real test is to see how well it does on data it wasn't trained
        // on.  We trained it on a very small dataset so the accuracy is not
        // extremely high, but it's still doing quite good.  Moreover, if you
        // train it on one of the large face landmarking datasets you will
        // obtain state-of-the-art results, as shown in the Kazemi paper.
        cout << "mean testing error:  "<< 
            test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

        // Finally, we save the model to disk so we can use it later.
        serialize("sp-gucioZDMpack2-filtered-v3.dat") << sp;
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
    for (unsigned long i = 36; i <= 41; ++i) 
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the right eye by averaging the points around 
    // the eye.
    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
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

