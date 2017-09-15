/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <stdio.h>
#include <gtest/gtest.h>
#include <dmlc/concurrency.h>
#include <mxnet/base.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#else
#include <winsock.h>
#endif
#include <fstream>
#include <thread>
#include "../../include/mxnet/c_predict_api.h"
#include "../include/test_data.h"

namespace mxnet {
namespace test {

class DirUtil {
  static std::string get_dir(const std::string path) {
    const std::string::size_type found = path.find_last_of("/\\");
    if (found == std::string::npos) {
      return std::string();
    }
    return std::string(path.c_str(), path.c_str() + found);
  }

  static bool does_dir_exist(const std::string path) {
    struct stat statbuf;
    if (!stat(path.c_str(), &statbuf)) {
      if (S_ISDIR(statbuf.st_mode)) {
        return true;
      }
      LOG(WARNING) << "Path " << path << " exists, but is not a directory";
    }
    return false;
  }

  static bool create_dir(const std::string& path) {
    if(path.empty()) {
      return true;
    }
    if(!create_dir_for_path(path)) {
      return false;
    }
    if(does_dir_exist(path)) {
      return true;
    }
    return mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0;
  }
 public:
  static bool create_dir_for_path(const std::string& path) {
    return create_dir(get_dir(path));
  }
};

static bool download_file(const std::string& url,
                          const std::string output,
                          bool overwrite = false) {
  sockaddr_in servaddr;
  hostent *hp;
  int sock_id;
  //std::string request = "GET /index.html HTTP/1.0\nFrom: mxnet\nUser-Agent: mxnet\n\n";
  std::string request = "GET ";
  request += url;
  request += " HTTP/1.0\nFrom: mxnet\nUser-Agent: mxnet\n\n";

  if(overwrite) {
    unlink(output.c_str());
  } else {
    struct stat statbuf;
    if(!stat(output.c_str(), &statbuf)) {
      if(!S_ISDIR(statbuf.st_mode)) {
        return true;
      }
      LOG(WARNING) << "Output file " << output << " is a directory";
      return false;
    }
  }

  if(!DirUtil::create_dir_for_path(output)) {
    LOG(WARNING) << "Unable to create directory for path: " << output;
    return false;
  }

  if((sock_id = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    LOG(WARNING) << "Could not get a socket";
    return false;
  }

  memset(&servaddr,0,sizeof(servaddr));

  const char *s_start = strchr(url.c_str(), ':');
  if(!s_start || *++s_start != '/' || *++s_start != '/') {
    LOG(WARNING) << "Malformed url: " << url;
    return false;
  }
  const char *s_end = strchr(++s_start, '/');
  if(!s_end)
    s_end = s_start + strlen(s_start);
  std::string address(s_start, s_end);

  if((hp = gethostbyname(address.c_str())) == NULL) {
    LOG(WARNING) << "Could not get address for " << url;
    return false;
  }

  memcpy((char *)&servaddr.sin_addr.s_addr, (char *)hp->h_addr, hp->h_length);

  //fill int port number and type
  servaddr.sin_port = htons(80);
  servaddr.sin_family = AF_INET;

  //make the connection
  if(connect(sock_id, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    LOG(WARNING) << "Could not connect to " << url;
    return false;
  }

  // change to send()
  ::write(sock_id, request.c_str(), request.size());

  std::ofstream outfile(output, std::ios::binary|std::ios::out);

  std::cout << "Downloading: " << output << "..." << std::flush;

  constexpr size_t BUFFER_SIZE = 1024 * 1024;
  char message[BUFFER_SIZE];
  size_t total_size = 0;
  ssize_t size_this_pass = 0;
  // change to recv()
  while((size_this_pass = read(sock_id, message, BUFFER_SIZE)) != 0) {
    outfile.write(message, size_this_pass);
    total_size += size_this_pass;
    std::cout << "." << std::flush;
  }

  if(errno) {
    LOG(WARNING) << "Error: " << strerror(errno);
    unlink(output.c_str()); // delete bad file
  }

  std::cout << "Download complete: " << total_size << std::endl << std::flush;

  return errno == 0;
}

/*! \brief Weak semaphore, can cause thread starvation.  Not to be used to production. */
class simple_weak_semaphore {
 public:
  simple_weak_semaphore(int count = 0)
    : count_{count} {}

  void post() {
    std::unique_lock<std::mutex> lck(mtx_);
    ++count_;
    cv_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lck(mtx_);
    while (!count_) {
      cv_.wait(lck);
    }
    --count_;
  }

 private:

  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<int> count_;
};

const mx_float DEFAULT_MEAN = 117.0;

using namespace std::chrono;

struct PredictorFree {
  void operator ()(PredictorHandle *p) {
    if(p) {
      MXPredFree(p);
    }
  }
};

typedef std::vector<std::unique_ptr<PredictorHandle, PredictorFree>> PredictorVector;

struct JobSynchronizer {
  PredictorVector       predictors_;
  cv::Mat               image_as_cv_;
  std::vector<mx_float> image_as_floats_;
  simple_weak_semaphore available_;
  simple_weak_semaphore done_;
};

/*! \brief Class to read file to buffer */
class BufferFile {
 public :
  const std::string file_path_;
  int length_;
  char *buffer_;

  explicit BufferFile(const std::string& file_path)
    : file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      length_ = 0;
      buffer_ = NULL;
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_ = new char[sizeof(char) * length_];
    ifs.read(buffer_, length_);
    ifs.close();
  }

  int GetLength() const {
    return length_;
  }

  const char *GetBuffer() const {
    return buffer_;
  }

  ~BufferFile() {
    if (buffer_) {
      delete[] buffer_;
      buffer_ = NULL;
    }
  }
};

static void GetImageFile(const cv::Mat &im_ori,
                         mx_float *image_data,
                         const int channels,
                         const cv::Size resize_size,
                         const mx_float *mean_data = nullptr) {
  cv::Mat im;

  resize(im_ori, im, resize_size);

  int size = im.rows * im.cols * channels;

  mx_float *ptr_image_r = image_data;
  mx_float *ptr_image_g = image_data + size / 3;
  mx_float *ptr_image_b = image_data + size / 3 * 2;

  float mean_b, mean_g, mean_r;
  mean_b = mean_g = mean_r = DEFAULT_MEAN;

  for (int i = 0; i < im.rows; i++) {
    uchar *data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
      if (mean_data) {
        mean_r = *mean_data;
        if (channels > 1) {
          mean_g = *(mean_data + size / 3);
          mean_b = *(mean_data + size / 3 * 2);
        }
        mean_data++;
      }
      if (channels > 1) {
        *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
        *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
      }

      *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
    }
  }
}

//static void GetImageFile(const std::string image_file,
//                         mx_float *image_data, const int channels,
//                         const cv::Size resize_size, const mx_float *mean_data = nullptr) {
//  // Read all kinds of file into a BGR color 3 channels image
//  cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);
//
//  GetImageFile(im_ori, image_data, channels, resize_size, mean_data);
//}

//static std::vector<mx_float> image_as_floats_;

//static std::vector<std::unique_ptr<PredictorHandle, PredictorFree>> predictors;

static std::atomic<bool> end_test(false);

#define NUM_INFERENCES_PER_THREAD 10

static void thread_inference_from_array(const int index,
                                       std::shared_ptr<JobSynchronizer> jobSynchronizer) {
  while (!end_test) {
    jobSynchronizer->available_.wait();

    high_resolution_clock::time_point time_start = high_resolution_clock::now();

    // Set Input Image
    MXPredSetInput(jobSynchronizer->predictors_[index].get(),
                   "data",
                   jobSynchronizer->image_as_floats_.data(),
                   jobSynchronizer->image_as_floats_.size());

    // Do Predict Forward
    MXPredForward(jobSynchronizer->predictors_[index].get());

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(jobSynchronizer->predictors_[index].get(),
                         output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) {
      size *= shape[i];
    }

    std::vector<float> data(size);

    MXPredGetOutput(jobSynchronizer->predictors_[index].get(), output_index, &(data[0]), size);

    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);

//    std::cout << "Thread " << index << ", inference: " << i << " took "
//              << time_span.count() << " sec" << std::endl;
    jobSynchronizer->done_.post();
  }
}

static std::shared_ptr<std::vector<uint8_t>> create_inception_v3_bn_params() {
  std::shared_ptr<std::vector<uint8_t>> p(new std::vector<uint8_t>());
  return p;
}

static void thread_inference_from_mat(int index, std::shared_ptr<JobSynchronizer> jobSynchronizer) {
  for (int i=0; i<NUM_INFERENCES_PER_THREAD; i++) {

    const auto image_size = static_cast<size_t>(jobSynchronizer->image_as_cv_.cols
                                                * jobSynchronizer->image_as_cv_.rows
                                                * jobSynchronizer->image_as_cv_.channels());

    const mx_float* nd_data = NULL;
    std::vector<mx_float> image_as_floats(image_size);
    GetImageFile(jobSynchronizer->image_as_cv_,
                 image_as_floats.data(),
                 jobSynchronizer->image_as_cv_.channels(),
                 cv::Size(jobSynchronizer->image_as_cv_.cols,
                          jobSynchronizer->image_as_cv_.rows), nd_data);

    high_resolution_clock::time_point time_start = high_resolution_clock::now();

    // Set Input Image
    MXPredSetInput(jobSynchronizer->predictors_[index].get(), "data",
                   image_as_floats.data(), image_as_floats.size());

    // Do Predict Forward
    MXPredForward(jobSynchronizer->predictors_[index].get());

    mx_uint output_index = 0;

    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(jobSynchronizer->predictors_[index].get(),
                         output_index, &shape, &shape_len);

    size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

    std::vector<float> data(size);

    MXPredGetOutput(jobSynchronizer->predictors_[index].get(), output_index, &(data[0]), size);

    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);

    std::cout << "Thread " << index << ", inference: " << i << " took "
              << time_span.count() << " sec" << std::endl;
  }
}

static void initialize(std::shared_ptr<JobSynchronizer> jobSynchronizer,
                       int dev_type,
                       int num_threads,
                       unsigned int batch_size,
                       unsigned int num_non_zero,
                       const std::string& json_model,
                       const uint8_t *param_data,
                       const size_t param_data_length,
                       const uint8_t *image_data,
                       const size_t image_data_length) {
  //BufferFile json_data(json_file);

  int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  cv::Mat rawData(1, static_cast<int>(image_data_length),
                  CV_8UC3, const_cast<uchar *>(image_data));
  jobSynchronizer->image_as_cv_ = cv::imdecode(rawData, cv::IMREAD_COLOR);
  GTEST_ASSERT_NE(jobSynchronizer->image_as_cv_.data, nullptr);

  // Image size and channels
  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = {
    batch_size,
    static_cast<mx_uint>(jobSynchronizer->image_as_cv_.channels()),
    static_cast<mx_uint>(jobSynchronizer->image_as_cv_.cols),
    static_cast<mx_uint>(jobSynchronizer->image_as_cv_.rows) };

  CHECK(!json_model.empty());

  jobSynchronizer->predictors_.resize(num_threads);
  for(int i = 0; i < num_threads; ++i) {
    jobSynchronizer->predictors_[i].reset(new PredictorHandle);
  }

  const size_t image_size = static_cast<size_t>(jobSynchronizer->image_as_cv_.cols
                                                * jobSynchronizer->image_as_cv_.rows
                                                * jobSynchronizer->image_as_cv_.channels());
  //const mx_float* nd_data = NULL;
  jobSynchronizer->image_as_floats_.resize(image_size * batch_size);

  for(size_t i = 0; i < batch_size; ++i) {
    if(i < num_non_zero) {
      std::copy(image_data,
                image_data + image_data_length,
                jobSynchronizer->image_as_floats_.data() + (image_size * i));
//      GetImageFile(image_file, image_as_floats_.data() + (image_size*i),
//                   channels, cv::Size(width, height), nd_data);
    } else {
      CHECK(false);
      std::fill(jobSynchronizer->image_as_floats_.begin()
                + (image_size * i), jobSynchronizer->image_as_floats_.end(), 0);
      break;
    }
  }

  // Create Predictors
  for(int i = 0; i < num_threads; ++i) {
    MXPredCreate(json_model.c_str(),
                 param_data,
                 param_data_length,
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 jobSynchronizer->predictors_[i].get());
  }

//  image_as_cv = cv::imread(image_file, cv::IMREAD_COLOR);
}

static void run_test(const int device_type,
                     const int num_threads,
                     const int batch_size,
                     const int num_non_zero,
                     const char *model_json,
                     const uint8_t *param_data,
                     const size_t param_data_length,
                     const uint8_t *image_data,
                     const size_t image_data_length,
                     const bool should_resize) {
  if(should_resize) {
    std::cout << "Will include time to resize image" << std::endl;
  } else {
    std::cout << "Will not include time to resize image" << std::endl;
  }

  // Models path for your model, you have to modify it
  //std::string json_file = std::string(argv[7]);
  //std::string param_file = std::string(argv[8]);

  std::shared_ptr<JobSynchronizer> jobSynchronizer(new JobSynchronizer());

  initialize(jobSynchronizer,
             device_type, num_threads, batch_size, num_non_zero,
             model_json,
             param_data, param_data_length,
             image_data, image_data_length);

  std::vector<std::unique_ptr<std::thread>> threads;
  threads.reserve(num_threads);
  for(int i=0; i<num_threads; i++) {
    if(should_resize) {
      threads.emplace_back(std::unique_ptr<std::thread>(
        new std::thread(thread_inference_from_mat, i, jobSynchronizer)));
    } else {
      threads.emplace_back(std::unique_ptr<std::thread>(
        new std::thread(thread_inference_from_array, i, jobSynchronizer)));
    }
  }

  do {
    high_resolution_clock::time_point time_start = high_resolution_clock::now();
    for(int i=0; i<num_threads; i++) {
      jobSynchronizer->available_.post();
    }
    for(int i=0; i<num_threads; i++) {
      jobSynchronizer->done_.wait();
    }
    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);

    std::cout << "All threads took " << time_span.count() << " sec" << std::endl;
  } while(!end_test);

  std::cout << "Done" << std::endl;

  threads.clear();
}

}  // namespace test
}  // namespace mxnet

static const char *inception_v3_bn_local_param_file = "data/inception_v3_bn_params.param";
static const char *model_resnet162_local_param_file = "data/model_resnet162_params.param";

TEST(Engine, MultiThreadedInference) {

  bool downloaded_ok = mxnet::test::download_file(
    mxnet::test::model_inception_v3_bn_params_url, inception_v3_bn_local_param_file, true);
  GTEST_ASSERT_EQ(downloaded_ok, false);

  downloaded_ok = mxnet::test::download_file(
    mxnet::test::model_resnet162_params_url, model_resnet162_local_param_file, true);
  GTEST_ASSERT_EQ(downloaded_ok, true);

  std::shared_ptr<std::vector<uint8_t>> params = mxnet::test::create_inception_v3_bn_params();

  //mxnet::test::BufferFile param_buffer(inception_v3_bn_local_param_file);

  mxnet::test::BufferFile param_buffer(model_resnet162_local_param_file);

  mxnet::test::run_test(mshadow::cpu::kDevCPU, 1, 10, 10,
                        //mxnet::test::model_inception_v3_bn,
                        mxnet::test::model_resnet162,
                        //params->data(), params->size(),  // parameter file
                        (const uint8_t *)param_buffer.GetBuffer(), param_buffer.GetLength(),
                        mxnet::test::test_image_cat_jpg, mxnet::test::test_image_cat_length,
                        false);
}
