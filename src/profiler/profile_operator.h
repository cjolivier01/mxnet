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

#ifndef MXNET_PROFILER_PROFILE_OPERATOR_H_
#define MXNET_PROFILER_PROFILE_OPERATOR_H_

#include <dmlc/logging.h>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include "./profiler.h"

namespace mxnet {
namespace profiler {

/*!
 *  _____              __  _  _        ____                         _
 * |  __ \            / _|(_)| |      / __ \                       | |
 * | |__) |_ __  ___ | |_  _ | | ___ | |  | |_ __   ___  _ __  __ _| |_  ___  _ __
 * |  ___/| '__|/ _ \|  _|| || |/ _ \| |  | | '_ \ / _ \| '__|/ _` | __|/ _ \| '__|
 * | |    | |  | (_) | |  | || |  __/| |__| | |_) |  __/| |  | (_| | |_| (_) | |
 * |_|    |_|   \___/|_|  |_||_|\___| \____/| .__/ \___||_|   \__,_|\__|\___/|_|
 *                                          | |
 *                                          |_|
 *
 * \brief Operator profiler object. Logs as both an independent event and a task in
 * the operator domain
 */
struct ProfileOperator : public ProfileEvent {
  /*!
   * \brief Operator attributes, used for additional naming when aggregate stats are enabled
   */
  struct Attributes {
    std::vector<nnvm::TShape> inputs_;
    std::vector<nnvm::TShape> outputs_;
    std::unordered_map<std::string, std::string> attr_;
    std::string to_string() const;
  };

  /*!
   * \brief Constructor
   * \param name Name of the operator
   */
  explicit inline ProfileOperator(const char *name, Attributes *attributes)
    : ProfileEvent(name)
      , as_task_(name, &domain_)
      , name_(name)
      , attributes_(attributes) {
    SetCategories(domain_.name());
  }

  /*!
   * \brief Start the profiling scope
   * \param dev_type Device type that the profiling will occur on
   * \param dev_id Device id associated with this opr
   */
  void start(mxnet::Context::DeviceType dev_type, uint32_t dev_id) {
    dev_type_ = dev_type;
    dev_id_ = dev_id;
    ProfileEvent::start();
    as_task_.start();
  }

  /*!
   * \brief Stop the profiling scope
   */
  void stop() override {
    as_task_.stop();
    ProfileEvent::stop();
  }

  /*!
   * \brief Operation execution statistics
   */
  struct OprExecStat : public DurationStat {
    /*!
     * \brief Constructor
     * \param name Name of the operator
     * \param dev_type Device type (i.e. CPU: 1, GPU: 2, CPUPinned: 3)
     * \param dev_id Device ID (ie GPU number)
     * \param start_time Time when operator starts
     * \param stop_time Time when operator completes
     */
    inline OprExecStat(const char *name, mxnet::Context::DeviceType dev_type, uint32_t dev_id,
                       uint64_t start_time, uint64_t stop_time,
                       Attributes *attributes)
      : DurationStat(ProfileStat::kDurationBegin, ProfileStat::kDurationEnd)
        , dev_type_(dev_type)
        , dev_id_(dev_id)
        , attributes_(attributes) {
      name_.set(name);
      categories_.set("operator");
      items_[kStart].timestamp_ = start_time;
      items_[kStop].timestamp_ = stop_time;
    }

    /*!
     * \brief Before emitting events, append name with attributes if necessary
     * \note This occurs in the dump thread and not in-line with the code being profiled
     */
    void OnBeforeEmitEvents() override {
      if(attributes_) {
        name_.append(" ");
        name_.append(attributes_->to_string().c_str());
      }
    }

    /*! \brief device type: CPU: 1, GPU: 2, CPUPinned: 3 */
    mxnet::Context::DeviceType dev_type_;
    /*! \brief device id */
    uint32_t dev_id_;
    /*! \brief Extra attributes */
    std::unique_ptr<Attributes> attributes_;
  };

 private:
  /*!
   * \brief Send this object's statistical datapoint to the profiler
   */
  void SendStat() override {
    Profiler::Get()->AddNewProfileStat<OprExecStat>(
      [this](OprExecStat *stat) {

      }, name_.c_str(), dev_type_, dev_id_,
      start_time_, ProfileStat::NowInMicrosec(),
      attributes_.release());
  }
  /*! \brief Also log the operator as a task in the operator domain */
  ProfileTask as_task_;
  /* !\brief Operator name */
  profile_stat_string name_;
  /*! \brief device type: CPU: 1, GPU: 2, CPUPinned: 3 */
  Context::DeviceType dev_type_;
  /*! \brief device id */
  uint32_t dev_id_;
  /*! \brief Operator domain */
  static ProfileDomain domain_;
  /*! \brief Optional operator attributes, ownership passed to OprExecStat */
  std::unique_ptr<Attributes> attributes_;
};

/*!
 * \brief Explicit 'Profiler::AddProfileStat' override for 'OprExecStat'
 * \param opr_stat Unique pointert to the operator statistic
 */
template<>
inline void Profiler::AddProfileStat<ProfileOperator::OprExecStat>(
  std::unique_ptr<ProfileOperator::OprExecStat> *opr_stat) {
  const size_t idx = DeviceIndex((*opr_stat)->dev_type_, (*opr_stat)->dev_id_);
  CHECK_LT(idx, DeviceCount());
  DeviceStats& dev_stat = profile_stat[idx];
  dev_stat.opr_exec_stats_->enqueue((*opr_stat).release());
}

}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_PROFILER_PROFILE_OPERATOR_H_
