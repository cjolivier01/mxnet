#include "./profile_operator.h"

namespace mxnet {
namespace profiler {

ProfileDomain ProfileOperator::domain_("operator");

std::string ProfileOperator::Attributes::to_string() const {
  std::stringstream ss;
  if(!inputs_.empty()) {
    ss << "in: [";
    for(size_t i = 0, n = inputs_.size(); i < n; ++i) {
      if(i) {
        ss << ",";
      }
      ss << inputs_[i];
    }
    ss << "]";
  }
  if(!outputs_.empty()) {
    ss << "out: [";
    for(size_t i = 0, n = outputs_.size(); i < n; ++i) {
      if(i) {
        ss << ",";
      }
      ss << outputs_[i];
    }
    ss << "]";
  }
  if(!attr_.empty()) {
    for (const auto &tt : attr_) {
      ss << " (" << tt.first << "=" << tt.second << ")";
    }
  }
  return ss.str();
}

}  // namespace profiler
}  // namespace mxnet
