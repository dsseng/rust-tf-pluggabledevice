#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"

struct SP_Stream_st {
  explicit SP_Stream_st(void* stream_h) : stream_handle(stream_h) {}
  void* stream_handle;
};

struct SP_Event_st {
  explicit SP_Event_st(void* event_h) : event_handle(event_h) {}
  void* event_handle;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
};
