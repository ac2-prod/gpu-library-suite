#ifndef GPU_SUITE_SYNTAX_ROCFFT_H
#define GPU_SUITE_SYNTAX_ROCFFT_H
// Declaration-only fixture: compile success is not a real ROCm compile/link check.
#include <stddef.h>
typedef struct rocfft_plan_t *rocfft_plan;
typedef struct rocfft_execution_info_t *rocfft_execution_info;
typedef struct rocfft_plan_description_t *rocfft_plan_description;
enum rocfft_status { rocfft_status_success = 0 };
enum rocfft_result_placement { rocfft_placement_notinplace };
enum rocfft_transform_type { rocfft_transform_type_complex_forward };
enum rocfft_precision { rocfft_precision_single };
rocfft_status rocfft_setup();
rocfft_status rocfft_cleanup();
rocfft_status rocfft_plan_create(rocfft_plan *, rocfft_result_placement,
  rocfft_transform_type, rocfft_precision, size_t, const size_t *, size_t,
  const rocfft_plan_description);
rocfft_status rocfft_plan_destroy(rocfft_plan);
rocfft_status rocfft_execution_info_create(rocfft_execution_info *);
rocfft_status rocfft_execution_info_destroy(rocfft_execution_info);
rocfft_status rocfft_plan_get_work_buffer_size(rocfft_plan, size_t *);
rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info, void *, size_t);
rocfft_status rocfft_execute(rocfft_plan, void **, void **, rocfft_execution_info);
rocfft_status rocfft_get_version_string(char *, size_t);
#endif
