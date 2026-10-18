// Sirius allocated-job helper, not a benchmark or teaching example.
#include <hip/hip_runtime_api.h>
#include <omp.h>
#include <cstdio>
#include <cstring>

static bool checked(hipError_t status, const char *operation) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    return false;
}

int main(int argc, char **argv) {
    if (argc != 2 || (std::strcmp(argv[1], "identity") && std::strcmp(argv[1], "verify"))) return 2;
    int count = 0, device = -1;
    char pci[32] = {};
    hipUUID uuid = {};
    if (!checked(hipGetDeviceCount(&count), "hipGetDeviceCount") || count < 1 ||
        !checked(hipSetDevice(0), "hipSetDevice") ||
        !checked(hipGetDevice(&device), "hipGetDevice") ||
        !checked(hipDeviceGetPCIBusId(pci, sizeof(pci), device), "hipDeviceGetPCIBusId") ||
        !checked(hipDeviceGetUuid(&uuid, device), "hipDeviceGetUuid")) return 1;
    char identity[33] = {};
    for (unsigned i = 0; i < 16; ++i)
        std::snprintf(identity + 2 * i, 3, "%02x", static_cast<unsigned char>(uuid.bytes[i]));
    bool verified = false;
    if (!std::strcmp(argv[1], "verify")) {
        if (omp_get_num_devices() < 1 || omp_get_default_device() != 0) return 1;
        int value = 0, initial = 1;
        int *data = &value;
        bool same_device = false;
        #pragma omp target data map(tofrom: data[0:1]) use_device_ptr(data)
        {
            hipPointerAttribute_t attributes = {};
            same_device = checked(hipPointerGetAttributes(&attributes, data), "hipPointerGetAttributes") &&
                          attributes.type == hipMemoryTypeDevice && attributes.device == device;
            if (same_device) {
                #pragma omp target is_device_ptr(data) map(from: initial)
                { initial = omp_is_initial_device(); data[0] = 1; }
            }
        }
        verified = same_device && initial == 0 && value == 1;
        if (!verified) {
            std::fprintf(stderr, "HIP/OpenMP device correspondence was not established\n");
            return 1;
        }
    }
    std::printf("{\"hip_device\":%d,\"visible_hip_devices\":%d,\"pci_bus_id\":\"%s\","
                "\"gpu_uuid\":\"%s\",\"hip_openmp_same_device\":%s}\n",
                device, count, pci, identity, verified ? "true" : "false");
    return 0;
}
