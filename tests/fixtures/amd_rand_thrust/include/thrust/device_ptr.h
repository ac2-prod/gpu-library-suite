#pragma once
namespace thrust {
template<class T> T *device_pointer_cast(T *p) { return p; }
}
