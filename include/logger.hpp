#pragma once
#include <iostream>

inline void logDebug(std::string_view s) {
#if !NDEBUG
  std::cout << "Debug: " << s << std::endl;
#endif // !NDEBUG
}
