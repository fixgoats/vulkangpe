#pragma once
#include <boost/pfr/core.hpp>
#include <type_traits>

template <typename T, typename U, typename = void>
struct is_safely_castable : std::false_type {};

template <typename T, typename U>
struct is_safely_castable<
    T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};

template <class T, class B>
T* pcast(B* x) {
  // bit_cast that checks at compile time if B is statically castable to T
  static_assert(is_safely_castable<T, B>(), "Types are not equivalent");
  return std::bit_cast<T*>(x);
}

template <auto Start, auto End, auto Inc, class F>
consteval void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

template <class T>
consteval std::array<size_t, boost::pfr::tuple_size_v<T>> struct_field_sizes() {
  constexpr size_t n = boost::pfr::tuple_size_v<T>;
  constexpr std::array<size_t, n> sizes;
  constexpr_for<0, n, 1>([&sizes](auto i) {
    sizes[i] = sizeof(boost::pfr::tuple_element_t<i, T>);
  });
  return sizes;
}
