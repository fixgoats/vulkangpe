#pragma once
#include <algebra.slang>

interface ISystem {
  This system();
}

T RK4Step<T, each G>(T y, float dt, expand each G args)
    where T : IVec, ISystem {
  T k1 = y.system();
  T k2 = (y + k1.scalar_mul(0.5 * dt)).system();
  T k3 = (y + k2.scalar_mul(0.5 * dt)).system();
  T k4 = (y + k3.scalar_mul(dt)).system();
  return y +
         (k1 + k2.scalar_mul(2.) + k3.scalar_mul(2.) + k4).scalar_mul(dt / 6.);
}
