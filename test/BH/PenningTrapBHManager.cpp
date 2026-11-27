// test/BH/PenningTrapBHManager.cpp  — CPU/OpenMP backend
#include "NBody/NBodyParticleContainer.hpp"
#include "PenningTrapBHManager.hpp"

namespace ippl::nbody {

template <class P>
void applyBorisKick(NBodyParticleContainer<P, 3>& pc,
                    typename P::Tc dt_half,
                    typename P::Tc B0,
                    typename P::Tc qm) {
    // identical body — Kokkos::parallel_for dispatches to OpenMP on CPU
    // ... (same as .cu above)
}

template void applyBorisKick<DoublePrecision>(...);
template void applyBorisKick<MixedPrecision>(...);
template void applyBorisKick<FloatPrecision>(...);

}  // namespace ippl::nbody