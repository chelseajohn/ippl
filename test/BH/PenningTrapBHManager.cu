// test/BH/PenningTrapBHManager.cu
// GPU kernel for the Boris rotation kick in the Penning trap BH manager.

#include "NBody/NBodyParticleContainer.hpp"
#include "PenningTrapBHManager.hpp"

namespace ippl::nbody {

template <class P>
void applyBorisKick(NBodyParticleContainer<P, 3>& pc,
                    typename P::Tc dt_half,
                    typename P::Tc B0,
                    typename P::Tc qm) {
    using Tc = typename P::Tc;

    const std::size_t localN = static_cast<std::size_t>(pc.getLocalNum());

    using Ta = typename P::Ta;
    Ta* Ex = getRaw<"Ex">(pc); Ta* Ey = getRaw<"Ey">(pc); Ta* Ez = getRaw<"Ez">(pc);
    Tc* px = getRaw<"Px">(pc); Tc* py = getRaw<"Py">(pc); Tc* pz = getRaw<"Pz">(pc);

    // Boris algorithm:
    //   v_minus = v + 0.5 * qm * E * dt
    //   t = qm * B * dt/2  (B = B0 * z-hat => t = {0, 0, qm*B0*dt_half})
    //   s = 2*t / (1 + |t|^2)
    //   v_prime = v_minus + v_minus x t
    //   v_plus  = v_minus + v_prime x s
    //   v_new   = v_plus + 0.5 * qm * E * dt

    Kokkos::parallel_for(
        "borisKick", static_cast<long>(localN),
        KOKKOS_LAMBDA(const long i) {
            // half E-kick
            Tc vx = px[i] + dt_half * qm * static_cast<Tc>(Ex[i]);
            Tc vy = py[i] + dt_half * qm * static_cast<Tc>(Ey[i]);
            Tc vz = pz[i] + dt_half * qm * static_cast<Tc>(Ez[i]);

            // Boris rotation (B = B0 z-hat)
            const Tc tz = qm * B0 * dt_half;
            const Tc s  = Tc(2) * tz / (Tc(1) + tz * tz);

            const Tc vx_prime = vx + vy * tz;
            const Tc vy_prime = vy - vx * tz;
            // vz unchanged by z-aligned B

            vx = vx + vy_prime * s;
            vy = vy - vx_prime * s;

            // half E-kick again
            px[i] = vx + dt_half * qm * static_cast<Tc>(Ex[i]);
            py[i] = vy + dt_half * qm * static_cast<Tc>(Ey[i]);
            pz[i] = vz + dt_half * qm * static_cast<Tc>(Ez[i]);
        });
    Kokkos::fence();
}

// Explicit instantiations
template void applyBorisKick<DoublePrecision>(
    NBodyParticleContainer<DoublePrecision, 3>&,
    DoublePrecision::Tc, DoublePrecision::Tc, DoublePrecision::Tc);
template void applyBorisKick<MixedPrecision>(
    NBodyParticleContainer<MixedPrecision, 3>&,
    MixedPrecision::Tc, MixedPrecision::Tc, MixedPrecision::Tc);
template void applyBorisKick<FloatPrecision>(
    NBodyParticleContainer<FloatPrecision, 3>&,
    FloatPrecision::Tc, FloatPrecision::Tc, FloatPrecision::Tc);

}  // namespace ippl::nbody