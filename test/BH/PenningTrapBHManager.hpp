// test/BH/PenningTrapBHManager.hpp
//
// Penning trap — Barnes-Hut backend.
// Mirrors alpine/PenningTrapManager.h but routes the field solve through
// ippl::nbody::NBodySolver instead of FFT-PIC.
//
// Physics:
//   - Particles confined in a Penning trap: electrostatic self-field (BH)
//     + constant axial magnetic field B = B0 * z-hat
//   - Leapfrog with Boris rotation for the magnetic kick
//   - Open boundary conditions (particles stay inside the trap)
//
// Step ordering: kickBorisHalf - drift - BH solve - kickBorisHalf

#ifndef IPPL_BH_PENNING_TRAP_BH_MANAGER_HPP
#define IPPL_BH_PENNING_TRAP_BH_MANAGER_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "NBody/NBodyManager.hpp"
#include "NBody/NBodyParticleContainer.hpp"

#include "BHRandom.hpp"

// Do NOT define file-scope Conserved/Dependent aliases here —
// use fields::StdConserved / fields::StdDependent from BHFieldLists.hpp
// (pulled in transitively through NBodyParticleContainer.hpp).

namespace ippl::nbody {

// Forward declaration of reduceExSumSq (defined in LandauDampingBHManager.cu/.cpp)
template <class P>
typename P::Tc reduceExSumSq(NBodyParticleContainer<P, 3>& pc);

// Forward declaration of applyBorisKick (defined in PenningTrapBHManager.cu/.cpp)
template <class P>
void applyBorisKick(NBodyParticleContainer<P, 3>& pc,
                    typename P::Tc dt_half,
                    typename P::Tc B0,
                    typename P::Tc qm);

// Device-parallel Penning trap IC sampler.
// Anisotropic Gaussian in position (sigma_x, sigma_y, sigma_z),
// zero initial velocity (trap holds particles from rest).
template <class P>
inline void samplePenningIC(NBodyParticleContainer<P, 3>& pc,
                             unsigned localN,
                             unsigned long firstGlobal,
                             typename P::Tc sigma_x,
                             typename P::Tc sigma_y,
                             typename P::Tc sigma_z,
                             typename P::Tc cx,
                             typename P::Tc cy,
                             typename P::Tc cz,
                             typename P::Th smoothingH,
                             typename P::Tm qPerParticle,
                             unsigned long seed) {
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;
    if (localN == 0) { return; }

    Tc* Rx = getRaw<"Rx">(pc); Tc* Ry = getRaw<"Ry">(pc); Tc* Rz = getRaw<"Rz">(pc);
    Tc* px = getRaw<"Px">(pc); Tc* py = getRaw<"Py">(pc); Tc* pz = getRaw<"Pz">(pc);
    Tm* q  = getRaw<"charge">(pc);
    Th* h  = getRaw<"h">(pc);

    // Copy all parameters into local const variables so the
    // KOKKOS_LAMBDA can capture them by value in CUDA.
    const uint64_t base    = static_cast<uint64_t>(seed)
                           + static_cast<uint64_t>(firstGlobal);
    const Tc lSigmaX       = sigma_x;
    const Tc lSigmaY       = sigma_y;
    const Tc lSigmaZ       = sigma_z;
    const Tc lCx           = cx;
    const Tc lCy           = cy;
    const Tc lCz           = cz;
    const Th lSmoothingH   = smoothingH;
    const Tm lQPerParticle = qPerParticle;

    Kokkos::parallel_for(
        "samplePenningIC", static_cast<long>(localN),
        KOKKOS_LAMBDA(const long t) {
            uint64_t s = base + static_cast<uint64_t>(t);
            Rx[t] = lCx + lSigmaX * bhNormal<Tc>(s);
            Ry[t] = lCy + lSigmaY * bhNormal<Tc>(s);
            Rz[t] = lCz + lSigmaZ * bhNormal<Tc>(s);
            px[t] = Tc(0);
            py[t] = Tc(0);
            pz[t] = Tc(0);
            q[t]  = lQPerParticle;
            h[t]  = lSmoothingH;
        });
    Kokkos::fence();
}

template <class P, unsigned Dim>
class PenningTrapBHManager : public NBodyManager<P, Dim> {
    static_assert(Dim == 3, "PenningTrapBHManager requires Dim == 3");

    using Base = NBodyManager<P, Dim>;
    using typename Base::Container;
    using typename Base::Solver;
    using typename Base::SolverParams;
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    using Tm = typename P::Tm;

public:
    struct Config {
        unsigned long N          = 1000000;
        int           Nt         = 300;
        float         theta      = 0.5f;
        unsigned long seed       = 42;
        int           numShells  = 1;
        Th            smoothH    = Th(0.1);   // softening ~ mean interparticle spacing
        bool          leafBasedH = true;
        // Penning trap physical parameters (normalised units, matching Alpine)
        Tc            B0         = Tc(5.0);   // axial magnetic field
        Tc            rmax       = Tc(20.0);  // trap box half-size
    };

    explicit PenningTrapBHManager(Config cfg)
        : Base(cfg.N, cfg.Nt,
               /*dt=*/Tc(0.5) * (cfg.rmax / Tc(2048)))  // dt = 0.5 * dx_finest
        , cfg_m(cfg)
        , L_m(cfg.rmax) {}

protected:

    void prepareSolverInputs(bool collect) override {
        auto& pc = this->pc();
        {
            static auto t = IpplTimings::getTimer("bh.syncGrav");
            ippl::nbody::GpuTimer scope(t, collect);
            // Use the namespaced aliases from BHFieldLists.hpp —
            // do NOT use file-scope Conserved/Dependent.
            syncGravBH<P, fields::StdConserved, fields::StdDependent>(pc);
        }
        {
            static auto t = IpplTimings::getTimer("bh.haloCharge");
            ippl::nbody::GpuTimer scope(t, collect);
            pc.exchangeHalos(std::tie(pc.template getDV<"charge">()),
                             pc.haloSendBuf(), pc.haloRecvBuf());
        }
    }

    void initializeContainer() override {
        using cstone::BoundaryType;
        const unsigned bucketSizeFocus = 64u;
        const unsigned bucketSize = std::max<unsigned>(
            bucketSizeFocus,
            static_cast<unsigned>(cfg_m.N / (100ul * this->numRanks())));
        // Penning trap: open boundaries (particles are confined by the trap
        // fields, not by periodic wrapping)
        auto pc = std::make_unique<Container>(
            this->rank(), this->numRanks(),
            bucketSize, bucketSizeFocus, cfg_m.theta,
            std::array<Tc, 6>{-L_m, L_m, -L_m, L_m, -L_m, L_m},
            std::array<BoundaryType, 3>{
                BoundaryType::open, BoundaryType::open, BoundaryType::open});
        this->setContainer(std::move(pc));
    }

    void initializeParticles() override {
        // Anisotropic Gaussian IC matching Alpine PenningTrapManager:
        //   sigma = (0.15, 0.05, 0.20) * domain_length
        const Tc length = Tc(2) * L_m;
        const Tc sx = Tc(0.15) * length;
        const Tc sy = Tc(0.05) * length;
        const Tc sz = Tc(0.20) * length;
        const Tc cx = Tc(0), cy = Tc(0), cz = Tc(0);  // centred at origin
        const Tm qPerParticle = static_cast<Tm>(
            -volume_m() / static_cast<Tc>(this->N()));

        samplePenningIC<P>(this->pc(),
                           static_cast<unsigned>(this->localN()),
                           this->firstGlobal(),
                           sx, sy, sz, cx, cy, cz,
                           cfg_m.smoothH, qPerParticle,
                           cfg_m.seed);
        this->pc().setUniformH(cfg_m.smoothH);
        this->pc().setLeafBasedH(cfg_m.leafBasedH);
    }

    void initializeSolverParams(SolverParams& p) override {
        // Electrostatic Coulomb (q/m = -1, plasma units)
        p.G         = -Tc(1) / (Tc(4) * Tc(M_PI));
        p.theta     = cfg_m.theta;
        p.numShells = cfg_m.numShells;
    }

    void post_initial_solve() override {
        openCsvHeader();
#ifdef IPPL_ENABLE_OPENPMD
        this->openOpenPMD("PenningTrapBH_" + std::to_string(this->N()) + ".h5");
#endif
    }

    void advanceImpl() override {
        // Boris leapfrog: half E-kick + B-rotation, drift, BH solve,
        // half E-kick + B-rotation
        borisKickHalf();
        this->drift();
        this->solve();
        borisKickHalf();
    }

    void dumpImpl() override {
        // Particle-sampled MC estimator of field energy (same as LandauDampingBH)
        const Tc sumSq       = reduceExSumSq<P>(this->pc());
        const Tc fieldEnergy = (volume_m() / static_cast<Tc>(this->N())) * sumSq;

        if (this->rank() != 0) { return; }
        std::ofstream row(csvPath_m, std::ios::app);
        row.precision(16);
        row.setf(std::ios::scientific, std::ios::floatfield);
        row << this->time() << " " << fieldEnergy << "\n";
    }

private:
    void borisKickHalf() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("borisKick");
        ScopedIpplTimer scope(t);
        // q/m = -1 for electrons in plasma units
        applyBorisKick<P>(this->pc(),
                          static_cast<Tc>(this->dt()) * Tc(0.5),
                          cfg_m.B0,
                          /*qm=*/Tc(-1));
    }

    void openCsvHeader() {
        std::ostringstream oss;
        oss << "data/FieldPenningBH_" << this->N() << ".csv";
        csvPath_m = oss.str();
        if (this->rank() != 0) { return; }
        std::ofstream header(csvPath_m, std::ios::out);
        header << "time Ex_field_energy\n";
    }

    Tc volume_m() const { return Tc(8) * L_m * L_m * L_m; }

    Config      cfg_m;
    const Tc    L_m;
    std::string csvPath_m;
};

}  // namespace ippl::nbody

#endif  // IPPL_BH_PENNING_TRAP_BH_MANAGER_HPP