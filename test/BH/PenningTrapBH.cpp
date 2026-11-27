// test/BH/PenningTrapBH.cpp
//
// Penning trap — Barnes-Hut backend.
//
// Usage:
//   PenningTrapBH <N> <Nt> <theta> [seed] [smoothH] [B0]
//                 [--precision=double|mixed|float] [--leaf-h=on|off]

#include "Ippl.h"
#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "IpplArgs.hpp"
#include "PenningTrapBHManager.hpp"

const char* TestName = "PenningTrapBH";

namespace {

struct CliInputs {
    unsigned long N         = 1000000;
    int           Nt        = 300;
    float         theta     = 0.5f;
    unsigned long seed      = 42;
    double        smoothH   = 0.1;
    double        B0        = 5.0;
    bool          leafBasedH = true;
};

template <class P>
int runWith(const CliInputs& in, Inform& msg) {
    using Manager = ippl::nbody::PenningTrapBHManager<P, 3>;
    typename Manager::Config cfg;
    cfg.N         = in.N;
    cfg.Nt        = in.Nt;
    cfg.theta     = in.theta;
    cfg.seed      = in.seed;
    cfg.smoothH   = static_cast<typename P::Th>(in.smoothH);
    cfg.B0        = static_cast<typename P::Tc>(in.B0);
    cfg.leafBasedH = in.leafBasedH;

    Manager manager(cfg);
    manager.pre_run();

    msg << "Starting iterations ..." << endl;
    manager.run(manager.getNt());
    msg << "End." << endl;
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    if (ippl::Info->getOutputLevel() < 1)
        ippl::Info->setOutputLevel(1);

    int returnCode = 0;
    {
        Inform msg(TestName);

        auto pos = ippl::nbody::filterIpplFlags(argc, argv);

        std::string precision;
        bool        leafBasedH;
        try {
            precision  = ippl::nbody::extractPrecisionFlag(pos);
            leafBasedH = ippl::nbody::extractLeafHFlag(pos, /*defaultOn=*/true);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            ippl::finalize();
            return 2;
        }

        CliInputs in;
        in.leafBasedH = leafBasedH;
        if (pos.size() > 1) in.N       = std::atoll(pos[1]);
        if (pos.size() > 2) in.Nt      = std::atoi(pos[2]);
        if (pos.size() > 3) in.theta   = std::atof(pos[3]);
        if (pos.size() > 4) in.seed    = std::atoll(pos[4]);
        if (pos.size() > 5) in.smoothH = std::atof(pos[5]);
        if (pos.size() > 6) in.B0      = std::atof(pos[6]);

        msg << "precision = " << precision
            << ", N = "       << in.N
            << ", Nt = "      << in.Nt
            << ", theta = "   << in.theta
            << ", seed = "    << in.seed
            << ", smoothH = " << in.smoothH
            << ", B0 = "      << in.B0
            << ", leaf-h = "  << (in.leafBasedH ? "on" : "off") << endl;

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        try {
            if (precision == "double") {
                returnCode = runWith<ippl::nbody::DoublePrecision>(in, msg);
            } else if (precision == "mixed") {
                returnCode = runWith<ippl::nbody::MixedPrecision>(in, msg);
            } else {
                returnCode = runWith<ippl::nbody::FloatPrecision>(in, msg);
            }
        } catch (const IpplException& e) {
            std::fprintf(stderr, "[rank %d] IpplException in %s: %s\n",
                         ippl::Comm->rank(), e.where().c_str(), e.what());
            returnCode = 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rank %d] std::exception: %s\n",
                         ippl::Comm->rank(), e.what());
            returnCode = 1;
        }

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();
    return returnCode;
}