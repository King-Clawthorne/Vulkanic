#include "config/RuntimeConfig.h"
#include "sky/RainbowScattering.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
void Check(bool p,const char* text) { if (!p) throw std::runtime_error(text); }
int main()
{
    try
    {
        const int bins=1025;
        std::vector<MieMatrixEntry> table(kSpectralBandCount*bins,{1,0,1,0});
        const auto original=table;
        AppendRainbowSamplingCdf(table,bins);
        const size_t offset=original.size();
        Check(std::memcmp(table.data(),original.data(),offset*sizeof(MieMatrixEntry))==0,"CDF altered phase table");
        Check(table[offset].f11==0 && table.back().f11==1,"CDF endpoints");
        for (int i=1;i<bins;++i)
        {
            Check(table[offset+i].f11>=table[offset+i-1].f11,"CDF monotonicity");
            Check(std::abs(table[offset+i].f11-0.5*(1-std::cos(kPi*i/(bins-1))))<1e-6,"Isotropic angular integral");
        }
        const auto config=ParseRuntimeConfig(R"({"rainbow":{"scatteringOrders":4,"multipleScatteringSamples":3,"multipleScatteringSteps":12},"sky":{"spectralConstants":{"SCATTERING_ORDERS":1}}})");
        Check(config.rainbow.scatteringOrders==4 && config.rainbow.multipleScatteringSamples==3
            && config.rainbow.multipleScatteringSteps==12 && config.skySpectral.scatteringOrders==1,"Independent rain controls");
        for (const char* invalid : {R"({"rainbow":{"scatteringOrders":0}})",R"({"rainbow":{"scatteringOrders":5}})",
            R"({"rainbow":{"multipleScatteringSamples":0}})",R"({"rainbow":{"multipleScatteringSteps":65}})"})
        {
            bool rejected=false;
            try { (void)ParseRuntimeConfig(invalid); } catch (const std::exception&) { rejected=true; }
            Check(rejected,"Invalid quality configuration accepted");
        }
        std::cout<<"PASS: sampling CDF, phase preservation, and independent config validation.\n";
        return 0;
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
