#include "config/RuntimeConfig.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

int main()
{
    try
    {
        for (float absorption : {0.0f, 8e-5f, 2e-4f})
        {
            RainbowConfig rain;
            rain.extinctionCoefficient = rain.scatteringCoefficient + absorption;
            const float initialExtinction = rain.extinctionCoefficient;
            for (int cycle = 0; cycle < 100; ++cycle)
            {
                for (float displayed : {5.0f, 0.0f, 3.0f, 1.2f})
                {
                    SetRainbowScattering(rain, displayed * 1e-4f);
                    if (std::abs((rain.extinctionCoefficient-rain.scatteringCoefficient)-absorption)>1e-9f)
                        throw std::runtime_error("Slider changed absorption");
                }
                if (std::abs(rain.extinctionCoefficient-initialExtinction)>1e-9f)
                    throw std::runtime_error("Returning to 1.2 changed extinction");
            }
        }
        RainbowConfig rain;
        SetRainbowScattering(rain, 5e-4f);
        SetRainbowScattering(rain, 0.0f);
        SetRainbowScattering(rain, 1.2f * 1e-4f);
        // Machine-readable values for a renderer capture of the actual round trip.
        std::cout << std::setprecision(9) << "{\"scatteringCoefficient\":" << rain.scatteringCoefficient
                  << ",\"extinctionCoefficient\":" << rain.extinctionCoefficient << "}\n";
        return 0;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
