#ifndef SCENE_IO_HPP_
#define SCENE_IO_HPP_

#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "launch_params.h"

namespace sceneIO {

    inline void toJson(nlohmann::json& j, const LaunchParams& lp)
    {
        j = {

        };
    }
    
}

#endif // SCENE_IO_HPP_