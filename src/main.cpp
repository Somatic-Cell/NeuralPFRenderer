#include "optix8.hpp"
#include <stdexcept>
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <assert.h>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "application.hpp"



extern "C" int main(int ac, char **av)
  {

    // FBX ファイルの読み込み
    std::vector<std::string> modelFileNames;

    // modelFileNames.push_back("Bistro_V5_2\\BistroExterior.fbx");
    // modelFileNames.push_back("Bistro_v5_2\\BistroInterior.fbx");
    // modelFileNames.push_back("EmeraldSquare_v4_1\\EmeraldSquare_Dusk.fbx");
    // modelFileNames.push_back("EmeraldSquare_v4_1\\EmeraldSquare_Day.fbx");
    modelFileNames.push_back("sponza\\sponza.obj");
    // modelFileNames.push_back("tokyo\\53394525_bldg_6677.fbx");
    // modelFileNames.push_back("main_sponza\\NewSponza_Main_Yup_003.fbx");
    // modelFileNames.push_back("pkg_a_curtains\\NewSponza_Curtains_FBX_YUp.fbx");
    // modelFileNames.push_back("CornellBox\\CornellBox-Original.obj");
    // modelFileNames.push_back("CornellBox\\water.obj");

    // const std::string envMapFileName("san_giuseppe_bridge_4k.hdr");
    // const std::string envMapFileName("night_sky.hdr");
    const std::string envMapFileName("symmetrical_garden_4k.hdr");

    try {

        std::vector<const Model*> models;
        models.reserve(modelFileNames.size());
        for(const auto& path : modelFileNames){
            std::cout << "Model File Name:" << path << std::endl;
            if(auto* m = loadModel(path)){
              models.push_back(m);
            }
        }

        

        
        Camera camera;

        const float worldScale = length(models[0]->bounds.getSpan()); // MEMO : 後で修正

        Application *application = new Application("Photonic RT", camera, models, envMapFileName, worldScale);

        application->enableFlyMode();
        application->run();
      
    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }