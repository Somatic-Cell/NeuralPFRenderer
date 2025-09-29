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

#include "application.hpp"



extern "C" int main(int ac, char **av)
  {

    // FBX ファイルの読み込み
    std::vector<std::string> fbxFileNames;

    // fbxFileNames.push_back("chess_set_2k\\chess_set_2k.fbx");
    fbxFileNames.push_back("Bistro_V5_2\\BistroExterior.fbx");
    // fbxFileNames.push_back("Bistro_v5_2\\BistroInterior.fbx");
    // fbxFileNames.push_back("EmeraldSquare_v4_1\\EmeraldSquare_Dusk.fbx");
    // fbxFileNames.push_back("EmeraldSquare_v4_1\\EmeraldSquare_Day.fbx");

    const std::string envMapFileName("san_giuseppe_bridge_4k.hdr");
    // const std::string envMapFileName("night_sky.hdr");
    // const std::string envMapFileName("symmetrical_garden_4k.hdr");

    try {

        Model* model;
        for(int i = 0; i < fbxFileNames.size(); i++){
            std::cout << "FBX File Name:" << fbxFileNames[i] << std::endl;
            model = loadFBX(fbxFileNames[i]);
        }

        
        Camera camera;

        const float worldScale = length(model->bounds.getSpan());

        Application *application = new Application("Simple Renderer", camera, model, envMapFileName, worldScale);

        application->enableFlyMode();
        application->run();
      
    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }