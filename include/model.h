#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <box.hpp>
#include "texture.hpp"

struct TriangleMesh {
    std::vector<float3> vertex;
    std::vector<float3> normal;
    std::vector<float4> tangent;
    std::vector<float2> diffuseTexcoord;
    std::vector<float2> normalTexcoord;
    std::vector<float2> emissiveTexcoord;
    std::vector<uint3>  index;

    int     materialID          {-1};
};


struct Material {
    // ~Material()
    // {
    //     if (pixel){
    //         delete[] pixel;
    //     }
    // }

    float3 diffuse;

    // Phong モデルに基づくマテリアル表現のパラメタ
    float3 specular;
    float3 shininess;

    // PBR のマテリアル表現のパラメタ
    float   roughness;
    float   metallic;

    float3  emissive;

    bool    isLight                 {false};
    bool    isGlass                 {false};

    std::vector<uint32_t>           pixel;
    int2        resolution          {make_int2(-1, -1)};
    // 各種フラグ
    // bump map に使う
    int         normalTextureID     {-1};
    // PBR に使う
    int         diffuseTextureID    {-1};
    int         rmTextureID         {-1};

    // ライトに使う
    int         emissiveTextureID   {-1};

    unsigned int diffuseUVIndex     {0};
    unsigned int normalUVIndex      {0};
    unsigned int emissiveUVIndex    {0};
    unsigned int rmUVIndex          {0};
};

struct Model {
    ~Model()
    {
        for(auto mesh : meshes) delete mesh;
        for(auto material : materials) delete material;
        for(auto texture : textures) delete texture;
    }

    std::vector<TriangleMesh*>  meshes;
    std::vector<Material*>      materials;
    std::vector<Texture*>       textures;
    Box3f                       bounds;     // モデル全体を囲う Bounding box
};

Model *loadFBX(const std::string &fbxFileName);

#endif // MODEL_H_