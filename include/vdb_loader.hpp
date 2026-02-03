#ifndef VDB_LOADER_HPP_
#define VDB_LOADER_HPP_

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridStats.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <mutex>

#include "optix8.hpp"

#include "cuda_buffer.h"


// -----------------------------------
// OpenVDB の初期化
// -----------------------------------
inline void ensureOpenVDBInitialized()
{
    static std::once_flag once;
    std::call_once(once, []() {openvdb::initialize(); });
}

// -----------------------------------
// VDB の 1種類のグリッドを表す構造体
// -----------------------------------
struct NanoVDBGrid {
    // NanoVDBGrid(const NanoVDBGrid&) = default;
    // NanoVDBGrid& operator=(const NanoVDBGrid&) = default;

    std::string name;
    std::string vdbType;
    CUDABuffer  buffer;
    size_t      bytes = 0;
    size_t      gridOffsetBytes = 0;
    std::array<float, 3> worldMin{0, 0, 0};
    std::array<float, 3> worldMax{0, 0, 0};

    CUdeviceptr deviceGridPtr() const {return buffer.getDevicePointer() + gridOffsetBytes;}
};

// -----------------------------------
// VDB ファイルをまとめて管理するクラス
// -----------------------------------
class NanoVDBVolumeAsset
{
public:
    NanoVDBVolumeAsset() = default;

    // リソース所有クラスなので
    // copy コンストラクタとコピー代入を禁止
    NanoVDBVolumeAsset(const NanoVDBVolumeAsset&) = delete;
    NanoVDBVolumeAsset& operator=(const NanoVDBVolumeAsset&) = delete;

    // move は OK
    NanoVDBVolumeAsset(NanoVDBVolumeAsset&&) noexcept = default;
    NanoVDBVolumeAsset& operator=(NanoVDBVolumeAsset&&) noexcept = default;
    
    // float グリッドをすべてロード
    void loadAllFloatGrids(const std::string& vdbPath)
    {
        ensureOpenVDBInitialized();

        openvdb::io::File file(vdbPath);
        file.open();
        auto grids = file.getGrids();

        m_path = vdbPath;
        m_grids.clear();

        for(auto& base : *grids) {
            auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
            if(!floatGrid) continue;

            NanoVDBGrid grid    = buildFromFloatGrid(*floatGrid);
            grid.name           = base->getName();
            grid.vdbType        = base->type();

            m_grids[grid.name] = std::move(grid);
        }

        file.close();

        if(m_grids.empty())
        {
            throw std::runtime_error("no FloatGrid found in VDB file: " + vdbPath);
        }
    }

    // 指定したグリッド名だけをロード
    void loadSelectedFloatGrids(
        const std::string& vdbPath, 
        const std::vector<std::string>& gridNames
    ){
        ensureOpenVDBInitialized();

        openvdb::io::File file(vdbPath);
        file.open();

        m_path = vdbPath;
        m_grids.clear();

        for(const auto& gridName : gridNames){
            openvdb::GridBase::Ptr base = file.readGrid(gridName);
            auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);

            if(!floatGrid){
                throw std::runtime_error("Grid is not openVDB::FloatGrid (gridName = " + gridName + ")"); 
            }

            NanoVDBGrid grid    = buildFromFloatGrid(*floatGrid);
            grid.name           = gridName;
            grid.vdbType        = base->type();

            m_grids[grid.name] = std::move(grid);
        }

        file.close();

        if(m_grids.empty()) {
            throw std::runtime_error("No grids loaded from VDB file: " + vdbPath); 
        }
    }

    bool hasGrid(const std::string& name) const 
    {
        return m_grids.find(name) != m_grids.end();
    }

    const NanoVDBGrid& getGrid(const std::string& name) const
    {
        auto it = m_grids.find(name);
        if(it == m_grids.end()){
            throw std::runtime_error("Grid not found: " + name);
        }
        return it->second;
    }

    std::vector<std::string> getListGridNames() const 
    {
        std::vector<std::string> names;
        names.reserve(m_grids.size());
        for(const auto& kv : m_grids) names.push_back(kv.first);
        std::sort(names.begin(), names.end());
        return names;
    }

    // Density らしい名前を優先して返す
    std::string chooseDefaultDensityLikeName() const
    {
        const std::vector<std::string> preferred = {"density", "dens", "cloud", "fog", "volume"};
        
        // 候補を探す
        for (const auto& want : preferred) {
            for(const auto& kv : m_grids){
                if(isEquals(kv.first, want)) return kv.first;
            }
        }

        // なければ一番最初を返す
        return m_grids.begin()->first;
    }

    std::string m_path;
    std::unordered_map<std::string, NanoVDBGrid> m_grids;



private:
    static bool isEquals(const std::string& a, const std::string& b){
        if(a.size() != b.size()) return false;
        for(size_t i = 0; i < a.size(); ++i){
            if(std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
        }
        return true;
    }

    
    static NanoVDBGrid buildFromFloatGrid(openvdb::FloatGrid& floatGrid){
        NanoVDBGrid out;

        // Active voxel bbox
        openvdb::CoordBBox bbox = floatGrid.evalActiveVoxelBoundingBox();
        if(bbox.empty()){
            throw std::runtime_error("Active voxel bbox is empty (no active voxels...?).");
        }

        const auto bMin = bbox.min();
        const auto bMax = bbox.max();

        const openvdb::Vec3d corners[8]{
            floatGrid.indexToWorld(openvdb::Vec3d(bMin.x(),   bMin.y(),   bMin.z())),
            floatGrid.indexToWorld(openvdb::Vec3d(bMax.x()+1, bMin.y(),   bMin.z())),
            floatGrid.indexToWorld(openvdb::Vec3d(bMin.x(),   bMax.y()+1, bMin.z())),
            floatGrid.indexToWorld(openvdb::Vec3d(bMax.x()+1, bMax.y()+1, bMin.z())),
            floatGrid.indexToWorld(openvdb::Vec3d(bMin.x(),   bMin.y(),   bMax.z()+1)),
            floatGrid.indexToWorld(openvdb::Vec3d(bMax.x()+1, bMin.y(),   bMax.z()+1)),
            floatGrid.indexToWorld(openvdb::Vec3d(bMin.x(),   bMax.y()+1, bMax.z()+1)),
            floatGrid.indexToWorld(openvdb::Vec3d(bMax.x()+1, bMax.y()+1, bMax.z()+1)),
        };

        double bboxMin[3] = {corners[0].x(), corners[0].y(), corners[0].z()};
        double bboxMax[3] = {corners[0].x(), corners[0].y(), corners[0].z()};

        for(int i = 1; i < 8; ++i){
            bboxMin[0] = std::min(bboxMin[0], corners[i].x());
            bboxMax[0] = std::max(bboxMax[0], corners[i].x());
            bboxMin[1] = std::min(bboxMin[1], corners[i].y());
            bboxMax[1] = std::max(bboxMax[1], corners[i].y());
            bboxMin[2] = std::min(bboxMin[2], corners[i].z());
            bboxMax[2] = std::max(bboxMax[2], corners[i].z());
        }

        out.worldMin = {(float)bboxMin[0], (float)bboxMin[1], (float)bboxMin[2]};
        out.worldMax = {(float)bboxMax[0], (float)bboxMax[1], (float)bboxMax[2]};

        // OpenVDB -> NanoVDB
        auto hostHandle = nanovdb::tools::createNanoGrid(floatGrid);
        out.bytes = hostHandle.size();

        if(out.bytes == 0 || hostHandle.data() == nullptr){
            throw std::runtime_error("NanoVDB handle is empty.");
        }
        auto* hGrid = hostHandle.grid<float>();
        if(!hGrid){
            throw std::runtime_error("Converted NanoVDB is not NanoGir<float>");
        }

        // min , max と bbox を再計算
        nanovdb::tools::updateGridStats(hGrid, nanovdb::tools::StatsMode::All);

        // 作成できたかどうか確認
        if(!hGrid->hasMinMax()){
            throw std::runtime_error("gridStats failed: grid has no Min/Max.");
        }

        // grid の先頭が必ずしも buffer の先頭とは限らない
        const auto* base = reinterpret_cast<const uint8_t*>(hostHandle.data());
        const auto* gridPtr = reinterpret_cast<const uint8_t*>(hGrid);
        out.gridOffsetBytes = static_cast<size_t>(gridPtr - base);

        // GPU にデータをアップロード
        out.buffer.resize(out.bytes);
        out.buffer.upload(base, out.bytes);

        std::cout   << "[NanoVDB] loaded grid=" << floatGrid.getName() << "\n"
                    << "bytes = " << out.bytes << "\n"
                    << "bbox = (" 
                    << out.worldMin[0] << ", " << out.worldMin[1] << ", " << out.worldMin[2] << ")-("
                    << out.worldMax[0] << ", " << out.worldMax[1] << ", " << out.worldMax[2] << ")\n";

        return out;
    }
};


#endif // VDB_LOADER_HPP_