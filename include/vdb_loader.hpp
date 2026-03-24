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
#include <cmath>

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
// Macrocell majorant grid
// -----------------------------------
struct MacrocellMajorantGrid {
    CUDABuffer  buffer;
    int3        baseCoord = make_int3(0);
    int3        dims      = make_int3(0);
    int         cellSizeVoxels  = 8;     // voxels / macrocell
    size_t      cellCount       = 0;
    size_t      nonZeroCellCount = 0;
    size_t      bytes           = 0;
    float       maxValue        = 0.0f;

    bool valid() const {return cellCount > 0 && bytes > 0; }
};

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
    MacrocellMajorantGrid macro;

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

    void setMacrocellCellSizeVoxels(int cellSize)
    {
        if(cellSize <= 0) {
            throw std::runtime_error("macrocell call size must be > 0.");
        }
        m_macrocellCellSizeVoxels = cellSize;
    }

    int getMacrocellCellSizeVoxels() const
    {
        return m_macrocellCellSizeVoxels;
    }
    
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

            NanoVDBGrid grid    = buildFromFloatGrid(*floatGrid, m_macrocellCellSizeVoxels);
            grid.name           = base->getName();
            grid.vdbType        = base->type();

            m_grids[grid.name] = std::move(grid);
        }

        file.close();

        if(m_grids.empty())
        {
            throw std::runtime_error("no FloatGrid found in VDB file: " + vdbPath);
        }

        // 明示的に選んでいない場合でも，安定した primary の名前を決める
        m_primaryGridName = chooseDefaultDensityLikeName();
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

            NanoVDBGrid grid    = buildFromFloatGrid(*floatGrid, m_macrocellCellSizeVoxels);
            grid.name           = gridName;
            grid.vdbType        = base->type();

            m_grids[grid.name] = std::move(grid);
        }

        file.close();

        if(m_grids.empty()) {
            throw std::runtime_error("No grids loaded from VDB file: " + vdbPath); 
        }

        m_primaryGridName = chooseDefaultDensityLikeName();

    }

    // ファイル内から density-like な FloatGrid を 1 本選んで読む
    void loadDensityLikeFloatGrid(const std::string& vdbPath)
    {
        ensureOpenVDBInitialized();

        openvdb::io::File file(vdbPath);
        file.open();

        std::vector<std::string> floatGridNames;
        auto grids = file.getGrids();
        for (auto& base : *grids) {
            auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
            if (!floatGrid) continue;
            floatGridNames.push_back(base->getName());
        }
        file.close();

        if (floatGridNames.empty()) {
            throw std::runtime_error("no FloatGrid found in VDB file: " + vdbPath);
        }

        const std::string selected = chooseDensityLikeNameFromCandidates(floatGridNames);
        loadSelectedFloatGrids(vdbPath, { selected });
        m_primaryGridName = selected;
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

    const NanoVDBGrid& getPrimaryGrid() const
    {
        if (m_primaryGridName.empty()) {
            throw std::runtime_error("Primary grid is not selected.");
        }
        return getGrid(m_primaryGridName);
    }

    const std::string& getPrimaryGridName() const
    {
        if (m_primaryGridName.empty()) {
            throw std::runtime_error("Primary grid name is empty.");
        }
        return m_primaryGridName;
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
    //     const std::vector<std::string> preferred = {"density", "dens", "cloud", "fog", "volume"};
        
    //     // 候補を探す
    //     for (const auto& want : preferred) {
    //         for(const auto& kv : m_grids){
    //             if(isEquals(kv.first, want)) return kv.first;
    //         }
    //     }

    //     // なければ一番最初を返す
    //     return m_grids.begin()->first;
        return chooseDensityLikeNameFromCandidates(getListGridNames());
    }

    std::string m_path;
    std::unordered_map<std::string, NanoVDBGrid> m_grids;
    std::string m_primaryGridName;



private:
    int m_macrocellCellSizeVoxels = 8;

    static bool isEquals(const std::string& a, const std::string& b){
        if(a.size() != b.size()) return false;
        for(size_t i = 0; i < a.size(); ++i){
            if(std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
        }
        return true;
    }


    static bool containsIgnoreCase(const std::string& text, const std::string& needle)
    {
        auto lower = [](const std::string& s) {
            std::string out = s;
            for (char& c : out) c = (char)std::tolower((unsigned char)c);
            return out;
        };
        const std::string lhs = lower(text);
        const std::string rhs = lower(needle);
        return lhs.find(rhs) != std::string::npos;
    }

    static std::string chooseDensityLikeNameFromCandidates(std::vector<std::string> names)
    {
        if (names.empty()) {
            throw std::runtime_error("No candidate grid names.");
        }

        std::sort(names.begin(), names.end());

        const std::vector<std::string> preferredExact = {
            "density", "dens", "cloud", "fog", "volume"
        };
        for (const auto& want : preferredExact) {
            for (const auto& name : names) {
                if (isEquals(name, want)) return name;
            }
        }

        const std::vector<std::string> preferredContains = {
            "density", "dens", "cloud", "fog", "volume"
        };
        for (const auto& want : preferredContains) {
            for (const auto& name : names) {
                if (containsIgnoreCase(name, want)) return name;
            }
        }

        // fallback は unordered_map の begin() ではなく、ソート済み先頭で安定化
        return names.front();
    }

    static size_t flatten3D(int x, int y, int z, const int3& dims)
    {
        return (static_cast<size_t>(z) * static_cast<size_t>(dims.y)
              + static_cast<size_t>(y)) * static_cast<size_t>(dims.x)
              + static_cast<size_t>(x);
    }

    static void updateMajorantCell(
        std::vector<float>& majorants,
        const int3& dims,
        int mx, int my, int mz,
        float value)
    {
        if (mx < 0 || my < 0 || mz < 0) return;
        if (mx >= dims.x || my >= dims.y || mz >= dims.z) return;
        const size_t idx = flatten3D(mx, my, mz, dims);
        majorants[idx] = std::max(majorants[idx], value);
    }

    static MacrocellMajorantGrid buildMacrocellMajorantFromFloatGrid(
        const openvdb::FloatGrid& floatGrid,
        const openvdb::CoordBBox& bbox,
        int cellSizeVoxels)
    {
        if (cellSizeVoxels <= 0) {
            throw std::runtime_error("cellSizeVoxels must be > 0.");
        }

        MacrocellMajorantGrid out;
        out.cellSizeVoxels = cellSizeVoxels;

        const openvdb::Coord bMin = bbox.min();
        const openvdb::Coord bMax = bbox.max();

        out.baseCoord = make_int3(bMin.x(), bMin.y(), bMin.z());

        const int3 voxelDims = make_int3(
            bMax.x() - bMin.x() + 1,
            bMax.y() - bMin.y() + 1,
            bMax.z() - bMin.z() + 1
        );

        out.dims = make_int3(
            (voxelDims.x + cellSizeVoxels - 1) / cellSizeVoxels,
            (voxelDims.y + cellSizeVoxels - 1) / cellSizeVoxels,
            (voxelDims.z + cellSizeVoxels - 1) / cellSizeVoxels
        );

        out.cellCount =
            static_cast<size_t>(out.dims.x) *
            static_cast<size_t>(out.dims.y) *
            static_cast<size_t>(out.dims.z);

        if (out.cellCount == 0) {
            throw std::runtime_error("macrocell grid is empty.");
        }

        std::vector<float> hostMajorants(out.cellCount, 0.0f);

        // active voxel のみ走査して，各 macrocell の max density を構築する。
        // trilinear 補間の upper-neighbor を保守的に含めるため，
        // macrocell 先頭 voxel に位置する active voxel は 1 つ前の cell にも反映する。
        for (auto it = floatGrid.cbeginValueOn(); it; ++it) {
            const float v = static_cast<float>(*it);
            if (!(v > 0.0f)) continue;

            const openvdb::Coord ijk = it.getCoord();

            const int lx = ijk.x() - out.baseCoord.x;
            const int ly = ijk.y() - out.baseCoord.y;
            const int lz = ijk.z() - out.baseCoord.z;

            if (lx < 0 || ly < 0 || lz < 0) continue;

            const int mx = lx / cellSizeVoxels;
            const int my = ly / cellSizeVoxels;
            const int mz = lz / cellSizeVoxels;

            const bool touchPrevX = (lx % cellSizeVoxels) == 0;
            const bool touchPrevY = (ly % cellSizeVoxels) == 0;
            const bool touchPrevZ = (lz % cellSizeVoxels) == 0;

            const int oxMax = touchPrevX ? 1 : 0;
            const int oyMax = touchPrevY ? 1 : 0;
            const int ozMax = touchPrevZ ? 1 : 0;

            for (int oz = 0; oz <= ozMax; ++oz) {
                for (int oy = 0; oy <= oyMax; ++oy) {
                    for (int ox = 0; ox <= oxMax; ++ox) {
                        updateMajorantCell(hostMajorants, out.dims, mx - ox, my - oy, mz - oz, v);
                    }
                }
            }
        }

        out.maxValue = 0.0f;
        out.nonZeroCellCount = 0;
        for (float v : hostMajorants) {
            out.maxValue = std::max(out.maxValue, v);
            if (v > 0.0f) ++out.nonZeroCellCount;
        }

        out.bytes = hostMajorants.size() * sizeof(float);
        out.buffer.resize(out.bytes);
        out.buffer.upload(hostMajorants.data(), hostMajorants.size());

        return out;
    }
    
    static NanoVDBGrid buildFromFloatGrid(openvdb::FloatGrid& floatGrid, int macrocellCellSizeVoxels){
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
        // CPU で macrocell majorant grid を構築して先に GPU へアップロード
        out.macro = buildMacrocellMajorantFromFloatGrid(floatGrid, bbox, macrocellCellSizeVoxels);

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
        auto* hGrid = hostHandle.grid<float>(); // grid の先頭
        if(!hGrid){
            throw std::runtime_error("Converted NanoVDB is not NanoGir<float>");
        }

        // min , max と bbox を再計算
        nanovdb::tools::updateGridStats(hGrid, nanovdb::tools::StatsMode::All);

        // 作成できたかどうか確認
        if(!hGrid->hasMinMax() || !hGrid->hasBBox()){
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
                    << "macrocell cellSize = " << out.macro.cellSizeVoxels << "\n"
                    << "macrocell dims = ("
                    << out.macro.dims.x << ", " << out.macro.dims.y << ", " << out.macro.dims.z << ")\n"
                    << "macrocell nonZero / total = "
                    << out.macro.nonZeroCellCount << " / " << out.macro.cellCount << "\n"
                    << "macrocell max density = " << out.macro.maxValue << "\n"
                    << "bbox = (" 
                    << out.worldMin[0] << ", " << out.worldMin[1] << ", " << out.worldMin[2] << ")-("
                    << out.worldMax[0] << ", " << out.worldMax[1] << ", " << out.worldMax[2] << ")\n";

        return out;
    }
};


#endif // VDB_LOADER_HPP_