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
    
        static void updateMajorantRange(
        std::vector<float>& majorants,
        const int3& dims,
        int mx0, int my0, int mz0,
        int mx1, int my1, int mz1,
        float value)
    {
        mx0 = std::max(mx0, 0);
        my0 = std::max(my0, 0);
        mz0 = std::max(mz0, 0);
    
        mx1 = std::min(mx1, dims.x - 1);
        my1 = std::min(my1, dims.y - 1);
        mz1 = std::min(mz1, dims.z - 1);
    
        if (mx0 > mx1 || my0 > my1 || mz0 > mz1) return;
    
        for (int mz = mz0; mz <= mz1; ++mz) {
            for (int my = my0; my <= my1; ++my) {
                for (int mx = mx0; mx <= mx1; ++mx) {
                    const size_t idx = flatten3D(mx, my, mz, dims);
                    majorants[idx] = std::max(majorants[idx], value);
                }
            }
        }
    }

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

            openvdb::CoordBBox valueBBox;
            if (!it.getBoundingBox(valueBBox) || valueBBox.empty()) continue;

            const openvdb::Coord vMin = valueBBox.min();
            const openvdb::Coord vMax = valueBBox.max();

            // bbox 基準のローカル voxel 座標
            const int lx0 = vMin.x() - out.baseCoord.x;
            const int ly0 = vMin.y() - out.baseCoord.y;
            const int lz0 = vMin.z() - out.baseCoord.z;

            const int lx1 = vMax.x() - out.baseCoord.x;
            const int ly1 = vMax.y() - out.baseCoord.y;
            const int lz1 = vMax.z() - out.baseCoord.z;

            // build 対象 bbox の外なら無視
            if (lx1 < 0 || ly1 < 0 || lz1 < 0) continue;
            if (lx0 >= voxelDims.x || ly0 >= voxelDims.y || lz0 >= voxelDims.z) continue;

            // 対象 bbox に clamp
            const int clx0 = std::max(lx0, 0);
            const int cly0 = std::max(ly0, 0);
            const int clz0 = std::max(lz0, 0);

            const int clx1 = std::min(lx1, voxelDims.x - 1);
            const int cly1 = std::min(ly1, voxelDims.y - 1);
            const int clz1 = std::min(lz1, voxelDims.z - 1);

            int mx0 = clx0 / cellSizeVoxels;
            int my0 = cly0 / cellSizeVoxels;
            int mz0 = clz0 / cellSizeVoxels;

            const int mx1 = clx1 / cellSizeVoxels;
            const int my1 = cly1 / cellSizeVoxels;
            const int mz1 = clz1 / cellSizeVoxels;

            // 現行実装の "touchPrev*" を bbox の最小端へ一般化
            if ((clx0 % cellSizeVoxels) == 0) --mx0;
            if ((cly0 % cellSizeVoxels) == 0) --my0;
            if ((clz0 % cellSizeVoxels) == 0) --mz0;

            updateMajorantRange(hostMajorants, out.dims, mx0, my0, mz0, mx1, my1, mz1, v);
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