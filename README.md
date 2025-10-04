# PhotonicRT v.0.1

フォトニック結晶をレイトレーシングするためのレンダラ．\
NVIDIA CUDA / OptiX を使用．

- [依存関係](#依存関係)
- [準備](#準備)
    1. [ダウンロード](#1-ダウンロード)
    1. [ビルド](#2-ビルド)
    1. [その他](#3-その他)
- [実行](#実行)
- [レンダラの機能](#レンダラの機能)

## 依存関係

### 要事前インストール
#### GPU レンダリング用 SDK
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) ( >= v.12.0)
- [NVIDIA OptiX](https://developer.nvidia.com/designworks/optix/download) (v.8.0.0 or v9.0.0)
#### OpenVDB ビルド用
- [tbb](https://github.com/uxlfoundation/oneTBB/releases?utm_source=chatgpt.com)
- [zlib](https://zlib.net/?utm_source=chatgpt.com)
- [openEXR](https://github.com/AcademySoftwareFoundation/openexr?utm_source=chatgpt.com)
- [Blosc](https://github.com/Blosc/c-blosc?utm_source=chatgpt.com)
- [Boost](https://www.boost.org/?utm_source=chatgpt.com)

vcpkg を使用できる場合は，PowerShell から
```
$ vcpkg install tbb blosc openexr zlib boost-iostreams 
```
とインストールしてしまうのが一番手っ取り早い．\
または，上記のリンクから逐一インストールする．

### その他使用する外部ライブラリ　（個別にダウンロードする必要なし．ライセンス管理のために列挙）
- [Open Asset Import Library (assimp)](https://github.com/assimp/assimp?tab=readme-ov-file#open-asset-import-library-assimp) ：アセットの読み込み
- [DirectXTex texture processing library](https://github.com/microsoft/DirectXTex)：``` .DDS ``` ファイルの読み込み
- [GLFW](https://github.com/glfw/glfw)：ウィンドウの描画
- [Dear ImGui](https://github.com/ocornut/imgui)：GUI の描画
- [stb](https://github.com/nothings/stb)：テクスチャの読み込み，出力画像の保存
- [JSON for Modern C++](https://github.com/nlohmann/json?tab=readme-ov-file)：シーンファイルの読み書き
- [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb)：ボリュームデータの読み込み
- NanoVDB (OpenVDB に内包)：GPU 上でのボリュームデータのサンプリング

## 準備 (Windows 環境)

### 1. ダウンロード

コマンドプロンプトなどを開き，作業したいディレクトリ上で，本リポジトリをローカル環境にクローンする：

```
$ git clone --recursive https://github.com/Somatic-Cell/PhotonicRT.git 
```
【注意】 サブモジュールを含めてクローンするために，```--recursive``` オプションを指定． \
これを指定すれば，[使用する外部ライブラリ](#使用する外部ライブラリ) を個別にダウンロードする必要がない．


### 2. ビルド

#### プロジェクトのビルド方法

[CMake](https://cmake.org/download/) を用いる．

<details><summary>CMake のインストール方法</summary>

[リンク](https://cmake.org/download/) 先から Windows x64 版をインストールすること．\
インストールできているかどうかを確認したい場合は
```
$ cmake
```
と実行すればわかる．

また，パスが通っているかどうかを確認したければ

```
$ where cmake
```
とすれば確認できる．

</details>

以下の手順に従ってビルドする（Release ビルドの場合）：
```
$ mkdir build
$ cd build
$ cmake .. 
$ cmake --build . --config Release --verbose
$ cd ..
```
クリーンビルドをしたい場合は，上記を実行する前に，以下を実行して build ディレクトリを削除する：
```
$ rmdir /S build
```
これらのコマンドを毎回打ち込むのは面倒くさい．\
上記一式のコマンドは，``` build.bat ``` ファイルにまとめて記述してあるので，プロジェクトのルートディレクトリで，単に
```
$ build.bat
```
と実行すれば全て自動で行ってくれる．

#### バージョン指定
<details><summary> GPU のアーキテクチャ 指定</summary>

``` CMakeLists.txt ``` では，[CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus) を ``` CUDA_CC ``` で指定する．
実行する環境に合わせて，適宜変更すること．

Compute capability の例

| 世代 | 機種名 | Compute capability |
| --- | --- | :---: |
| Ampere | GeForce RTX 30系統 | 86 |
| Ada | GeForce RTX 40系統 | 89 |

</details>


<details><summary> (Release / Debug) ビルドのバージョン指定</summary>

```Debug``` / ```Release``` ビルドを切り替えたい場合は，単に上記のビルドで
```
$ cmake --build . --config Release --verbose
```
を
```
$ cmake --build . --config Debug --verbose
```
に置き換えればよい．
実行時間を計測する際は，必ず ```Release``` を指定すること．

</details>

<details><summary>OptiX のバージョン変更</summary>

OptiX のバージョンを変更したい場合は，プロジェクトのルートディレクトリにある 
 ```CMakeLists.txt``` を編集する．
デフォルトでは OptiX 9.0.0 で動作するようになっているが，変更したい場合は，

```
find_package(OptiX9)

if(OptiX9_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX9_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 9.0.0 not found.")
endif()
```
あたりの，OptiX9 を OptiX(バージョン) に変更すればよい．
対応しているバージョンは，```utils/cmake/FindOptiX*.cmake``` というファイルがあるもの．
</details>


### 3. その他

## 実行



## レンダラの機能

## 実験開発用メモ

### デバッグ
