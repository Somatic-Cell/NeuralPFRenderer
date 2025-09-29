# Simple Renderer Using OptiX

研究や，検証につかうためのシンプルなレンダラ．\
OptiX を使用．

- [依存関係](#依存関係)
    1. [ダウンロード](#1-ダウンロード)
    1. [ビルド](#2-ビルド)
    1. [その他](#3-その他)
- [レンダラの機能](#レンダラの機能)

## 依存関係

- CUDA ( >= v.12.0)
- OptiX (>= v.8.0.0)

## 準備 (Windows 環境)

### 1. ダウンロード

本リポジトリをローカル環境にクローンする

```
git clone --recursive https://github.com/Somatic-Cell/SimpleRenderer.git 
```
サブモジュールを含めてクローンするために，```--recursive```オプションを指定．


### 2. ビルド

CMake を用いる．

### 3. その他

## レンダラの機能