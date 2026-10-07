# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**nucrust** は Rust で実装される GPU 加速の原子核反応率計算フレームワーク。核天体物理学（Hauser-Feshbach 統計模型・R-matrix 理論）を対象とし、既存 Fortran コード（TALYS, AZURE2）に対して 50〜500 倍の高速化を目標とする。

**現状**: Phase 1〜4 の実装完了（v0.1.0 開発中）。全クレートが実装済みで、残作業は外部コード（TALYS/AZURE2）との精度検証ベンチマーク・一部 CLI 機能（HDF5/REACLIB エクスポート等）。詳細は `CHANGELOG.md` と `task.md` を参照。

## ビルド・開発コマンド

### ワークスペース全体
```bash
cargo build --workspace                     # 全クレートビルド
cargo build --workspace --release           # リリースビルド
cargo test --workspace                      # 全ユニットテスト実行
cargo test --workspace --features nucrust-data/hdf5_io,nucrust/parallel,nucrust/simd
                                            # cuda 以外の全機能でテスト（受け入れ基準・CI と同一）
cargo clippy --workspace --all-targets -- -D warnings  # Lint（受け入れ基準: 警告ゼロ）
cargo fmt --all                             # フォーマット
```

### 個別クレートのテスト
```bash
cargo test -p nucrust-special               # 例: Coulomb 関数クレートのみ
cargo test -p nucrust-hf fe56               # 例: テスト名でフィルタ
```

### CPU 並列化（macOS では主要な高速化経路）
```bash
cargo build --release -p nucrust --features "parallel simd"
```

### GPU 機能（CUDA 環境必須。ビルド時に nvcc が必要）
```bash
cargo test -p nucrust-gpu --features cuda
cargo build -p nucrust-gpu --features cuda --release
```

### Python バインディング（abi3: CPython ≥ 3.9。環境管理は uv）
```bash
cd crates/nucrust-python && maturin develop  # 開発用インストール
cd crates/nucrust-python && maturin build --release
```

### ドキュメント（mdBook ユーザーガイド）
```bash
mdbook build                                # book/build に HTML 生成
mdbook test                                 # 本文中の Rust コード例をコンパイル検証
```

## リポジトリ構成

```
nucrust/                    ← ワークスペースルート
├── crates/
│   ├── nucrust-core/       ← Nuclide型, SpinParity型, 単位系, エラー型, Wigner記号
│   ├── nucrust-special/    ← Coulomb関数・Bessel関数（純Rust実装）
│   ├── nucrust-data/       ← RIPL-3 / REACLIB パーサー, TOML設定, HDF5 I/O
│   ├── nucrust-optical/    ← 光学模型・透過係数計算（Numerov法, 結合チャンネル）
│   ├── nucrust-hf/         ← Hauser-Feshbach 統計模型（WFC 補正含む）
│   ├── nucrust-rmatrix/    ← R-matrix 理論（Lane-Thomas / Brune 変換）
│   ├── nucrust-astro/      ← 天体物理反応率（MACS, SEF, REACLIBフィット）
│   ├── nucrust-gpu/        ← GPU バックエンド（cudarc + CUDA C カーネル）
│   ├── nucrust-python/     ← PyO3 + rust-numpy バインディング
│   └── nucrust/            ← 集約クレート・CLI バイナリ（calc/batch/fit/export）
├── kernels/                ← CUDA C カーネルソース
├── book/                   ← mdBook ユーザーガイド（book.toml はルート）
├── docs/                   ← 設計時ドキュメント（凍結・歴史的資料）
├── data/                   ← テスト用サンプルデータ（RIPL-3 / REACLIB）
├── scripts/                ← mpmath 参照値生成スクリプト等
└── tests/reference_data/   ← 精度検証用 golden file
```

**クレート依存方向**（`nucrust-core` が最下層）:

- `nucrust-core` ← 全クレートの基盤（依存なし）
- `nucrust-special` ← core
- `nucrust-data` ← core
- `nucrust-optical` ← core, special, data
- `nucrust-hf` ← core, optical, data
- `nucrust-rmatrix` ← core, special, data
- `nucrust-astro` ← core, data, hf, rmatrix
- `nucrust-gpu` ← core, special（独立した計算バックエンド。astro には依存しない）
- `nucrust-python` / `nucrust`（CLI） ← 上記の主要クレート群

## アーキテクチャ設計

### 設計原則
- **型駆動設計**: `Nuclide`, `SpinParity`, `ReactionChannel` 等の強い型でドメイン概念を表現
- **ゼロコスト抽象化**: 物理モデルはトレイトで定義し、実装選択はモノモーフィズムで行う
- **バックエンド透過性**: `OpticalModel` / `LevelDensity` / `GammaStrengthFn` トレイトにより CPU/GPU 実装を切り替え可能

### 主要計算フロー
1. `nucrust-data`: RIPL-3 核準位データ・光学模型パラメータ読み込み
2. `nucrust-special`: Coulomb 波動関数（Thompson-Barnett COULCC アルゴリズム）計算
3. `nucrust-optical`: Numerov 法で光学模型を解き、透過係数 $T_{lj}$ を算出
4. `nucrust-hf`: Hauser-Feshbach 公式で断面積を合算（WFC 補正含む）
5. `nucrust-astro`: Maxwell-Boltzmann 積分で MACS / REACLIB 7パラメータフィット出力
6. `nucrust-gpu`: 上記計算をバッチ GPU 実行（cudarc + CUDA C カーネル）

### GPU 設計（`nucrust-gpu`）
- **バッチ Numerov**: エネルギー点 × 部分波の 2D グリッドを GPU で並列実行（各スレッドが独立）
- **HF 合算**: warp-reduce による効率的なチャンネル合算
- **R-matrix**: batched LU 分解
- CUDA C カーネルは `hipify-clang` 互換設計（ROCm 移植を将来対応）
- 混合精度: FP32 + 反復精緻化（GPU/FP64 参照との相対誤差 < 10⁻⁵）

## テスト戦略

### 現在実装されているテスト
1. **ユニットテスト**: 各クレート内の `#[cfg(test)]` モジュール
2. **精度検証テスト**: mpmath（高精度参照値, `scripts/` で生成）との数値比較
   （golden file は `tests/reference_data/`, 検証テストは `crates/nucrust-special/tests/`）
3. **性質ベーステスト**: `proptest` で物理的制約（断面積 ≥ 0, 詳細釣り合い 等）を検証
   （nucrust-core / nucrust-optical / nucrust-hf の dev-dependencies）

### 未導入（計画のみ）
- `criterion` 回帰ベンチマーク（`nucrust-special` の `benches/coulomb.rs` のみ作成済み。他クレートはベンチターゲット未作成）
- `cargo-fuzz` ファズテスト
- TALYS / AZURE2 との比較検証（task.md の未完了項目 ACC-02〜04）

### 精度要件（受け入れ基準）
| 要件 | 比較対象 | 許容誤差 |
|------|---------|---------|
| Coulomb 関数 | mpmath 高精度 | 相対誤差 < 10⁻¹² |
| 透過係数 | TALYS | 相対誤差 < 10⁻⁶ |
| HF 断面積 | TALYS | 相対誤差 < 10⁻⁶ |
| R-matrix | AZURE2 | 相対誤差 < 10⁻⁶ |
| GPU 混合精度 | FP64 参照 | 相対誤差 < 10⁻⁵ |

## 主要技術スタック

| カテゴリ | 採用ライブラリ | 理由 |
|---------|-------------|------|
| 線形代数 | `faer` | 純Rust、BLAS同等性能 |
| GPU | `cudarc` v0.16 | CudaContext+CudaStream API |
| CPU並列 | `rayon` | ワークスティーリング |
| SIMD | `wide` | stable Rust で使用可能 |
| テスト | `proptest` + `criterion` | 性質ベース + ベンチマーク |
| パーサー | 手書き `&str` スライス方式 | 固定カラム形式にはコンビネータ不要（nom は不採用） |
| Python | `PyO3` + `rust-numpy` + `maturin` | - |
| データ | `serde` + `hdf5-metno` | RIPL-3 / REACLIB / HDF5 |

## ドキュメント

ユーザー向けドキュメントの正本は **`book/`（mdBook）と rustdoc**。変更時はコードと同時に更新すること。

`docs/` は実装開始前の設計時ドキュメント（**凍結済み・歴史的資料**）。現状の仕様とは乖離があるため、現在の挙動の根拠として引用しないこと:
- `docs/nucrust_srs.md` — Software Requirements Specification (SRS v0.2.1)
- `docs/design.md` — 詳細設計書（クレート構成・型定義・アルゴリズム擬似コード）
- `docs/pre-research.md` — 既存コード分析・Rustエコシステム評価・GPU高速化見積もり
- `docs/research_RQ01-09.md` — Thompson-Barnett COULCC, RIPL-3フォーマット, cudarc API 等9つの技術的課題の解決
- `docs/background.md` — 理論背景・文献レビュー

実装の現状把握には `CHANGELOG.md`（実装済み機能）と `task.md`（タスク完了状況）を参照すること。
