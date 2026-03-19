# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**nucrust** は Rust で実装される GPU 加速の原子核反応率計算フレームワーク。核天体物理学（Hauser-Feshbach 統計模型・R-matrix 理論）を対象とし、既存 Fortran コード（TALYS, AZURE2）に対して 50〜500 倍の高速化を目標とする。

**現状**: 設計・仕様・リサーチ完了済み。ソースコード実装はこれから（Phase 1 から開始）。

## ビルド・開発コマンド

### ワークスペース全体
```bash
cargo build --workspace                     # 全クレートビルド
cargo build --workspace --release           # リリースビルド
cargo test --workspace                      # 全ユニットテスト実行
cargo test --workspace --all-features       # 全機能有効でテスト（受け入れ基準）
cargo clippy --workspace -- -D warnings     # Lint（受け入れ基準: 警告ゼロ）
cargo fmt --all                             # フォーマット
```

### 個別クレートのテスト
```bash
cargo test -p nucrust-special               # 例: Coulomb 関数クレートのみ
cargo test -p nucrust-hf test_fe56_ngamma   # 例: 特定テスト関数のみ
```

### ベンチマーク
```bash
cargo bench -p nucrust-special              # Criterion ベンチマーク
cargo bench -p nucrust-special -- --save-baseline baseline
```

### GPU 機能（CUDA 環境必須）
```bash
cargo test --features cuda --workspace
cargo build --features cuda --release
```

### Python バインディング（Phase 4）
```bash
cd crates/nucrust-python && maturin develop  # 開発用インストール
cd crates/nucrust-python && maturin build --release
```

## クレート構成（Cargo ワークスペース）

```
nucrust/                    ← ワークスペースルート
├── crates/
│   ├── nucrust-core/       ← Nuclide型, SpinParity型, 単位系, エラー型
│   ├── nucrust-special/    ← Coulomb関数・Bessel関数（純Rust実装）
│   ├── nucrust-data/       ← RIPL-3 / REACLIB パーサー
│   ├── nucrust-optical/    ← 光学模型・透過係数計算（Numerov法）
│   ├── nucrust-hf/         ← Hauser-Feshbach 統計模型
│   ├── nucrust-rmatrix/    ← R-matrix 理論（Lane-Thomas / Brune 変換）
│   ├── nucrust-astro/      ← 天体物理反応率（MACS, SEF, S因子）
│   ├── nucrust-gpu/        ← GPU バックエンド（cudarc + CUDA C カーネル）
│   └── nucrust-python/     ← PyO3 + rust-numpy バインディング
└── src/                    ← nucrust 集約クレート・CLI（calc/batch/fit）
```

**クレート依存方向**: `nucrust-core` → `nucrust-special` → `nucrust-data` / `nucrust-optical` → `nucrust-hf` / `nucrust-rmatrix` → `nucrust-astro` → `nucrust-gpu` → `nucrust-python` / `nucrust`

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
- **R-matrix**: batched LU 分解（cuSOLVER `cusolverDnZgetrfBatched`）
- CUDA C カーネルは `hipify-clang` 互換設計（ROCm 移植を将来対応）
- 混合精度: FP32 + 反復精緻化（GPU/FP64 参照との相対誤差 < 10⁻⁵）

## テスト戦略

### 5層テストアーキテクチャ
1. **ユニットテスト**: 各クレート内の `#[cfg(test)]` モジュール
2. **精度検証テスト**: mpmath (50桁精度) / COULCC.f / GSL との数値比較（golden file）
3. **性質ベーステスト**: `proptest` で物理的制約（断面積 ≥ 0, 詳細釣り合い 等）を検証
4. **回帰ベンチマーク**: `criterion.rs` + `iai`（TALYS/AZURE2 golden file 比較）
5. **ファズテスト**: `cargo-fuzz`（パーサー・特殊関数の境界ケース）

### 精度要件（受け入れ基準）
| 要件 | 比較対象 | 許容誤差 |
|------|---------|---------|
| Coulomb 関数 | mpmath 50桁 | 相対誤差 < 10⁻¹² |
| 透過係数 | TALYS | 相対誤差 < 10⁻⁶ |
| HF 断面積 | TALYS | 相対誤差 < 10⁻⁶ |
| R-matrix | AZURE2 | 相対誤差 < 10⁻⁶ |
| GPU 混合精度 | FP64 参照 | 相対誤差 < 10⁻⁵ |

## 主要技術スタック

| カテゴリ | 採用ライブラリ | 理由 |
|---------|-------------|------|
| 線形代数 | `faer` | 純Rust、BLAS同等性能 |
| GPU | `cudarc` v0.19.3 | 最も成熟（CudaContext+CudaStream API） |
| CPU並列 | `rayon` | ワークスティーリング |
| SIMD | `wide` | stable Rust で使用可能 |
| テスト | `proptest` + `criterion` | 性質ベース + ベンチマーク |
| パーサー | `nom` | ゼロコピーの `&str` スライス方式 |
| Python | `PyO3` + `rust-numpy` + `maturin` | - |
| データ | `serde` + `hdf5-metno` | RIPL-3 / REACLIB / HDF5 |

## 実装ロードマップ

- **Phase 1 (M1-M6)**: `nucrust-core` 型定義 + `nucrust-special` Coulomb 関数 + `nucrust-data` RIPL-3 パーサー
- **Phase 2 (M7-M12)**: `nucrust-optical` + `nucrust-hf` + `nucrust-rmatrix` (CPU)
- **Phase 3 (M13-M18)**: `nucrust-gpu` GPU 加速 + ベンチマーク B1-B4 検証
- **Phase 4 (M19-M24)**: `nucrust-astro` + `nucrust-python` + Multi-GPU + ROCm 評価

## 設計ドキュメント

詳細な仕様・設計・リサーチ結果は `docs/` に格納:
- `docs/nucrust_srs.md` — Software Requirements Specification (SRS v0.2.1)
- `docs/design.md` — 詳細設計書 v1.0.0（クレート構成・型定義・アルゴリズム擬似コード）
- `docs/pre-research.md` — 既存コード分析・Rustエコシステム評価・GPU高速化見積もり
- `docs/reserch_RQ01-09.md` — Thompson-Barnett COULCC, RIPL-3フォーマット, cudarc API 等9つの技術的課題の解決
- `docs/background.md` — 理論背景・文献レビュー

**Phase 1 実装開始時は `docs/design.md` と `docs/reserch_RQ01-09.md` の RQ-01（Coulomb関数擬似コード）を最初に参照すること。**
